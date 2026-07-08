// Package chroma provides a VectorStore backed by a Chroma vector database.
// It communicates via Chroma's REST API using only the standard library.
package chroma

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/grafaelw/golangchain/embeddings"
	"github.com/grafaelw/golangchain/schema"
	"github.com/grafaelw/golangchain/vectorstore"
)

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

// Option configures the Chroma store.
type Option func(*Store)

// WithEmbeddingFunction disables local embedding so Chroma's server-side
// embedding function is used instead. The embedder passed to New is ignored
// when this is true.
func WithEmbeddingFunction(fnName string) Option {
	return func(s *Store) { s.embeddingFn = fnName }
}

// WithHTTPClient overrides the HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(s *Store) { s.client = client }
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

// Store implements vectorstore.VectorStore backed by Chroma.
type Store struct {
	baseURL      string
	collectionID string
	collection   string
	embeddingFn  string
	client       *http.Client
	embedder     embeddings.Embedder
}

// New creates a Chroma-backed vector store. If the named collection does not
// exist it is created automatically.
//
//	store, err := chroma.New(
//	    "http://localhost:8000",
//	    "my-collection",
//	    embedder,
//	)
func New(baseURL, collectionName string, embedder embeddings.Embedder, opts ...Option) (*Store, error) {
	if baseURL == "" {
		return nil, errors.New("chroma: base URL is required")
	}
	if collectionName == "" {
		return nil, errors.New("chroma: collection name is required")
	}
	s := &Store{
		baseURL:    strings.TrimSuffix(baseURL, "/"),
		collection: collectionName,
		client:     &http.Client{Timeout: 30 * time.Second},
		embedder:   embedder,
	}
	for _, o := range opts {
		o(s)
	}
	if err := s.ensureCollection(context.Background()); err != nil {
		return nil, fmt.Errorf("chroma: init: %w", err)
	}
	return s, nil
}

var _ vectorstore.VectorStore = (*Store)(nil)

// ---------------------------------------------------------------------------
// VectorStore implementation
// ---------------------------------------------------------------------------

// AddDocuments embeds and indexes documents.
func (s *Store) AddDocuments(ctx context.Context, docs []schema.Document) error {
	if len(docs) == 0 {
		return nil
	}

	ids := make([]string, len(docs))
	metadatas := make([]map[string]any, len(docs))
	documents := make([]string, len(docs))

	for i, doc := range docs {
		id := fmt.Sprintf("doc_%d", i)
		if v, ok := doc.Metadata["id"]; ok {
			id = fmt.Sprint(v)
		}
		ids[i] = id
		documents[i] = doc.PageContent
		metadatas[i] = doc.Metadata
	}

	bodyMap := map[string]any{
		"ids":       ids,
		"documents": documents,
		"metadatas": metadatas,
	}

	if s.embeddingFn == "" {
		texts := make([]string, len(docs))
		for i, doc := range docs {
			texts[i] = doc.PageContent
		}
		vectors, err := s.embedder.EmbedDocuments(ctx, texts)
		if err != nil {
			return fmt.Errorf("chroma: embed: %w", err)
		}
		if len(vectors) != len(docs) {
			return fmt.Errorf("chroma: expected %d vectors, got %d", len(docs), len(vectors))
		}
		bodyMap["embeddings"] = vectors
	}

	body, _ := json.Marshal(bodyMap)
	url := fmt.Sprintf("/api/v1/collections/%s/add", s.collectionID)
	resp, err := s.request(ctx, http.MethodPost, url, body)
	if err != nil {
		return fmt.Errorf("chroma: add: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("chroma: add: status %d: %s", resp.StatusCode, string(b))
	}
	return nil
}

// SimilaritySearch embeds the query and returns the k nearest documents.
func (s *Store) SimilaritySearch(ctx context.Context, query string, k int) ([]schema.Document, error) {
	vec, err := s.embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("chroma: embed query: %w", err)
	}
	return s.SimilaritySearchByVector(ctx, vec, k)
}

// SimilaritySearchByVector returns the k nearest documents to a query vector.
func (s *Store) SimilaritySearchByVector(ctx context.Context, vector []float64, k int) ([]schema.Document, error) {
	body, _ := json.Marshal(map[string]any{
		"query_embeddings": [][]float64{vector},
		"n_results":        k,
	})
	url := fmt.Sprintf("/api/v1/collections/%s/query", s.collectionID)
	resp, err := s.request(ctx, http.MethodPost, url, body)
	if err != nil {
		return nil, fmt.Errorf("chroma: query: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("chroma: query: status %d: %s", resp.StatusCode, string(b))
	}

	var result struct {
		IDs       [][]string         `json:"ids"`
		Documents [][]string         `json:"documents"`
		Metadatas [][]map[string]any `json:"metadatas"`
		Distances [][]float64        `json:"distances"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("chroma: query: decode: %w", err)
	}

	docs := make([]schema.Document, 0)
	for setIdx := range result.IDs {
		for i, id := range result.IDs[setIdx] {
			content := ""
			if setIdx < len(result.Documents) && i < len(result.Documents[setIdx]) {
				content = result.Documents[setIdx][i]
			}
			score := 0.0
			if setIdx < len(result.Distances) && i < len(result.Distances[setIdx]) {
				score = 1.0 - result.Distances[setIdx][i]
			}
			md := map[string]any{"id": id}
			if setIdx < len(result.Metadatas) && i < len(result.Metadatas[setIdx]) {
				md = result.Metadatas[setIdx][i]
				md["id"] = id
			}
			docs = append(docs, schema.Document{
				PageContent: content,
				Score:       score,
				Metadata:    md,
			})
		}
	}
	return docs, nil
}

// Delete removes documents by ID from Chroma.
func (s *Store) Delete(ctx context.Context, ids []string) error {
	body, _ := json.Marshal(map[string]any{"ids": ids})
	url := fmt.Sprintf("/api/v1/collections/%s/delete", s.collectionID)
	resp, err := s.request(ctx, http.MethodPost, url, body)
	if err != nil {
		return fmt.Errorf("chroma: delete: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("chroma: delete: status %d: %s", resp.StatusCode, string(b))
	}
	return nil
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

func (s *Store) ensureCollection(ctx context.Context) error {
	metadata := map[string]any{}
	if s.embeddingFn != "" {
		metadata["hnsw:space"] = "cosine"
	}
	getURL := fmt.Sprintf("/api/v1/collections/%s", s.collection)
	resp, err := s.request(ctx, http.MethodGet, getURL, nil)
	if err == nil && resp.StatusCode < 400 {
		defer func() { _ = resp.Body.Close() }()
		var col struct {
			ID string `json:"id"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&col); err == nil && col.ID != "" {
			s.collectionID = col.ID
			return nil
		}
	}
	if resp != nil {
		_ = resp.Body.Close()
	}

	body, _ := json.Marshal(map[string]any{
		"name":     s.collection,
		"metadata": metadata,
	})
	resp, err = s.request(ctx, http.MethodPost, "/api/v1/collections", body)
	if err != nil {
		return fmt.Errorf("chroma: create collection: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("chroma: create collection: status %d: %s", resp.StatusCode, string(b))
	}
	var col struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&col); err != nil {
		return fmt.Errorf("chroma: create collection: decode: %w", err)
	}
	s.collectionID = col.ID
	return nil
}

func (s *Store) request(ctx context.Context, method, path string, body []byte) (*http.Response, error) {
	var r io.Reader
	if body != nil {
		r = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, s.baseURL+path, r)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	return s.client.Do(req)
}
