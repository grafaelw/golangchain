// Package pinecone provides a VectorStore backed by Pinecone's serverless
// vector database. It communicates via Pinecone's REST API using only the
// standard library.
package pinecone

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

// Option configures the Pinecone store.
type Option func(*Store)

// WithNamespace sets the Pinecone namespace (default "").
func WithNamespace(ns string) Option {
	return func(s *Store) { s.namespace = ns }
}

// WithDimension sets the expected vector dimension (default 1536).
func WithDimension(dim int) Option {
	return func(s *Store) { s.dimension = dim }
}

// WithHTTPClient overrides the HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(s *Store) { s.client = client }
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

// Store implements vectorstore.VectorStore backed by Pinecone.
type Store struct {
	apiKey    string
	baseURL   string
	namespace string
	dimension int
	client    *http.Client
	embedder  embeddings.Embedder
}

// New creates a Pinecone-backed vector store.
//
//	store, err := pinecone.New(
//	    os.Getenv("PINECONE_API_KEY"),
//	    "https://myindex-abc123.svc.us-east1-aws.pinecone.io",
//	    embedder,
//	    pinecone.WithDimension(1536),
//	)
func New(apiKey, baseURL string, embedder embeddings.Embedder, opts ...Option) (*Store, error) {
	if apiKey == "" {
		return nil, errors.New("pinecone: API key is required")
	}
	if baseURL == "" {
		return nil, errors.New("pinecone: base URL is required")
	}
	s := &Store{
		apiKey:    apiKey,
		baseURL:   strings.TrimSuffix(baseURL, "/"),
		dimension: 1536,
		embedder:  embedder,
		client:    &http.Client{Timeout: 30 * time.Second},
	}
	for _, o := range opts {
		o(s)
	}
	return s, nil
}

var _ vectorstore.VectorStore = (*Store)(nil)

// ---------------------------------------------------------------------------
// VectorStore implementation
// ---------------------------------------------------------------------------

// AddDocuments embeds and upserts documents into Pinecone.
func (s *Store) AddDocuments(ctx context.Context, docs []schema.Document) error {
	if len(docs) == 0 {
		return nil
	}

	texts := make([]string, len(docs))
	for i, doc := range docs {
		texts[i] = doc.PageContent
	}
	vectors, err := s.embedder.EmbedDocuments(ctx, texts)
	if err != nil {
		return fmt.Errorf("pinecone: embed: %w", err)
	}
	if len(vectors) != len(docs) {
		return fmt.Errorf("pinecone: expected %d vectors, got %d", len(docs), len(vectors))
	}

	type vector struct {
		ID       string         `json:"id"`
		Values   []float64      `json:"values"`
		Metadata map[string]any `json:"metadata,omitempty"`
	}

	vs := make([]vector, len(docs))
	for i, doc := range docs {
		id := fmt.Sprintf("doc_%d", i)
		if v, ok := doc.Metadata["id"]; ok {
			id = fmt.Sprint(v)
		}
		md := make(map[string]any)
		md["content"] = doc.PageContent
		for k, v := range doc.Metadata {
			md[k] = v
		}
		vs[i] = vector{ID: id, Values: vectors[i], Metadata: md}
	}

	body, _ := json.Marshal(map[string]any{
		"vectors":   vs,
		"namespace": s.namespace,
	})
	resp, err := s.request(ctx, http.MethodPost, "/vectors/upsert", body)
	if err != nil {
		return fmt.Errorf("pinecone: upsert: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("pinecone: upsert: status %d: %s", resp.StatusCode, string(b))
	}
	return nil
}

// SimilaritySearch embeds the query and returns the k nearest documents.
func (s *Store) SimilaritySearch(ctx context.Context, query string, k int) ([]schema.Document, error) {
	vec, err := s.embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("pinecone: embed query: %w", err)
	}
	return s.SimilaritySearchByVector(ctx, vec, k)
}

// SimilaritySearchByVector returns the k nearest documents to a query vector.
func (s *Store) SimilaritySearchByVector(ctx context.Context, vector []float64, k int) ([]schema.Document, error) {
	body, _ := json.Marshal(map[string]any{
		"vector":          vector,
		"topK":            k,
		"includeMetadata": true,
		"includeValues":   false,
		"namespace":       s.namespace,
	})
	resp, err := s.request(ctx, http.MethodPost, "/query", body)
	if err != nil {
		return nil, fmt.Errorf("pinecone: query: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("pinecone: query: status %d: %s", resp.StatusCode, string(b))
	}

	var result struct {
		Matches []struct {
			ID       string         `json:"id"`
			Score    float64        `json:"score"`
			Metadata map[string]any `json:"metadata"`
		} `json:"matches"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("pinecone: query: decode: %w", err)
	}

	docs := make([]schema.Document, 0, len(result.Matches))
	for _, m := range result.Matches {
		content := ""
		if v, ok := m.Metadata["content"]; ok {
			content = fmt.Sprint(v)
		}
		docs = append(docs, schema.Document{
			PageContent: content,
			Score:       m.Score,
			Metadata:    map[string]any{"id": m.ID},
		})
	}
	return docs, nil
}

// Delete removes documents by ID from Pinecone.
func (s *Store) Delete(ctx context.Context, ids []string) error {
	body, _ := json.Marshal(map[string]any{
		"ids":       ids,
		"namespace": s.namespace,
	})
	resp, err := s.request(ctx, http.MethodPost, "/vectors/delete", body)
	if err != nil {
		return fmt.Errorf("pinecone: delete: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("pinecone: delete: status %d: %s", resp.StatusCode, string(b))
	}
	return nil
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

func (s *Store) request(ctx context.Context, method, path string, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, s.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Api-Key", s.apiKey)
	req.Header.Set("Content-Type", "application/json")
	return s.client.Do(req)
}

// Close closes the HTTP transport's idle connections.
func (s *Store) Close() error {
	s.client.CloseIdleConnections()
	return nil
}
