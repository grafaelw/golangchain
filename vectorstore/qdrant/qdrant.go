// Package qdrant provides a VectorStore backed by a Qdrant vector database.
// It communicates via Qdrant's REST API using only the standard library.
package qdrant

import (
	"bytes"
	"context"
	"crypto/sha256"
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

// Option configures the Qdrant store.
type Option func(*Store)

// WithCollection sets the collection name (default "documents").
func WithCollection(name string) Option {
	return func(s *Store) { s.collection = name }
}

// WithVectorSize sets the expected vector dimension (default 1536).
func WithVectorSize(size int) Option {
	return func(s *Store) { s.vectorSize = size }
}

// WithDistance sets the distance metric: "Cosine", "Euclid", "Dot".
func WithDistance(metric string) Option {
	return func(s *Store) { s.distance = metric }
}

// WithHTTPClient overrides the HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(s *Store) { s.client = client }
}

// WithAPIKey sets an optional API key header (Qdrant Cloud).
func WithAPIKey(key string) Option {
	return func(s *Store) { s.apiKey = key }
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

// Store implements vectorstore.VectorStore backed by Qdrant.
type Store struct {
	baseURL    string
	collection string
	vectorSize int
	distance   string
	client     *http.Client
	apiKey     string
	embedder   embeddings.Embedder
	created    bool
}

// New creates a Qdrant-backed vector store.
func New(baseURL string, embedder embeddings.Embedder, opts ...Option) (*Store, error) {
	if baseURL == "" {
		return nil, errors.New("qdrant: base URL is required")
	}
	s := &Store{
		baseURL:    strings.TrimSuffix(baseURL, "/"),
		collection: "documents",
		vectorSize: 1536,
		distance:   "Cosine",
		embedder:   embedder,
		client:     &http.Client{Timeout: 30 * time.Second},
	}
	for _, o := range opts {
		o(s)
	}
	return s, nil
}

// Ensure interface compliance.
var _ vectorstore.VectorStore = (*Store)(nil)

// ---------------------------------------------------------------------------
// VectorStore implementation
// ---------------------------------------------------------------------------

// AddDocuments embeds and indexes documents.
func (s *Store) AddDocuments(ctx context.Context, docs []schema.Document) error {
	if len(docs) == 0 {
		return nil
	}
	if err := s.ensureCollection(ctx); err != nil {
		return err
	}

	texts := make([]string, len(docs))
	for i, doc := range docs {
		texts[i] = doc.PageContent
	}
	vectors, err := s.embedder.EmbedDocuments(ctx, texts)
	if err != nil {
		return fmt.Errorf("qdrant: embed: %w", err)
	}
	if len(vectors) != len(docs) {
		return fmt.Errorf("qdrant: expected %d vectors, got %d", len(docs), len(vectors))
	}

	type point struct {
		ID      any            `json:"id"`
		Vector  []float64      `json:"vector"`
		Payload map[string]any `json:"payload"`
	}

	points := make([]point, len(docs))
	for i, doc := range docs {
		src := fmt.Sprintf("doc_%d", i)
		if v, ok := doc.Metadata["id"]; ok {
			src = fmt.Sprint(v)
		}
		payload := map[string]any{"content": doc.PageContent, "_src_id": src}
		for k, v := range doc.Metadata {
			payload[k] = v
		}
		points[i] = point{ID: stringToUUID(src), Vector: vectors[i], Payload: payload}
	}

	body, _ := json.Marshal(map[string]any{"points": points})
	resp, err := s.request(ctx, http.MethodPut, "/collections/"+s.collection+"/points", body)
	if err != nil {
		return fmt.Errorf("qdrant: add documents: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("qdrant: add documents: status %d: %s", resp.StatusCode, string(b))
	}
	return nil
}

// SimilaritySearch embeds the query and returns the k nearest documents.
func (s *Store) SimilaritySearch(ctx context.Context, query string, k int) ([]schema.Document, error) {
	vec, err := s.embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("qdrant: embed query: %w", err)
	}
	return s.SimilaritySearchByVector(ctx, vec, k)
}

// SimilaritySearchByVector returns the k nearest documents to a query vector.
func (s *Store) SimilaritySearchByVector(ctx context.Context, vector []float64, k int) ([]schema.Document, error) {
	if err := s.ensureCollection(ctx); err != nil {
		return nil, err
	}

	body, _ := json.Marshal(map[string]any{
		"vector":       vector,
		"limit":        k,
		"with_payload": true,
	})
	resp, err := s.request(ctx, http.MethodPost, "/collections/"+s.collection+"/points/search", body)
	if err != nil {
		return nil, fmt.Errorf("qdrant: search: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("qdrant: search: status %d: %s", resp.StatusCode, string(b))
	}

	var result struct {
		Result []struct {
			ID      any     `json:"id"`
			Score   float64 `json:"score"`
			Payload struct {
				Content string `json:"content"`
			} `json:"payload"`
		} `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("qdrant: search: decode: %w", err)
	}

	docs := make([]schema.Document, 0, len(result.Result))
	for _, r := range result.Result {
		docs = append(docs, schema.Document{
			PageContent: r.Payload.Content,
			Score:       r.Score,
			Metadata:    map[string]any{"id": fmt.Sprint(r.ID)},
		})
	}
	return docs, nil
}

// Delete removes documents by IDs.
func (s *Store) Delete(ctx context.Context, ids []string) error {
	if err := s.ensureCollection(ctx); err != nil {
		return err
	}
	for _, id := range ids {
		url := "/collections/" + s.collection + "/points/delete"
		body, _ := json.Marshal(map[string]any{
			"points": []string{stringToUUID(id)},
		})
		resp, err := s.request(ctx, http.MethodPost, url, body)
		if err != nil {
			return fmt.Errorf("qdrant: delete: %w", err)
		}
		_ = resp.Body.Close()
	}
	return nil
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

func (s *Store) ensureCollection(ctx context.Context) error {
	if s.created {
		return nil
	}
	body, _ := json.Marshal(map[string]any{
		"vectors": map[string]any{
			"size":     s.vectorSize,
			"distance": s.distance,
		},
	})
	resp, err := s.request(ctx, http.MethodPut, "/collections/"+s.collection, body)
	if err != nil {
		return fmt.Errorf("qdrant: create collection: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 400 && resp.StatusCode != 409 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("qdrant: create collection %q: status %d: %s", s.collection, resp.StatusCode, string(b))
	}
	s.created = true
	return nil
}

func (s *Store) request(ctx context.Context, method, path string, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, s.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if s.apiKey != "" {
		req.Header.Set("api-key", s.apiKey)
	}
	return s.client.Do(req)
}

func stringToUUID(s string) string {
	h := sha256.Sum256([]byte(s))
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		h[0:4], h[4:6], h[6:8], h[8:10], h[10:16])
}
