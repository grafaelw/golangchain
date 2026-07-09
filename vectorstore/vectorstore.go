// Package vectorstore provides the VectorStore interface and an in-memory
// implementation backed by cosine similarity search.
//
// The pluggable interface allows swapping in Qdrant, Weaviate, pgvector,
// or Azure AI Search without changing application code.
package vectorstore

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"

	"github.com/grafaelw/golangchain/embeddings"
	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// VectorStore interface
// ---------------------------------------------------------------------------

// VectorStore stores documents as dense vectors and retrieves the most similar
// documents for a given query.
type VectorStore interface {
	// AddDocuments embeds and indexes a batch of documents.
	AddDocuments(ctx context.Context, docs []schema.Document) error

	// SimilaritySearch returns the k most similar documents to the query string.
	SimilaritySearch(ctx context.Context, query string, k int) ([]schema.Document, error)

	// SimilaritySearchByVector returns the k most similar documents to a
	// pre-computed query vector (avoids an extra embedding call).
	SimilaritySearchByVector(ctx context.Context, vector []float64, k int) ([]schema.Document, error)

	// Delete removes all documents matching the given IDs from the store.
	// The semantics of "ID" are implementation-defined.
	Delete(ctx context.Context, ids []string) error
}

// ---------------------------------------------------------------------------
// InMemoryVectorStore
// ---------------------------------------------------------------------------

// entry holds a document alongside its embedding vector and an optional ID.
type entry struct {
	id     string
	doc    schema.Document
	vector []float64
}

// InMemoryVectorStore is a VectorStore backed by a slice of entries searched
// via cosine similarity. Suitable for prototyping and small corpora (<100k docs).
//
//	store := vectorstore.NewInMemoryVectorStore(embedder)
//	_ = store.AddDocuments(ctx, docs)
//	results, _ := store.SimilaritySearch(ctx, "my query", 5)
type InMemoryVectorStore struct {
	mu       sync.RWMutex
	entries  []entry
	embedder embeddings.Embedder
}

// NewInMemoryVectorStore creates an empty in-memory vector store that uses
// embedder to convert text into vectors.
func NewInMemoryVectorStore(embedder embeddings.Embedder) *InMemoryVectorStore {
	return &InMemoryVectorStore{embedder: embedder}
}

// AddDocuments embeds each document and stores it.
func (s *InMemoryVectorStore) AddDocuments(ctx context.Context, docs []schema.Document) error {
	if len(docs) == 0 {
		return nil
	}

	texts := make([]string, len(docs))
	for i, d := range docs {
		texts[i] = d.PageContent
	}

	vectors, err := s.embedder.EmbedDocuments(ctx, texts)
	if err != nil {
		return fmt.Errorf("vectorstore: embed documents: %w", err)
	}
	if len(vectors) != len(docs) {
		return fmt.Errorf("vectorstore: expected %d vectors, got %d", len(docs), len(vectors))
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	for i, doc := range docs {
		id := ""
		if v, ok := doc.Metadata["id"]; ok {
			id = fmt.Sprint(v)
		}
		s.entries = append(s.entries, entry{id: id, doc: doc, vector: vectors[i]})
	}
	return nil
}

// SimilaritySearch embeds the query and returns the k nearest documents.
func (s *InMemoryVectorStore) SimilaritySearch(ctx context.Context, query string, k int) ([]schema.Document, error) {
	qv, err := s.embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("vectorstore: embed query: %w", err)
	}
	return s.SimilaritySearchByVector(ctx, qv, k)
}

// SimilaritySearchByVector returns the k nearest documents to a pre-computed vector.
func (s *InMemoryVectorStore) SimilaritySearchByVector(_ context.Context, vector []float64, k int) ([]schema.Document, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(s.entries) == 0 {
		return nil, nil
	}

	type scored struct {
		score float64
		doc   schema.Document
	}

	results := make([]scored, 0, len(s.entries))
	for _, e := range s.entries {
		sim, err := cosineSimilarity(vector, e.vector)
		if err != nil {
			continue // dimension mismatch — skip silently
		}
		doc := e.doc
		doc.Score = sim
		results = append(results, scored{score: sim, doc: doc})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	if k > len(results) {
		k = len(results)
	}
	out := make([]schema.Document, k)
	for i := 0; i < k; i++ {
		out[i] = results[i].doc
	}
	return out, nil
}

// Delete removes entries whose metadata "id" field matches any of the given IDs.
func (s *InMemoryVectorStore) Delete(_ context.Context, ids []string) error {
	idSet := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		idSet[id] = struct{}{}
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	kept := s.entries[:0]
	for _, e := range s.entries {
		if _, drop := idSet[e.id]; !drop {
			kept = append(kept, e)
		}
	}
	s.entries = kept
	return nil
}

// Len returns the number of indexed documents.
func (s *InMemoryVectorStore) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.entries)
}

// Close clears the store. Implements io.Closer for clean teardown.
func (s *InMemoryVectorStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries = nil
	return nil
}

// ---------------------------------------------------------------------------
// Cosine similarity
// ---------------------------------------------------------------------------

func cosineSimilarity(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("dimension mismatch: %d vs %d", len(a), len(b))
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0, nil
	}
	return dot / denom, nil
}

// ---------------------------------------------------------------------------
// RetrieverTool — exposes a VectorStore as a tools.Tool for agents
// ---------------------------------------------------------------------------

// RetrieverTool wraps a VectorStore so an agent can call it as a tool.
// It formats the retrieved documents as a numbered list.
type RetrieverTool struct {
	store VectorStore
	k     int
	name  string
	desc  string
}

// NewRetrieverTool creates a RetrieverTool that fetches k documents.
func NewRetrieverTool(store VectorStore, k int, name, description string) *RetrieverTool {
	return &RetrieverTool{store: store, k: k, name: name, desc: description}
}

func (r *RetrieverTool) Name() string        { return r.name }
func (r *RetrieverTool) Description() string { return r.desc }
func (r *RetrieverTool) Schema() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}`)
}

func (r *RetrieverTool) Run(ctx context.Context, input string) (string, error) {
	docs, err := r.store.SimilaritySearch(ctx, input, r.k)
	if err != nil {
		return "", fmt.Errorf("retriever: %w", err)
	}
	if len(docs) == 0 {
		return "No relevant documents found.", nil
	}
	var sb strings.Builder
	for i, doc := range docs {
		fmt.Fprintf(&sb, "[%d] %s\n", i+1, doc.PageContent)
	}
	return sb.String(), nil
}
