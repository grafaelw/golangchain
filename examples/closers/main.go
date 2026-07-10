// Example: closers
//
// Demonstrates the io.Closer implementations added to resource-holding
// types across the library.  Proper cleanup is critical for vector stores
// that flush to disk and HTTP clients that hold idle connections.
//
// Highlights:
//
//   - CachingLLM.Close() cascades to both the inner LLM and the cache
//     backend if they implement io.Closer.
//
//   - InMemoryVectorStore.Close() clears all stored entries.
//
//   - FileVectorStore.Close() flushes pending writes then clears memory.
//
//   - Embedder.Close() cleans up the underlying HTTP transport (idle
//     connections) in OllamaEmbedder and is a no-op for OpenAI/Azure
//     (which reuse the global http.DefaultTransport).
//
//     Run this example with:
//     go run ./examples/closers
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/grafaelw/golangchain/embeddings"
	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/llmutil"
	"github.com/grafaelw/golangchain/schema"
	"github.com/grafaelw/golangchain/vectorstore"
)

func main() {
	ctx := context.Background()

	// -------------------------------------------------------------------------
	// 1. CachingLLM.Close() — cascading close
	// -------------------------------------------------------------------------
	fmt.Println("--- 1. CachingLLM.Close() ---")

	mock := &closableMock{label: "InnerLLM"}
	mcache := &closableMockCache{label: "Cache"}

	cache := llmutil.NewCachingLLM(mock, mcache)
	fmt.Println("  Closing CachingLLM...")
	if err := cache.Close(); err != nil {
		log.Fatal(err)
	}

	// -------------------------------------------------------------------------
	// 2. InMemoryVectorStore.Close() — clears entries
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 2. InMemoryVectorStore.Close() ---")
	emb := &mockEmbedder{embedding: []float64{1.0, 2.0, 3.0}}
	store := vectorstore.NewInMemoryVectorStore(emb)
	_ = store.AddDocuments(ctx, []schema.Document{
		{PageContent: "doc1"},
		{PageContent: "doc2"},
	})
	docs, _ := store.SimilaritySearch(ctx, "doc", 5)
	fmt.Printf("  Before close: %d documents\n", len(docs))

	if err := store.Close(); err != nil {
		log.Fatal(err)
	}
	docs, _ = store.SimilaritySearch(ctx, "doc", 5)
	fmt.Printf("  After close:  %d documents\n", len(docs))

	// -------------------------------------------------------------------------
	// 3. FileVectorStore.Close() — flush + clear
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 3. FileVectorStore.Close() (flush + clear) ---")
	emb2 := &mockEmbedder{embedding: []float64{0.5, 0.6, 0.7}}
	fstore, err := vectorstore.NewFileVectorStore("/tmp/golangchain_closer_example.json", emb2)
	if err != nil {
		log.Fatal(err)
	}
	_ = fstore.AddDocuments(ctx, []schema.Document{
		{PageContent: "persisted doc"},
	})
	fmt.Println("  Stored 1 document to disk.")
	if err := fstore.Close(); err != nil {
		log.Fatal(err)
	}
	fmt.Println("  Closed — flushed and cleared in-memory cache.")

	// -------------------------------------------------------------------------
	// 4. Graceful shutdown pattern (defer)
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 4. Graceful shutdown pattern ---")
	// In real code you'd wire up defer at construction time:
	//
	//     store := vectorstore.NewInMemoryVectorStore(emb)
	//     defer store.Close()
	//
	// For composable shutdowns you can collect closers:
	closers := []interface{ Close() error }{store, fstore}
	fmt.Printf("  %d closers ready for deferred cleanup\n", len(closers))
	for _, c := range closers {
		_ = c.Close()
	}
	fmt.Println("  All closers shut down.")
}

// mockEmbedder is a fake embeddings.Embedder for testing.
type mockEmbedder struct {
	embedding []float64
}

func (m *mockEmbedder) EmbedDocuments(_ context.Context, texts []string) ([][]float64, error) {
	result := make([][]float64, len(texts))
	for i := range texts {
		result[i] = m.embedding
	}
	return result, nil
}

func (m *mockEmbedder) EmbedQuery(_ context.Context, _ string) ([]float64, error) {
	return m.embedding, nil
}

// closableMock implements llm.LLM and io.Closer.
type closableMock struct {
	label  string
	closed bool
}

func (m *closableMock) Generate(_ context.Context, _ []schema.Message, _ ...llm.Option) (*schema.Generation, error) {
	return &schema.Generation{Text: m.label}, nil
}

func (m *closableMock) Stream(ctx context.Context, msgs []schema.Message, opts ...llm.Option) (<-chan schema.StreamChunk, error) {
	ch := make(chan schema.StreamChunk, 1)
	gen, err := m.Generate(ctx, msgs, opts...)
	if err != nil {
		close(ch)
		return ch, err
	}
	ch <- schema.StreamChunk{Text: gen.Text, Value: gen.Text, Done: true}
	close(ch)
	return ch, nil
}

func (m *closableMock) ModelName() string { return "mock-" + m.label }

func (m *closableMock) Close() error {
	if m.closed {
		return fmt.Errorf("%s already closed", m.label)
	}
	m.closed = true
	fmt.Printf("  %s.Close() called\n", m.label)
	return nil
}

// closableMockCache implements llmutil.Cache and io.Closer.
type closableMockCache struct {
	label  string
	closed bool
}

func (c *closableMockCache) Get(_ context.Context, _ string) (*schema.Generation, bool, error) {
	return nil, false, nil
}

func (c *closableMockCache) Set(_ context.Context, _ string, _ *schema.Generation) error {
	return nil
}

func (c *closableMockCache) Close() error {
	if c.closed {
		return fmt.Errorf("%s already closed", c.label)
	}
	c.closed = true
	fmt.Printf("  %s.Close() called\n", c.label)
	return nil
}

var _ llm.LLM = (*closableMock)(nil)
var _ llmutil.Cache = (*closableMockCache)(nil)
var _ embeddings.Embedder = (*mockEmbedder)(nil)
