// Package vectorstore provides the VectorStore interface and an in-memory
// implementation backed by cosine similarity search.
//
// # VectorStore interface
//
//	type VectorStore interface {
//	    AddDocuments(ctx context.Context, docs []schema.Document) error
//	    SimilaritySearch(ctx context.Context, query string, k int) ([]schema.Document, error)
//	    SimilaritySearchByVector(ctx context.Context, vector []float64, k int) ([]schema.Document, error)
//	    Delete(ctx context.Context, ids []string) error
//	}
//
// # Implementations
//
//   - [InMemoryVectorStore] — cosine similarity over an in-memory slice;
//     suitable for prototyping and corpora up to ~100k documents.
//   - [FileVectorStore]     — InMemoryVectorStore backed by a JSON file on
//     disk; AddDocuments and Delete auto-persist. Cached vectors are reloaded
//     on open, avoiding a re-embed on restart.
//
// # RetrieverTool
//
// [RetrieverTool] wraps any VectorStore as a [tools.Tool], making it usable
// directly by agents without extra glue code:
//
//	store   := vectorstore.NewInMemoryVectorStore(embedder)
//	_       = store.AddDocuments(ctx, docs)
//	retriever := vectorstore.NewRetrieverTool(store, 5, "search_docs", "Search project documentation.")
//	executor := agent.NewAgentExecutor(myAgent, []tools.Tool{retriever})
//
// # Example
//
//	store := vectorstore.NewInMemoryVectorStore(myEmbedder)
//	_ = store.AddDocuments(ctx, []schema.Document{
//	    {PageContent: "Go is a compiled language.", Metadata: map[string]any{"id": "1"}},
//	})
//	results, _ := store.SimilaritySearch(ctx, "golang compiler", 3)
package vectorstore
