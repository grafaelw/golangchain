// Package embeddings provides the Embedder interface and implementations for
// converting text into dense vector representations.
//
// # Embedder interface
//
//	type Embedder interface {
//	    EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error)
//	    EmbedQuery(ctx context.Context, text string) ([]float64, error)
//	}
//
// # Implementations
//
//   - [OpenAIEmbedder] — OpenAI text-embedding models (default: text-embedding-3-small)
//   - [AzureEmbedder]  — Azure OpenAI embedding deployments
//
// # Example
//
//	embedder, _ := embeddings.NewOpenAIEmbedder(
//	    os.Getenv("OPENAI_API_KEY"),
//	    embeddings.WithModel("text-embedding-3-large"),
//	)
//	vectors, _ := embedder.EmbedDocuments(ctx, []string{"Hello", "World"})
//
// Embedders are typically passed to [vectorstore.NewInMemoryVectorStore] or
// other VectorStore implementations.
package embeddings
