// Package azureaisearch provides a VectorStore backed by Azure AI Search
// (formerly "Cognitive Search"). It communicates via Azure's REST API using
// only the standard library.
//
// # Usage
//
//	store, err := azureaisearch.New(
//	    os.Getenv("AZURE_SEARCH_ENDPOINT"),  // e.g. "https://myservice.search.windows.net"
//	    os.Getenv("AZURE_SEARCH_API_KEY"),
//	    embedder,
//	    azureaisearch.WithIndexName("myindex"),
//	    azureaisearch.WithDimensions(1536),
//	)
//	_ = store.AddDocuments(ctx, docs)
//	results, _ := store.SimilaritySearch(ctx, "my query", 5)
//
// # Index schema
//
// The store automatically creates the index if it does not exist. The index
// contains three fields:
//   - id (Edm.String, key)
//   - content (Edm.String, searchable)
//   - contentVector (Collection(Edm.Single), vectorSearchable)
//
// Vector search uses HNSW with cosine distance by default.
package azureaisearch
