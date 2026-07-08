// Package pinecone provides a VectorStore backed by Pinecone's serverless
// vector database. It communicates via Pinecone's REST API using only the
// standard library.
//
// # Usage
//
//	store, err := pinecone.New(
//	    os.Getenv("PINECONE_API_KEY"),
//	    "https://myindex-abc123.svc.us-east1-aws.pinecone.io",
//	    embedder,
//	    pinecone.WithDimension(1536),
//	)
//	_ = store.AddDocuments(ctx, docs)
//	results, _ := store.SimilaritySearch(ctx, "my query", 5)
package pinecone
