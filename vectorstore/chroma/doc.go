// Package chroma provides a VectorStore backed by a Chroma vector database.
// It communicates via Chroma's REST API (v1) using only the standard library.
//
// # Usage
//
//	store, err := chroma.New(
//	    "http://localhost:8000",
//	    "my-collection",
//	    embedder,
//	)
//	_ = store.AddDocuments(ctx, docs)
//	results, _ := store.SimilaritySearch(ctx, "my query", 5)
//
// When using Chroma's built-in embedding function, pass WithEmbeddingFunction:
//
//	store, err := chroma.New(
//	    "http://localhost:8000",
//	    "my-collection",
//	    nil,
//	    chroma.WithEmbeddingFunction("default"),
//	)
package chroma
