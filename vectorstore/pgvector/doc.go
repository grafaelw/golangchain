// Package pgvector provides a VectorStore backed by PostgreSQL with the
// pgvector extension. It supports cosine distance (<=>) similarity search
// and automatic table creation.
//
// Prerequisites:
//
//	CREATE EXTENSION IF NOT EXISTS vector;
//
// Usage:
//
//	db, _ := sql.Open("postgres", "postgres://user:pass@localhost/db")
//	store, _ := pgvector.New(db, "documents", 1536, embedder)
//	_ = store.AddDocuments(ctx, docs)
//	results, _ := store.SimilaritySearch(ctx, "query", 5)
package pgvector
