// Package cohere provides an embeddings.Embedder backed by Cohere's Embed
// API (https://docs.cohere.com/reference/embed).
//
// # Usage
//
//	emb, err := cohere.New(
//	    os.Getenv("COHERE_API_KEY"),
//	    cohere.WithModel("embed-english-v3.0"),
//	)
//	vectors, _ := emb.EmbedDocuments(ctx, []string{"hello", "world"})
//
// The embedder uses "search_document" input type for documents and
// "search_query" for queries, matching Cohere's recommended RAG usage.
package cohere
