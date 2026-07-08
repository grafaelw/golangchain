// Example: vectorstore
//
// Demonstrates the vectorstore and embeddings packages:
//
//   - embeddings.AzureEmbedder — generates real vector embeddings via an
//     Azure OpenAI Foundry embedding deployment
//   - InMemoryVectorStore      — indexes Documents by cosine similarity
//   - RetrieverTool            — wraps a VectorStore as a Tool, so an agent
//     can call it like any other tool
//
// # Usage — Azure AI Foundry (default)
//
// Create a .env file with:
//
//	OPENAI_KEY=<azure-api-key>
//	OPENAI_ENDPOINT=https://<resource>.cognitiveservices.azure.com
//	OPENAI_EMBEDDING_DEPLOYMENT=<deployment-name>
//	OPENAI_API_VERSION=2024-02-01
//
// Then run:
//
//	go run ./examples/vectorstore
//
// # Usage — OpenAI API
//
// Replace the embedder initialisation block with:
//
//	embedder, err := embeddings.NewOpenAIEmbedder(
//	    os.Getenv("OPENAI_API_KEY"),
//	    "text-embedding-3-small",
//	)
//
// Create a .env file with:
//
//	OPENAI_API_KEY=sk-...
package main

import (
	"context"
	"fmt"
	"log"
	"net/url"
	"os"

	"github.com/joho/godotenv"

	"github.com/grafaelw/golangchain/embeddings"
	"github.com/grafaelw/golangchain/schema"
	"github.com/grafaelw/golangchain/vectorstore"
)

// resourceOrigin strips any path/query from an Azure endpoint, so this
// example works whether OPENAI_ENDPOINT is a bare resource URL
// (https://<resource>.cognitiveservices.azure.com) or the full "Target URI"
// Azure AI Foundry's portal hands you, which already has
// /openai/deployments/<deployment>/embeddings?api-version=... appended.
func resourceOrigin(endpoint string) string {
	u, err := url.Parse(endpoint)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return endpoint
	}
	return u.Scheme + "://" + u.Host
}

func main() {
	ctx := context.Background()

	// -------------------------------------------------------------------------
	// 1. Load .env (silently ignored if the file doesn't exist,
	// 	  so real environment variables still work in CI/production)
	// -------------------------------------------------------------------------
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found - using environment variables")
	}

	// -------------------------------------------------------------------------
	// 2. Create the embedder — Azure OpenAI via AzureEmbedder.
	//
	// To use the OpenAI API instead, replace this block with:
	//
	//     embedder, err := embeddings.NewOpenAIEmbedder(
	//         os.Getenv("OPENAI_API_KEY"),
	//         "text-embedding-3-small",
	//     )
	//
	// and set OPENAI_API_KEY in your .env.
	// -------------------------------------------------------------------------
	embedder, err := embeddings.NewAzureEmbedder(
		os.Getenv("AZURE_OPENAI_API_KEY"),
		resourceOrigin(os.Getenv("AZURE_OPENAI_ENDPOINT")),
		os.Getenv("OPENAI_EMBEDDING_DEPLOYMENT"),
		os.Getenv("OPENAI_API_VERSION"),
	)
	if err != nil {
		log.Fatal(err)
	}

	// -------------------------------------------------------------------------
	// 3. Build the store and index documents
	// -------------------------------------------------------------------------
	store := vectorstore.NewInMemoryVectorStore(embedder)

	docs := []schema.Document{
		{PageContent: "Go is a statically typed compiled language designed at Google.", Metadata: map[string]any{"id": "doc1", "topic": "Go"}},
		{PageContent: "Python is an interpreted high-level programming language.", Metadata: map[string]any{"id": "doc2", "topic": "Python"}},
		{PageContent: "Rust guarantees memory safety without a garbage collector.", Metadata: map[string]any{"id": "doc3", "topic": "Rust"}},
		{PageContent: "Go's goroutines make concurrent programming simple and efficient.", Metadata: map[string]any{"id": "doc4", "topic": "Go"}},
		{PageContent: "LangChain provides composable building blocks for LLM applications.", Metadata: map[string]any{"id": "doc5", "topic": "LLM"}},
	}
	if err := store.AddDocuments(ctx, docs); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Indexed %d documents\n", store.Len())

	// -------------------------------------------------------------------------
	// 4. Similarity search
	// -------------------------------------------------------------------------
	fmt.Println("\n--- Similarity search ---")
	results, err := store.SimilaritySearch(ctx, "goroutine concurrency golang", 3)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Top-3 results for 'goroutine concurrency golang':")
	for i, d := range results {
		fmt.Printf("  [%d] score=%.4f  %s\n", i+1, d.Score, truncate(d.PageContent, 50))
	}

	// -------------------------------------------------------------------------
	// 5. Delete by metadata ID
	// -------------------------------------------------------------------------
	fmt.Println("\n--- Delete ---")
	if err := store.Delete(ctx, []string{"doc1"}); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("After deleting doc1: %d documents remain\n", store.Len())

	// -------------------------------------------------------------------------
	// 6. RetrieverTool — wrap the store as a Tool for agents
	// -------------------------------------------------------------------------
	fmt.Println("\n--- RetrieverTool ---")
	retriever := vectorstore.NewRetrieverTool(
		store, 2,
		"knowledge_base",
		"Searches indexed programming-language documents and returns the most relevant snippets.",
	)
	fmt.Printf("Tool: %s — %s\n", retriever.Name(), retriever.Description())

	out, err := retriever.Run(ctx, "memory safety without garbage collection")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("RetrieverTool result:\n%s\n", out)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
