// This example demonstrates the Azure AI Search vector store integration for
// production-grade semantic search on Azure. Azure AI Search handles indexing,
// vector search, and hybrid search at scale.
//
// Prerequisites:
//   - An Azure AI Search resource (free tier works)
//   - An Azure OpenAI embedding deployment (for generating embeddings)
//
// # Setup
//
// Create a .env file with:
//
//	AZURE_SEARCH_ENDPOINT=https://<service>.search.windows.net
//	AZURE_SEARCH_API_KEY=<admin-or-query-key>
//	AZURE_OPENAI_API_KEY=<key>
//	AZURE_OPENAI_ENDPOINT=https://<resource>.cognitiveservices.azure.com
//	OPENAI_EMBEDDING_DEPLOYMENT=<embedding-deployment-name>
//	OPENAI_API_VERSION=2024-02-01
//
// # Run
//
//	go run ./examples/rag/azure_ai_search
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/joho/godotenv"

	"github.com/grafaelw/golangchain/embeddings"
	"github.com/grafaelw/golangchain/schema"
	"github.com/grafaelw/golangchain/vectorstore/azureaisearch"
)

func main() {
	_ = godotenv.Load()

	ctx := context.Background()

	// ---------------------------------------------------------------------------
	// 1. Create the embedder
	//
	// Azure embeddings — used in the RAG and vectorstore examples.
	// To switch to the OpenAI API, replace the block below with:
	//
	//     embedder, err := embeddings.NewOpenAIEmbedder(
	//         os.Getenv("OPENAI_API_KEY"),
	//         embeddings.WithModel("text-embedding-3-small"),
	//     )
	// ---------------------------------------------------------------------------
	embedder, err := embeddings.NewAzureEmbedder(
		os.Getenv("AZURE_OPENAI_API_KEY"),
		os.Getenv("AZURE_OPENAI_ENDPOINT"),
		os.Getenv("OPENAI_EMBEDDING_DEPLOYMENT"),
		os.Getenv("OPENAI_API_VERSION"),
	)
	if err != nil {
		log.Fatalf("create embedder: %v", err)
	}

	// ---------------------------------------------------------------------------
	// 2. Connect to Azure AI Search
	//
	// Options:
	//   azureaisearch.WithIndexName("myindex")     — custom index name
	//   azureaisearch.WithDimensions(1536)          — match your embedding dim
	//   azureaisearch.WithSemanticConfig("config")  — enable semantic reranking
	// ---------------------------------------------------------------------------
	store, err := azureaisearch.New(
		os.Getenv("AZURE_SEARCH_ENDPOINT"),
		os.Getenv("AZURE_SEARCH_API_KEY"),
		embedder,
		azureaisearch.WithIndexName("golangchain-demo"),
		azureaisearch.WithDimensions(1536),
	)
	if err != nil {
		log.Fatalf("create azure ai search store: %v", err)
	}

	section("Azure AI Search vector store")

	// ---------------------------------------------------------------------------
	// 3. Index documents
	// ---------------------------------------------------------------------------
	docs := []schema.Document{
		{
			PageContent: "Go is a statically typed, compiled programming language designed at Google by Robert Griesemer, Rob Pike, and Ken Thompson.",
			Metadata:    map[string]any{"id": "go-intro"},
		},
		{
			PageContent: "Generics were introduced in Go 1.18, allowing type parameters on functions and types. This enables writing reusable, type-safe code.",
			Metadata:    map[string]any{"id": "generics"},
		},
		{
			PageContent: "The Go runtime includes a garbage collector, goroutine scheduler, and memory allocator. Goroutines are lightweight threads multiplexed onto OS threads.",
			Metadata:    map[string]any{"id": "runtime"},
		},
		{
			PageContent: "Go modules (go.mod) provide dependency management. The go command downloads and verifies modules from proxies like proxy.golang.org.",
			Metadata:    map[string]any{"id": "modules"},
		},
		{
			PageContent: "Go's standard library is comprehensive. net/http provides HTTP client/server, encoding/json handles JSON, and database/sql abstracts database access.",
			Metadata:    map[string]any{"id": "stdlib"},
		},
	}

	fmt.Printf("  Connecting to Azure AI Search at %s ...\n", os.Getenv("AZURE_SEARCH_ENDPOINT"))
	err = store.AddDocuments(ctx, docs)
	if err != nil {
		fmt.Printf("  (skipping — Azure AI Search not reachable: %v)\n", err)
		fmt.Println("\n  Create a free Azure AI Search resource at:")
		fmt.Println("  https://portal.azure.com/#create/Microsoft.Search")
		return
	}
	fmt.Printf("  ✓ Created index 'golangchain-demo' with %d documents\n\n", len(docs))

	// ---------------------------------------------------------------------------
	// 4. Semantic search
	// ---------------------------------------------------------------------------
	section("Semantic search")

	queries := []string{
		"who created the Go language?",
		"how does Go handle dependencies?",
		"explain goroutines",
	}

	for _, q := range queries {
		results, err := store.SimilaritySearch(ctx, q, 2)
		if err != nil {
			log.Printf("search error: %v", err)
			continue
		}
		fmt.Printf("  Query: %s\n", q)
		for _, r := range results {
			fmt.Printf("    [score=%.3f, id=%s] %s\n",
				r.Score, r.Metadata["id"], truncate(r.PageContent, 80))
		}
	}
	fmt.Println()

	// ---------------------------------------------------------------------------
	// 5. Delete by ID
	// ---------------------------------------------------------------------------
	section("Delete documents")
	fmt.Println("  Deleting document 'stdlib' ...")
	if err := store.Delete(ctx, []string{"stdlib"}); err != nil {
		log.Printf("delete error: %v", err)
	}
	results, _ := store.SimilaritySearch(ctx, "HTTP and JSON libraries", 1)
	if len(results) > 0 {
		fmt.Printf("  After delete, closest match: [id=%s] %s\n",
			results[0].Metadata["id"], truncate(results[0].PageContent, 60))
	}
	fmt.Println()

	// ---------------------------------------------------------------------------
	// 6. Using with a RAG chain
	// ---------------------------------------------------------------------------
	section("RAG QA integration")
	fmt.Println("  Integrate with RetrievalQAChain:")
	fmt.Println("    vsRetriever := retriever.NewVectorStoreRetriever(store, 4)")
	fmt.Println("    qa := chain.NewRetrievalQAChain(vsRetriever, llmModel)")
	fmt.Println("    answer, _ := qa.Invoke(ctx, \"What are goroutines?\")")
	fmt.Println()
	fmt.Println("  Integrate with agents:")
	fmt.Println("    tool := vectorstore.NewRetrieverTool(store, 5, \"search\", \"...\")")
	fmt.Println("    executor := agent.NewAgentExecutor(myAgent, []tools.Tool{tool})")
}

func section(title string) {
	fmt.Println(strings.Repeat("─", 72))
	fmt.Println(title)
	fmt.Println(strings.Repeat("─", 72))
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}
