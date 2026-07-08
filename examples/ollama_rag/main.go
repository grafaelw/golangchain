// This example demonstrates fully-local RAG with zero external API
// dependencies: Ollama for text generation AND embeddings, plus an
// in-memory or file vector store.
//
// Prerequisites:
//
//	ollama pull llama3.2        # for text generation
//	ollama pull nomic-embed-text # for embeddings
//
//	go run ./examples/ollama_rag
//
// Everything runs on your machine — no cloud API keys required.
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	ollamaembeddings "github.com/grafaelw/golangchain/embeddings/ollama"
	ollamaLLM "github.com/grafaelw/golangchain/llm/ollama"

	"github.com/grafaelw/golangchain/chain"
	"github.com/grafaelw/golangchain/retriever"
	"github.com/grafaelw/golangchain/schema"
	"github.com/grafaelw/golangchain/vectorstore"
)

func main() {
	ctx := context.Background()

	// ---------------------------------------------------------------------------
	// 1. Local LLM — Ollama for text generation
	// ---------------------------------------------------------------------------
	llmModel, err := ollamaLLM.New(
		ollamaLLM.WithModel("gemma4"),
		// ollamaLLM.WithBaseURL("http://192.168.1.5:11434/v1"),
	)
	if err != nil {
		log.Fatalf("create ollama LLM: %v\nIs Ollama running? → ollama serve", err)
	}
	section("Local Ollama LLM: " + llmModel.ModelName())

	// Quick generation test
	gen, err := llmModel.Generate(ctx,
		[]schema.Message{schema.NewHumanMessage("Say hello in Dutch in one word.")},
	)
	if err != nil {
		fmt.Printf("  (skipping — Ollama not reachable: %v)\n\n", err)
		fmt.Println("  Make sure Ollama is running: ollama serve")
		fmt.Println("  Then pull the model:          ollama pull gemma4")
		fmt.Println("  Then pull embeddings:         ollama pull nomic-embed-text")
		return
	}
	fmt.Printf("  LLM test: %s\n\n", strings.TrimSpace(gen.Text))

	// ---------------------------------------------------------------------------
	// 2. Local embeddings — Ollama /api/embed
	// ---------------------------------------------------------------------------
	emb, err := ollamaembeddings.New(
		ollamaembeddings.WithModel("nomic-embed-text"),
	)
	if err != nil {
		log.Fatalf("create ollama embedder: %v", err)
	}
	section("Local Ollama embeddings: nomic-embed-text")

	// Test embedding
	vec, err := emb.EmbedQuery(ctx, "golang concurrency")
	if err != nil {
		fmt.Printf("  (skipping embeddings — pull nomic-embed-text first: ollama pull nomic-embed-text)\n\n")
		return
	}
	fmt.Printf("  Query vector dimension: %d\n\n", len(vec))

	// ---------------------------------------------------------------------------
	// 3. Build the vector store and index some knowledge
	// ---------------------------------------------------------------------------
	section("Indexing documents")
	store := vectorstore.NewInMemoryVectorStore(emb)

	docs := []schema.Document{
		{
			PageContent: "Goroutines are lightweight threads managed by the Go runtime. They are multiplexed onto OS threads and have a small initial stack size (2KB).",
			Metadata:    map[string]any{"id": "1", "topic": "concurrency"},
		},
		{
			PageContent: "Channels are typed conduits for communication between goroutines. They can be buffered or unbuffered and support send, receive, and close operations.",
			Metadata:    map[string]any{"id": "2", "topic": "concurrency"},
		},
		{
			PageContent: "Go modules provide dependency management. A module is defined by a go.mod file that declares the module path and its dependencies with version constraints.",
			Metadata:    map[string]any{"id": "3", "topic": "modules"},
		},
		{
			PageContent: "Interfaces in Go are satisfied implicitly — a type implements an interface simply by having the required methods. No explicit declaration is needed.",
			Metadata:    map[string]any{"id": "4", "topic": "types"},
		},
		{
			PageContent: "The select statement lets a goroutine wait on multiple channel operations. It blocks until one case can proceed, then executes that case.",
			Metadata:    map[string]any{"id": "5", "topic": "concurrency"},
		},
	}

	if err := store.AddDocuments(ctx, docs); err != nil {
		log.Fatalf("add documents: %v", err)
	}
	fmt.Printf("  Indexed %d documents\n\n", len(docs))

	// ---------------------------------------------------------------------------
	// 4. Semantic search — find relevant docs
	// ---------------------------------------------------------------------------
	section("Semantic search")

	queries := []string{
		"how do goroutines work?",
		"how does Go handle dependencies?",
	}

	for _, q := range queries {
		results, err := store.SimilaritySearch(ctx, q, 2)
		if err != nil {
			log.Printf("search error: %v", err)
			continue
		}
		fmt.Printf("  Query: %s\n", q)
		for _, r := range results {
			fmt.Printf("    [score=%.3f] %s\n", r.Score, truncate(r.PageContent, 80))
		}
	}
	fmt.Println()

	// ---------------------------------------------------------------------------
	// 5. RAG QA — ask questions grounded in indexed docs
	// ---------------------------------------------------------------------------
	section("RAG QA — fully local")

	vsRetriever := retriever.NewVectorStoreRetriever(store, 2)
	qa := chain.NewRetrievalQAChain(vsRetriever, llmModel)
	qa.ReturnSource = true

	ragQuestions := []string{
		"What is a goroutine?",
		"How do Go interfaces work?",
	}

	for _, q := range ragQuestions {
		result, err := qa.Invoke(ctx, q)
		if err != nil {
			fmt.Printf("  Q: %s\n  Error: %v\n\n", q, err)
			continue
		}
		m := result.(map[string]any)
		fmt.Printf("  Q: %s\n  A: %s\n\n", q, truncate(fmt.Sprint(m["answer"]), 200))
	}

	fmt.Println("  ✓ Fully local RAG pipeline — no cloud APIs used.")
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
