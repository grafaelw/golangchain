// Example: rag
//
// End-to-end retrieval-augmented QA that exercises many of the packages
// added by this refresh:
//
//   - documentloader.NewDirectoryLoader  — file → []schema.Document
//   - textsplitter.NewMarkdownSplitter   — structural chunking
//   - embeddings.AzureEmbedder           — real embeddings via Azure OpenAI
//   - vectorstore.NewInMemoryVectorStore — cosine similarity index
//   - retriever.NewVectorStoreRetriever + BM25 + Ensemble (RRF)
//                                         — hybrid retrieval
//   - chain.NewRetrievalQAChain          — RAG stuff-prompt
//   - llmutil.NewCachingLLM              — response memoisation
//   - eval.Run                           — mini evaluation harness
//   - tracing.NewFileJSONLinesExporter   — structured trace output
//
// # Usage — Azure AI Foundry (default)
//
// Create a .env file with:
//
//	AZURE_OPENAI_API_KEY=<your-key>          # chat LLM
//	OPENAI_KEY=<azure-api-key>               # embeddings
//	OPENAI_ENDPOINT=https://<resource>.cognitiveservices.azure.com
//	OPENAI_EMBEDDING_DEPLOYMENT=<deployment-name>
//	OPENAI_API_VERSION=2024-02-01
//
// Then run:
//
//	go run ./examples/rag
//
// # Usage — OpenAI API
//
// Replace the embedder block with:
//
//	embedder, err := embeddings.NewOpenAIEmbedder(
//	    os.Getenv("OPENAI_API_KEY"),
//	    "text-embedding-3-small",
//	)
//
// Replace the model block with:
//
//	model, err := openai.New(
//	    openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
//	    openai.WithModel("gpt-4o-mini"),
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
	"path/filepath"

	"github.com/joho/godotenv"

	"github.com/grafaelw/golangchain/callbacks"
	"github.com/grafaelw/golangchain/chain"
	"github.com/grafaelw/golangchain/documentloader"
	"github.com/grafaelw/golangchain/embeddings"
	"github.com/grafaelw/golangchain/eval"
	"github.com/grafaelw/golangchain/llm/openai"
	"github.com/grafaelw/golangchain/llmutil"
	"github.com/grafaelw/golangchain/retriever"
	"github.com/grafaelw/golangchain/textsplitter"
	"github.com/grafaelw/golangchain/tracing"
	"github.com/grafaelw/golangchain/vectorstore"
)

// resourceOrigin strips path/query from an Azure endpoint so this example
// works whether OPENAI_ENDPOINT is a bare resource URL or the full Target URI.
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
	// 1. Load .env (silently ignored if missing; env vars still work).
	// -------------------------------------------------------------------------
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found - using environment variables")
	}

	// -------------------------------------------------------------------------
	// 2. Build a small on-disk corpus (real files → loader).
	// -------------------------------------------------------------------------
	dir, err := os.MkdirTemp("", "golangchain-rag-*")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)

	files := map[string]string{
		"go.md":     "# The Go Programming Language\n\nGo (or Golang) was released in 2009 by Google. It is a statically typed, compiled language known for its simplicity and first-class concurrency via goroutines.",
		"python.md": "# Python\n\nPython is an interpreted, dynamically typed language created by Guido van Rossum. It was first released in 1991 and is famous for its clean syntax and rich ecosystem.",
		"rust.md":   "# Rust\n\nRust is a systems programming language focused on memory safety without a garbage collector. It was first released in 2010 by Mozilla and is popular for low-level, high-performance software.",
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			log.Fatal(err)
		}
	}

	docs, err := documentloader.NewDirectoryLoader(dir).Load(ctx)
	if err != nil {
		log.Fatal(err)
	}
	splitter := textsplitter.NewMarkdownSplitter(
		textsplitter.WithChunkSize(200),
		textsplitter.WithChunkOverlap(20),
	)
	chunks := splitter.SplitDocuments(docs)
	fmt.Printf("Loaded %d docs → %d chunks\n", len(docs), len(chunks))

	// -------------------------------------------------------------------------
	// 3. Real embeddings — Azure OpenAI via AzureEmbedder.
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
	store := vectorstore.NewInMemoryVectorStore(embedder)
	if err := store.AddDocuments(ctx, chunks); err != nil {
		log.Fatal(err)
	}

	// -------------------------------------------------------------------------
	// 4. Hybrid retriever: BM25 (lexical) + VectorStore (semantic), fused
	//    with Reciprocal Rank Fusion.
	// -------------------------------------------------------------------------
	lex := retriever.NewBM25Retriever(chunks, 3)
	vec := retriever.NewVectorStoreRetriever(store, 3)
	hybrid := retriever.NewEnsembleRetriever(
		[]retriever.Retriever{lex, vec},
		[]float64{0.4, 0.6}, // weight semantic slightly higher
		3,
	)

	// -------------------------------------------------------------------------
	// 5. Real chat LLM — Azure AI Foundry via the openai package, wrapped
	//    with a memory cache so repeated questions are free.
	//
	// To use the OpenAI API instead, replace this block with:
	//
	//     model, err := openai.New(
	//         openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
	//         openai.WithModel("gpt-4o-mini"),
	//     )
	//
	// and set OPENAI_API_KEY in your .env.
	// -------------------------------------------------------------------------
	model, err := openai.New(
		openai.WithAPIKey(os.Getenv("AZURE_OPENAI_API_KEY")),
		openai.WithModel("gpt-5.4-nano"),
		openai.WithBaseURL("https://ai-lab-nl-sweden-foundry.services.ai.azure.com/openai/v1/"),
	)
	if err != nil {
		log.Fatal(err)
	}
	cached := llmutil.NewCachingLLM(model, llmutil.NewMemoryCache())

	// -------------------------------------------------------------------------
	// 6. Tracing to JSON lines (jq / Loki / OTel-file-receiver friendly).
	// -------------------------------------------------------------------------
	tracePath := filepath.Join(dir, "trace.jsonl")
	exporter, closer, err := tracing.NewFileJSONLinesExporter(tracePath)
	if err != nil {
		log.Fatal(err)
	}
	defer closer.Close()
	cm := callbacks.NewCallbackManager(exporter)

	// -------------------------------------------------------------------------
	// 7. Build the RAG chain and answer a question.
	// -------------------------------------------------------------------------
	qa := chain.NewRetrievalQAChain(hybrid, cached)
	qa.ReturnSource = true
	_ = cm // hook cm into agent/graph pipelines that accept a CallbackManager

	fmt.Println("\n--- RetrievalQAChain ---")
	out, err := qa.Invoke(ctx, "When was Go released and what makes it distinctive?")
	if err != nil {
		log.Fatal(err)
	}
	res := out.(map[string]any)
	fmt.Printf("Q: When was Go released and what makes it distinctive?\nA: %s\n", res["answer"])

	// -------------------------------------------------------------------------
	// 8. Tiny evaluation over a two-row dataset.
	// -------------------------------------------------------------------------
	fmt.Println("\n--- Evaluation ---")
	dataset := eval.Dataset{
		{Input: "When was Go released?", Expected: "2009"},
		{Input: "Who created Python?", Expected: "Guido"},
	}
	report, err := eval.Run(ctx, dataset,
		func(ctx context.Context, in any) (any, error) {
			o, err := qa.Invoke(ctx, in)
			if err != nil {
				return nil, err
			}
			return o.(map[string]any)["answer"], nil
		},
		[]eval.Evaluator{eval.Contains{CaseInsensitive: true}},
		2,
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Summary: %+v\n", report.Summary)
	fmt.Printf("\nJSONL trace written to %s\n", tracePath)
}
