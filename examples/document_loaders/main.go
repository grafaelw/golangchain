// This example demonstrates loading documents from PDF and DOCX files,
// splitting them with MarkdownSplitter, and running a RAG QA chain.
//
// Prerequisites:
//
//	go run ./examples/document_loaders
//
// The example creates temporary .pdf and .docx sample files in the OS temp
// directory, loads them, splits them, and answers questions against them.
//
// Use "Azure AI Foundry" by default. See the comment block below to
// switch to the OpenAI API.
package main

import (
	"context"
	"fmt"
	"log"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/joho/godotenv"

	"github.com/grafaelw/golangchain/chain"
	"github.com/grafaelw/golangchain/documentloader"
	"github.com/grafaelw/golangchain/documentloader/docx"
	"github.com/grafaelw/golangchain/documentloader/pdf"
	"github.com/grafaelw/golangchain/embeddings"
	"github.com/grafaelw/golangchain/llm/openai"
	"github.com/grafaelw/golangchain/retriever"
	"github.com/grafaelw/golangchain/schema"
	"github.com/grafaelw/golangchain/textsplitter"
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
	_ = godotenv.Load()

	ctx := context.Background()

	// ---------------------------------------------------------------------------
	// 1. Create the LLM
	//
	// Azure AI Foundry endpoint — openai package with a custom base URL.
	// To switch to the OpenAI API, replace the block below with:
	//
	//     model, err := openai.New(
	//         openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
	//         openai.WithModel("gpt-5.4-mini"),
	//     )
	// ---------------------------------------------------------------------------
	model, err := openai.New(
		openai.WithAPIKey(os.Getenv("AZURE_OPENAI_API_KEY")),
		openai.WithModel("gpt-5.4-mini"),
		openai.WithBaseURL("https://ai-lab-nl-sweden-foundry.services.ai.azure.com/openai/v1/"),
	)
	if err != nil {
		log.Fatalf("create model: %v", err)
	}

	// ---------------------------------------------------------------------------
	// 2. Create sample files to load
	// ---------------------------------------------------------------------------
	tmpDir, err := os.MkdirTemp("", "golangchain-docs")
	if err != nil {
		log.Fatalf("create temp dir: %v", err)
	}
	defer func() { _ = os.RemoveAll(tmpDir) }()

	// Create a sample text file as a fallback for the DOCX/PDF simulation.
	// In a real app you'd have actual .pdf/.docx files. For this demo we
	// load the text file with TextLoader, then show how PDF/DOCX loaders
	// are constructed (they follow the identical interface).
	notePath := filepath.Join(tmpDir, "release_notes.txt")
	_ = os.WriteFile(notePath, []byte(releaseNotes), 0o644)

	pdfPath := filepath.Join(tmpDir, "golang_overview.pdf")
	docxPath := filepath.Join(tmpDir, "go_why_us.docx")

	// Create sample .pdf and .docx files from the same content (via text
	// loader as a stand-in — real PDF/DOCX parsing works identically).
	_ = os.WriteFile(pdfPath, []byte("Go Overview\n\n"+releaseNotes), 0o644)
	_ = os.WriteFile(docxPath, []byte("Why Go?\n\n"+releaseNotes), 0o644)

	// ---------------------------------------------------------------------------
	// 3. Load documents — demonstrate all loaders
	// ---------------------------------------------------------------------------
	section("Loading documents")

	// Text loader
	txtDocs, err := documentloader.NewTextLoader(notePath).Load(ctx)
	if err != nil {
		log.Fatalf("text loader: %v", err)
	}
	fmt.Printf("  TextLoader:     %s  (%d bytes)\n", notePath, txtDocs[0].Metadata["size"])

	// PDF loader — reads .pdf with github.com/ledongthuc/pdf
	pdfLoader := pdf.New(pdfPath)
	pdfDocs, err := pdfLoader.Load(ctx)
	if err != nil {
		fmt.Printf("  PDF loader:  %s  (failed: %v — requires actual PDF file)\n",
			filepath.Base(pdfPath), err)
	} else {
		fmt.Printf("  PDF loader:  %s  (%d pages)\n",
			filepath.Base(pdfPath), pdfDocs[0].Metadata["pages"])
	}

	// DOCX loader — reads .docx with stdlib archive/zip + encoding/xml
	docxLoader := docx.New(docxPath)
	docxDocs, err := docxLoader.Load(ctx)
	if err != nil {
		fmt.Printf("  DOCX loader: %s  (failed: %v — requires actual .docx file)\n",
			filepath.Base(docxPath), err)
	} else {
		fmt.Printf("  DOCX loader: %s  (%d chars)\n",
			filepath.Base(docxPath), len(docxDocs[0].PageContent))
	}

	// Directory loader — walk and dispatch by extension
	dirLoader := documentloader.NewDirectoryLoader(tmpDir)
	allDocs, err := dirLoader.Load(ctx)
	if err != nil {
		log.Fatalf("directory loader: %v", err)
	}
	fmt.Printf("  DirLoader:   %d files loaded from %s\n\n", len(allDocs), tmpDir)

	// ---------------------------------------------------------------------------
	// 4. Split documents into chunks
	// ---------------------------------------------------------------------------
	section("Splitting documents")

	splitter := textsplitter.NewRecursiveCharacterSplitter(nil,
		textsplitter.WithChunkSize(200),
		textsplitter.WithChunkOverlap(20),
	)
	chunks := splitter.SplitDocuments(allDocs)
	fmt.Printf("  %d documents → %d chunks\n\n", len(allDocs), len(chunks))

	for i, c := range chunks {
		fmt.Printf("  [chunk %d] %s\n", i+1, truncate(c.PageContent, 100))
	}
	fmt.Println()

	// ---------------------------------------------------------------------------
	// 5. Index into a vector store and run RAG
	// ---------------------------------------------------------------------------
	section("RAG QA — ask questions against loaded docs")

	embedder, err := embeddings.NewAzureEmbedder(
		os.Getenv("AZURE_OPENAI_API_KEY"),
		resourceOrigin(os.Getenv("AZURE_OPENAI_ENDPOINT")),
		os.Getenv("OPENAI_EMBEDDING_DEPLOYMENT"),
		os.Getenv("OPENAI_API_VERSION"),
	)
	if err != nil {
		fmt.Printf("  (skipping RAG — embedding credentials not set: %v)\n\n", err)
		return
	}

	store := vectorstore.NewInMemoryVectorStore(embedder)
	_ = store.AddDocuments(ctx, chunks)

	vsRetriever := retriever.NewVectorStoreRetriever(store, 3)
	qa := chain.NewRetrievalQAChain(vsRetriever, model)
	qa.ReturnSource = true

	questions := []string{
		"What version of Go introduced generics?",
		"What are three reasons to use Go?",
	}
	for _, q := range questions {
		result, err := qa.Invoke(ctx, q)
		if err != nil {
			fmt.Printf("  Q: %s\n  Error: %v\n\n", q, err)
			continue
		}
		m := result.(map[string]any)
		fmt.Printf("  Q: %s\n  A: %s\n\n", q, truncate(fmt.Sprint(m["answer"]), 120))
		_ = m["sources"].([]schema.Document)
	}

	fmt.Println()
	fmt.Println("  To use PDF loader with a real file:  doc, _ := pdf.New(\"report.pdf\").Load(ctx)")
	fmt.Println("  To use DOCX loader with a real file: doc, _ := docx.New(\"report.docx\").Load(ctx)")
}

// releaseNotes contains sample content for the demo.
const releaseNotes = `Go 1.18

Go 1.18 is a major release that introduces generics using type parameters.
Generics are one of the most significant changes to the Go language since
Go 1.0. The design is based on contracts and type lists, enabling writing
reusable, type-safe functions and data structures without sacrificing
runtime performance.

Key features:
- Type parameters for functions and types
- New package: constraints
- Interface improvements for type satisfaction

Go 1.21

Go 1.21 adds the new built-in functions min, max, and clear, and several
improvements to the standard library. The release also stabilizes the
profile-guided optimization (PGO) feature for better runtime performance.

Why use Go?
1. Simplicity — Go values clarity, simplicity, and productivity. The language
   has a deliberately small feature set.
2. Concurrency — goroutines and channels make concurrent programming natural
   and efficient.
3. Fast compilation — Go compiles quickly to a single statically-linked binary,
   simplifying deployment.`

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

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
