// This example demonstrates Runnable.Batch() — running multiple
// invocations concurrently across any Runnable (LCEL pipeline, LLMChain,
// SequentialChain, etc.).
//
//	go run ./examples/basics/batch
//
// Use "Azure AI Foundry" by default. See the comment block below to
// switch to the OpenAI API.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/joho/godotenv"

	"github.com/grafaelw/golangchain/chain"
	"github.com/grafaelw/golangchain/llm/openai"
	"github.com/grafaelw/golangchain/output"
	"github.com/grafaelw/golangchain/prompt"
)

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
	//         openai.WithModel("gpt-4o-mini"),
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
	// 2. Build a reusable LLMChain
	// ---------------------------------------------------------------------------
	translator := chain.NewLLMChain(
		prompt.MustNewChatPromptTemplate(
			prompt.MustSystem("Translate the following text to {{.language}}. Reply with ONLY the translation, no explanation."),
			prompt.MustHuman("{{.text}}"),
		),
		model,
		output.AsAny(output.StrOutputParser{}),
	)

	// ---------------------------------------------------------------------------
	// 3. Batch — run multiple translations concurrently
	//
	// Runnable.Batch() fans out all inputs concurrently with no artificial cap.
	// For rate-limited models, wrap the LLM with llmutil.NewRateLimitedLLM().
	// ---------------------------------------------------------------------------
	section("Batch — concurrent translations")

	inputs := []any{
		map[string]any{"language": "French", "text": "Hello, how are you?"},
		map[string]any{"language": "German", "text": "Good morning"},
		map[string]any{"language": "Spanish", "text": "Thank you very much"},
		map[string]any{"language": "Italian", "text": "Where is the train station?"},
		map[string]any{"language": "Dutch", "text": "I would like a coffee please"},
	}

	start := time.Now()
	results, err := translator.Batch(ctx, inputs)
	elapsed := time.Since(start)
	if err != nil {
		log.Fatalf("batch: %v", err)
	}

	for i, r := range results {
		vars := inputs[i].(map[string]any)
		fmt.Printf("  %-8s %q → %q\n",
			vars["language"], vars["text"], r)
	}
	fmt.Printf("\n  5 translations in %v (concurrent)\n\n", elapsed.Round(time.Millisecond))

	// ---------------------------------------------------------------------------
	// 4. Batch with FuncRunnable pipelines
	// ---------------------------------------------------------------------------
	section("Batch — pipeline composition")

	upper := chain.NewFuncRunnable("uppercase",
		func(_ context.Context, in any) (any, error) {
			return strings.ToUpper(in.(string)), nil
		},
	).Pipe(
		chain.NewFuncRunnable("exclaim",
			func(_ context.Context, in any) (any, error) {
				return in.(string) + "!!!", nil
			},
		),
	)

	pipelineInputs := []any{"hello", "world", "golangchain"}
	pipeResults, err := upper.Batch(ctx, pipelineInputs)
	if err != nil {
		log.Fatalf("batch pipeline: %v", err)
	}
	for i, r := range pipeResults {
		fmt.Printf("  %q → %q\n", pipelineInputs[i], r)
	}
	fmt.Println()

	// ---------------------------------------------------------------------------
	// 5. Sequential run (for comparison)
	// ---------------------------------------------------------------------------
	section("Sequential (for comparison)")

	start = time.Now()
	for _, in := range inputs {
		out, _ := translator.Invoke(ctx, in)
		_ = out
	}
	fmt.Printf("  5 translations in %v (sequential)\n", time.Since(start).Round(time.Millisecond))
}

func section(title string) {
	fmt.Println(strings.Repeat("─", 72))
	fmt.Println(title)
	fmt.Println(strings.Repeat("─", 72))
}
