// Example: memory_and_tools
//
// Demonstrates the three Memory implementations and the built-in Tool
// helpers in isolation (outside of an agent loop):
//
//   - ConversationBufferMemory  — full history, unbounded
//   - ConversationWindowMemory  — sliding window of last k turns
//   - ConversationSummaryMemory — compresses old turns via an LLM call
//   - Calculator                — recursive-descent arithmetic parser
//   - FuncTool                  — wraps any function as a Tool
//   - ToToolDefs / FindTool     — helpers for working with []Tool
//
// ConversationSummaryMemory needs a real LLM to compress old turns.
//
// # Usage — Azure AI Foundry (default)
//
// Create a .env file with:
//
//	AZURE_OPENAI_API_KEY=<your-key>
//
// Then run:
//
//	go run ./examples/memory_and_tools
//
// # Usage — OpenAI API
//
// Replace the model initialisation block with:
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
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/joho/godotenv"

	"github.com/grafaelw/golangchain/llm/openai"
	"github.com/grafaelw/golangchain/memory"
	"github.com/grafaelw/golangchain/schema"
	"github.com/grafaelw/golangchain/tools"
)

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
	// 2. Create the LLM — Azure AI Foundry via the openai package.
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

	// -------------------------------------------------------------------------
	// 3. ConversationBufferMemory — full, unbounded history
	// -------------------------------------------------------------------------
	fmt.Println("--- 3. ConversationBufferMemory ---")
	buf := memory.NewConversationBufferMemory()
	_ = buf.SaveContext(ctx, "Hi!", "Hello! How can I help?")
	_ = buf.SaveContext(ctx, "What is Go?", "Go is a statically typed compiled language.")
	_ = buf.SaveContext(ctx, "Who created it?", "Go was created at Google.")

	vars, err := buf.LoadMemoryVariables(ctx)
	if err != nil {
		panic(err)
	}
	history := vars["history"].([]schema.Message)
	fmt.Printf("Buffer: %d messages stored\n", len(history))
	for _, msg := range history {
		fmt.Printf("  %-8s %s\n", msg.Role, msg.Content)
	}

	// -------------------------------------------------------------------------
	// 4. ConversationWindowMemory (k=2) — sliding window
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 4. ConversationWindowMemory (k=2) ---")
	win := memory.NewConversationWindowMemory(2)
	for _, turn := range [][2]string{
		{"Turn 1 question", "Turn 1 answer"},
		{"Turn 2 question", "Turn 2 answer"},
		{"Turn 3 question", "Turn 3 answer"},
		{"Turn 4 question", "Turn 4 answer"},
	} {
		_ = win.SaveContext(ctx, turn[0], turn[1])
	}
	wVars, err := win.LoadMemoryVariables(ctx)
	if err != nil {
		panic(err)
	}
	wHistory := wVars["history"].([]schema.Message)
	fmt.Printf("Window (k=2): %d messages visible (2 oldest turns discarded)\n", len(wHistory))
	for _, msg := range wHistory {
		fmt.Printf("  %-8s %s\n", msg.Role, msg.Content)
	}

	// -------------------------------------------------------------------------
	// 5. ConversationSummaryMemory — compresses old turns via an LLM call
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 5. ConversationSummaryMemory ---")
	sum := memory.NewConversationSummaryMemory(model)
	sum.MaxRecent = 2 // compress after 2 turns

	_ = sum.SaveContext(ctx, "What is Go?", "A statically typed compiled language.")
	_ = sum.SaveContext(ctx, "Who made it?", "Google.")
	// This third save triggers compression (MaxRecent=2 => threshold=4 messages).
	_ = sum.SaveContext(ctx, "When was it released?", "In 2009.")
	_ = sum.SaveContext(ctx, "Is it fast?", "Yes, comparable to C.")
	_ = sum.SaveContext(ctx, "Trigger summary now", "Compressing...")

	msgs := sum.Messages()
	fmt.Printf("Summary memory: %d messages (first is the compressed summary)\n", len(msgs))
	for _, msg := range msgs {
		fmt.Printf("  %-8s %s\n", msg.Role, msg.Content)
	}

	// -------------------------------------------------------------------------
	// 6. Calculator tool
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 6. Calculator tool ---")
	calc := tools.Calculator{}
	fmt.Printf("Calculator: %s\n", calc.Description())

	exprs := []string{
		"2 + 2",
		"10 * (3 + 4)",
		"2 ^ 10",
		"sqrt(144)",
		"abs(-99)",
		"floor(3.9)",
		`{"expression":"(1 + 2) * 3"}`, // JSON input, as sent by tool-calling agents
	}
	for _, e := range exprs {
		r, err := calc.Run(ctx, e)
		if err != nil {
			panic(err)
		}
		fmt.Printf("  %-30s = %s\n", e, r)
	}

	// -------------------------------------------------------------------------
	// 7. FuncTool — wrap any function as a Tool
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 7. FuncTool ---")
	wordCountTool := tools.NewFuncTool(
		"word_count",
		"Counts the number of words in the input text.",
		json.RawMessage(`{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}`),
		func(_ context.Context, input string) (string, error) {
			var args struct {
				Text string `json:"text"`
			}
			if err := json.Unmarshal([]byte(input), &args); err != nil {
				return "", fmt.Errorf("word_count: invalid input: %w", err)
			}
			return fmt.Sprintf("%d words", len(strings.Fields(args.Text))), nil
		},
	)

	r, err := wordCountTool.Run(ctx, `{"text":"The quick brown fox jumps over the lazy dog"}`)
	if err != nil {
		panic(err)
	}
	fmt.Println("FuncTool word_count:", r)

	// -------------------------------------------------------------------------
	// 8. ToToolDefs / FindTool
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 8. ToToolDefs / FindTool ---")
	allTools := []tools.Tool{calc, wordCountTool}
	defs := tools.ToToolDefs(allTools)
	for _, d := range defs {
		fmt.Printf("ToolDef: %-12s %s\n", d.Name, truncate(d.Description, 50))
	}
	found := tools.FindTool(allTools, "calculator")
	fmt.Printf("\nFindTool(\"calculator\") -> %s\n", found.Name())
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
