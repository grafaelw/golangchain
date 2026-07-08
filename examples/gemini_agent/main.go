// This example demonstrates a ToolCallingAgent using the Google Gemini
// provider with native function-calling support.
//
// Prerequisites:
//
//	export GEMINI_API_KEY=...
//
//	go run ./examples/gemini_agent
//
// The ToolCallingAgent sends FunctionDeclaration tool definitions to the
// Gemini API via the native function-calling protocol. The model returns
// FunctionCall parts that are dispatched to the matching tool.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/joho/godotenv"

	"github.com/grafaelw/golangchain/agent"
	"github.com/grafaelw/golangchain/llm/gemini"
	"github.com/grafaelw/golangchain/tools"
)

func main() {
	_ = godotenv.Load()

	ctx := context.Background()

	// ---------------------------------------------------------------------------
	// 1. Create the Gemini LLM provider
	// ---------------------------------------------------------------------------
	model, err := gemini.New(ctx,
		gemini.WithAPIKey(os.Getenv("GEMINI_API_KEY")),
		gemini.WithModel("gemini-2.5-flash"),
	)
	if err != nil {
		log.Fatalf("create gemini model: %v", err)
	}

	// ---------------------------------------------------------------------------
	// 2. Build the tool set
	// ---------------------------------------------------------------------------
	agentTools := []tools.Tool{
		tools.Calculator{},
		tools.NewDuckDuckGoSearch(),
	}

	// ---------------------------------------------------------------------------
	// 3. Create and run the ToolCallingAgent
	//
	// The agent converts tool definitions to Gemini FunctionDeclarations and
	// parses FunctionCall parts from Gemini's response. No ReAct text-parsing
	// needed — the structured JSON schema ensures reliable tool dispatch.
	// ---------------------------------------------------------------------------
	executor := agent.NewAgentExecutor(
		agent.NewToolCallingAgent(model, agentTools,
			"You are a helpful assistant. Use the calculator for math. "+
				"Use search for factual questions. Be concise.",
		),
		agentTools,
		agent.WithMaxIter(5),
		agent.WithVerbose(true),
	)

	questions := []string{
		"What is 2^10 + 500?",
		"Who is the CEO of Google as of 2025?",
	}

	for _, q := range questions {
		section(q)
		answer, err := executor.Run(ctx, q)
		if err != nil {
			log.Printf("ERROR: %v\n", err)
			continue
		}
		fmt.Printf("Answer: %s\n\n", answer)
	}

	// ---------------------------------------------------------------------------
	// 4. Streaming events — see each tool call as it happens
	// ---------------------------------------------------------------------------
	section("Streaming agent events")
	fmt.Println("Question: What is the square root of 65536?")
	for event := range executor.Stream(ctx, "What is the square root of 65536?") {
		switch event.Type {
		case agent.EventToolCall:
			fmt.Printf("  🔧 Calling: %s(%s)\n",
				event.Action.Tool, truncate(event.Action.ToolInput, 60))
		case agent.EventToolResult:
			fmt.Printf("  📋 Result: %s\n", truncate(event.Observation, 80))
		case agent.EventFinalAnswer:
			fmt.Printf("  ✅ Answer: %s\n", event.Answer)
		case agent.EventError:
			fmt.Printf("  ❌ Error: %v\n", event.Err)
		}
	}
}

func section(title string) {
	fmt.Println(strings.Repeat("─", 72))
	fmt.Println(title)
	fmt.Println(strings.Repeat("─", 72))
}

func truncate(s string, max int) string {
	s = strings.TrimSpace(s)
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}
