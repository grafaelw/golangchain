// This example demonstrates a ToolCallingAgent using the Anthropic Claude
// provider with native tool_use support (function calling).
//
// Prerequisites:
//
//	export ANTHROPIC_API_KEY=sk-ant-...
//
//	go run ./examples/anthropic_agent
//
// Unlike the ReAct agent (text-based tool pattern), the ToolCallingAgent
// sends tool definitions to the API and the model returns structured
// tool_use blocks — exactly like OpenAI's function-calling API.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/joho/godotenv"

	"github.com/grafaelw/golangchain/agent"
	"github.com/grafaelw/golangchain/llm/anthropic"
	"github.com/grafaelw/golangchain/tools"
)

func main() {
	_ = godotenv.Load()

	ctx := context.Background()

	// ---------------------------------------------------------------------------
	// 1. Create the Anthropic LLM provider
	// ---------------------------------------------------------------------------
	model, err := anthropic.New(
		anthropic.WithAPIKey(os.Getenv("AZURE_ANTHROPIC_FOUNDRY_API_KEY")),
		anthropic.WithModel("claude-sonnet-4-6"),
		anthropic.WithBaseURL(os.Getenv("AZURE_ANTHROPIC_FOUNDRY_BASE_URL")), // optional since this is tested from Azure Foundry
	)
	if err != nil {
		log.Fatalf("create anthropic model: %v", err)
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
	// The ToolCallingAgent sends tool_definitions to the Anthropic API as
	// native tool_use blocks. The model decides which tool(s) to call and
	// returns structured tool_use responses — no text parsing needed.
	// ---------------------------------------------------------------------------
	executor := agent.NewAgentExecutor(
		agent.NewToolCallingAgent(model, agentTools,
			"You are a helpful assistant that uses tools when needed. "+
				"For any math question, use the calculator tool. "+
				"For any factual question, use the search tool.",
		),
		agentTools,
		agent.WithMaxIter(5),
		agent.WithVerbose(true),
	)

	questions := []string{
		"What is 1337 * 42?",
		"What is the population of Amsterdam?",
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
	// 4. Streaming events
	// ---------------------------------------------------------------------------
	section("Streaming tool-call events")
	fmt.Println("Question: How many seconds are in a leap year?")
	for event := range executor.Stream(ctx, "How many seconds are in a leap year?") {
		switch event.Type {
		case agent.EventToolCall:
			fmt.Printf("  🔧 Calling: %s(%s)\n", event.Action.Tool, truncate(event.Action.ToolInput, 60))
		case agent.EventToolResult:
			fmt.Printf("  📋 Result: %s\n", truncate(event.Observation, 80))
		case agent.EventFinalAnswer:
			fmt.Printf("  ✅ Answer: %s\n", event.Answer)
		case agent.EventError:
			fmt.Printf("  ❌ Error: %v\n", event.Err)
		}
	}
	fmt.Println()
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
