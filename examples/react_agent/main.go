// Example: react_agent
//
// Demonstrates a ToolCallingAgent (or ReActAgent) with Calculator and
// DuckDuckGoSearch tools, driven by AgentExecutor with streaming.
//
// Usage:
//
//	Put OPENAI_API_KEY=sk-... in a .env file, then: go run .
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/joho/godotenv"

	"github.com/grafaelw/golangchain/agent"
	"github.com/grafaelw/golangchain/callbacks"
	"github.com/grafaelw/golangchain/llm/openai"
	"github.com/grafaelw/golangchain/tools"
)

func main() {
	ctx := context.Background()

	// -------------------------------------------------------------------------
	// 1. Load .env (silently ignored if the file doesn't exist,
	// 	  so real environment variables still work in CI/production)

	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found - using environment variables")
	}

	// -------------------------------------------------------------------------
	// 2. Create the LLM
	// -------------------------------------------------------------------------
	model, err := openai.New(
		openai.WithAPIKey(os.Getenv("AZURE_OPENAI_API_KEY")),
		openai.WithModel("gpt-5.4-nano"),
		openai.WithBaseURL(os.Getenv("OTHER_MODELS_ENDPOINT")),
	)
	if err != nil {
		log.Fatal(err)
	}

	// -------------------------------------------------------------------------
	// 3. Define tools
	// -------------------------------------------------------------------------
	agentTools := []tools.Tool{
		tools.Calculator{},
		tools.NewDuckDuckGoSearch(),
		tools.NewHTTPFetch(),
	}

	// -------------------------------------------------------------------------
	// 4. Create a ToolCallingAgent (uses native function-calling API)
	// -------------------------------------------------------------------------
	systemPrompt := `You are a helpful research assistant. Use tools when you need
real-time information or to perform calculations. Always show your reasoning.`

	toolAgent := agent.NewToolCallingAgent(model, agentTools, systemPrompt)

	// -------------------------------------------------------------------------
	// 5. Wire up a logging callback
	// -------------------------------------------------------------------------
	cb := callbacks.NewCallbackManager(
		callbacks.NewLoggingHandler(log.Printf),
	)

	// -------------------------------------------------------------------------
	// 6. Create AgentExecutor
	// -------------------------------------------------------------------------
	executor := agent.NewAgentExecutor(
		toolAgent,
		agentTools,
		agent.WithCallbackManager(cb),
		agent.WithMaxIter(8),
		agent.WithVerbose(true),
	)

	// -------------------------------------------------------------------------
	// 7. Run queries
	// -------------------------------------------------------------------------
	queries := []string{
		"What is 1337 * 42 + sqrt(144)?",
		"What is the current population of Amsterdam?",
	}

	for _, q := range queries {
		fmt.Printf("\n=== Query: %s ===\n", q)

		// Stream events in real time
		for event := range executor.Stream(ctx, q) {
			switch event.Type {
			case agent.EventThought:
				fmt.Printf("[Thought] %s\n", event.Thought)
			case agent.EventToolCall:
				fmt.Printf("[Tool Call] %s(%s)\n", event.Action.Tool, event.Action.ToolInput)
			case agent.EventToolResult:
				truncated := event.Observation
				if len(truncated) > 200 {
					truncated = truncated[:200] + "..."
				}
				fmt.Printf("[Observation] %s\n", truncated)
			case agent.EventFinalAnswer:
				fmt.Printf("[Answer] %s\n", event.Answer)
			case agent.EventError:
				fmt.Printf("[Error] %v\n", event.Err)
			}
		}
	}

	// -------------------------------------------------------------------------
	// 8. Demonstrate ReActAgent (text-based, works with any LLM)
	// -------------------------------------------------------------------------
	fmt.Println("\n=== ReAct Agent (text-based) ===")
	reactAgent := agent.NewReActAgent(model, agentTools)
	reactExecutor := agent.NewAgentExecutor(reactAgent, agentTools, agent.WithMaxIter(6))

	answer, err := reactExecutor.Run(ctx, "Calculate (100 + 50) * 3 and tell me if it's greater than 400.")
	if err != nil {
		log.Printf("ReAct error: %v", err)
	} else {
		fmt.Printf("ReAct Answer: %s\n", answer)
	}
}
