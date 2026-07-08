// Example: tracing
//
// Demonstrates the golangchain tracing system — a LangSmith-inspired
// observability layer that plugs in via a single CallbackManager with zero
// changes to your existing chains or agents.
//
// Two handlers are composed in one CallbackManager:
//
//  1. tracing.PrettyHandler  — prints a live, colour-coded trace to stderr
//     as each operation executes (LLM calls, tools, chains, graph nodes …).
//
//  2. tracing.Tracer/TracerHandler — records every run into an in-memory
//     tree for programmatic inspection, JSON export, and token summaries.
//
// Three scenarios showcase different nesting depths:
//
//   - Scenario A: LLMChain          (Chain → LLM)
//   - Scenario B: ToolCallingAgent  (Agent → LLM → Tool → LLM → …)
//   - Scenario C: StateGraph        (Graph Node → LLM, Graph Node → LLM)
//
// # Usage — Azure AI Foundry (default)
//
// Create a .env file with:
//
//	AZURE_OPENAI_API_KEY=<your-key>
//
// Then run:
//
//	go run ./examples/tracing
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
	"fmt"
	"log"
	"os"

	"github.com/joho/godotenv"

	"github.com/grafaelw/golangchain/agent"
	"github.com/grafaelw/golangchain/callbacks"
	"github.com/grafaelw/golangchain/chain"
	"github.com/grafaelw/golangchain/graph"
	"github.com/grafaelw/golangchain/llm/openai"
	"github.com/grafaelw/golangchain/output"
	"github.com/grafaelw/golangchain/prompt"
	"github.com/grafaelw/golangchain/schema"
	"github.com/grafaelw/golangchain/tools"
	"github.com/grafaelw/golangchain/tracing"
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
	// 2. Build the LLM — Azure AI Foundry via the openai package.
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
	// 3. Set up the shared CallbackManager
	//
	//    PrettyHandler  → live colour output to stderr (LangSmith-style)
	//    TracerHandler  → programmatic run tree recorded in memory
	// -------------------------------------------------------------------------
	tracer := tracing.NewTracer()
	cb := callbacks.NewCallbackManager(
		tracing.NewPrettyHandler(os.Stderr),
		tracer.Handler(),
	)

	// =========================================================================
	// Scenario A — LLMChain  (Chain → LLM)
	// =========================================================================
	section("Scenario A: LLMChain  (Chain → LLM)")
	tracer.Reset()

	chatPrompt := prompt.MustNewChatPromptTemplate(
		prompt.MustSystem("You are a concise assistant. Answer in one sentence."),
		prompt.MustHuman("{{.question}}"),
	)

	llmChain := chain.NewLLMChain(
		chatPrompt,
		model,
		output.AsAny(output.StrOutputParser{}),
		chain.WithChainName("SimpleQAChain"),
		chain.WithChainCallbacks(cb),
	)

	result, err := llmChain.Invoke(ctx, map[string]any{
		"question": "What is the capital of Japan?",
	})
	if err != nil {
		log.Printf("chain error: %v", err)
	} else {
		fmt.Printf("\nAnswer: %v\n", result)
	}

	printSummary(tracer)

	// =========================================================================
	// Scenario B — ToolCallingAgent  (Agent → LLM → Tool → LLM)
	// =========================================================================
	section("Scenario B: ToolCallingAgent  (Agent → LLM → Tool → LLM)")
	tracer.Reset()

	agentTools := []tools.Tool{
		tools.Calculator{},
		tools.NewDuckDuckGoSearch(),
	}

	toolAgent := agent.NewToolCallingAgent(
		model,
		agentTools,
		"You are a helpful assistant. Use tools when needed.",
	)
	executor := agent.NewAgentExecutor(
		toolAgent,
		agentTools,
		agent.WithCallbackManager(cb),
		agent.WithMaxIter(6),
	)

	agentAnswer, err := executor.Run(ctx, "What is 1337 multiplied by 42?")
	if err != nil {
		log.Printf("agent error: %v", err)
	} else {
		fmt.Printf("\nAgent answer: %s\n", agentAnswer)
	}

	printSummary(tracer)

	// =========================================================================
	// Scenario C — StateGraph  (Graph Node → LLM, Graph Node → LLM)
	// =========================================================================
	section("Scenario C: StateGraph  (Node → LLM, Node → LLM)")
	tracer.Reset()

	type State struct {
		Messages []schema.Message
	}

	reducer := func(cur, upd State) State {
		cur.Messages = append(cur.Messages, upd.Messages...)
		return cur
	}

	// Helper: build a node that calls the LLM with a fixed system prompt.
	makeNode := func(sysPrompt string) graph.NodeFunc[State] {
		return func(ctx context.Context, state State) (State, error) {
			msgs := append(
				[]schema.Message{schema.NewSystemMessage(sysPrompt)},
				state.Messages...,
			)
			gen, genErr := model.Generate(ctx, msgs)
			if genErr != nil {
				return state, genErr
			}
			return State{
				Messages: []schema.Message{schema.NewAIMessage(gen.Text)},
			}, nil
		}
	}

	g := graph.NewStateGraph(reducer).WithName("TracingDemo")
	g.MustAddNode("summarise", makeNode(
		"Summarise the user's question in one concise sentence."))
	g.MustAddNode("answer", makeNode(
		"Using the summary provided, give a short, direct answer."))
	g.MustAddEdge(graph.START, "summarise")
	g.MustAddEdge("summarise", "answer")
	g.MustAddEdge("answer", graph.END)

	compiled, err := g.Compile(
		graph.WithGraphCallbacks[State](cb),
	)
	if err != nil {
		log.Fatalf("compile: %v", err)
	}

	finalState, err := compiled.Invoke(ctx, State{
		Messages: []schema.Message{
			schema.NewHumanMessage("Why is the sky blue?"),
		},
	})
	if err != nil {
		log.Printf("graph error: %v", err)
	} else {
		last := finalState.Messages[len(finalState.Messages)-1]
		fmt.Printf("\nGraph answer: %s\n", last.Content)
	}

	printSummary(tracer)

	// =========================================================================
	// JSON export — full run tree from the last scenario
	// =========================================================================
	section("JSON Export  (last scenario's run tree)")
	if err := tracer.ExportJSON(os.Stdout); err != nil {
		log.Printf("export error: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// section prints a visible divider to stderr.
func section(title string) {
	fmt.Fprintf(os.Stderr,
		"\n\033[1;37m══════════════════════════════════════════\n  %s\n══════════════════════════════════════════\033[0m\n\n",
		title)
}

// printSummary prints the tracer's text summary and total token usage.
func printSummary(t *tracing.Tracer) {
	fmt.Fprintf(os.Stderr, "\n\033[90m── Tracer summary ──────────────────────\033[0m\n")
	fmt.Fprint(os.Stderr, t.Summary())

	tok := t.TotalTokens()
	if tok.TotalTokens > 0 {
		fmt.Fprintf(os.Stderr,
			"\033[90mTokens: ↑%d prompt  ↓%d completion  %d total\033[0m\n",
			tok.PromptTokens, tok.CompletionTokens, tok.TotalTokens)
	}

}
