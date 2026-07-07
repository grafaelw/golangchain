// Example: state_graph
//
// Demonstrates a LangGraph-equivalent StateGraph with:
//   - A multi-node agent loop (agent → tools → agent → END)
//   - Conditional routing
//   - Human-in-the-loop interrupt
//   - Checkpointing with MemoryCheckpointer
//   - Streaming GraphEvents
//
// Usage:
//
//	Put OPENAI_API_KEY=sk-... in a .env file, then: go run .
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/joho/godotenv"

	"github.com/grafaelw/golangchain/graph"
	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/llm/openai"
	"github.com/grafaelw/golangchain/schema"
	"github.com/grafaelw/golangchain/tools"
)

// ---------------------------------------------------------------------------
// State definition
// ---------------------------------------------------------------------------

// AgentState is the shared state threaded through every graph node.
// The StateReducer appends messages and overwrites Next.
type AgentState struct {
	Messages []schema.Message `json:"messages"`
	Next     string           `json:"next"` // routing signal set by agent node
}

// ---------------------------------------------------------------------------
// Tools available to the agent
// ---------------------------------------------------------------------------

var agentTools = []tools.Tool{
	tools.Calculator{},
	tools.NewDuckDuckGoSearch(),
}

// ---------------------------------------------------------------------------
// Node: agent
// Calls the LLM with tool definitions; sets state.Next based on response.
// ---------------------------------------------------------------------------

func makeAgentNode(model llm.LLM) graph.NodeFunc[AgentState] {
	toolDefs := tools.ToToolDefs(agentTools)

	return func(ctx context.Context, state AgentState) (AgentState, error) {
		system := schema.NewSystemMessage(
			"You are a helpful assistant. Use tools when needed to answer the user. " +
				"When you have a final answer, respond with plain text (no tool calls).")

		msgs := append([]schema.Message{system}, state.Messages...)

		gen, err := model.Generate(ctx, msgs,
			llm.WithTools(toolDefs...),
			llm.WithTemperature(0),
		)
		if err != nil {
			return state, fmt.Errorf("agent node: %w", err)
		}

		// Determine routing
		var next string
		var aiMsg schema.Message
		if len(gen.Message.ToolCalls) > 0 {
			next = "tools"
			aiMsg = gen.Message
		} else {
			next = "end"
			aiMsg = schema.NewAIMessage(gen.Text)
		}

		return AgentState{
			Messages: []schema.Message{aiMsg},
			Next:     next,
		}, nil
	}
}

// ---------------------------------------------------------------------------
// Node: tools
// Dispatches tool calls from the last AI message and appends observations.
// ---------------------------------------------------------------------------

func toolsNode(ctx context.Context, state AgentState) (AgentState, error) {
	// Find the last AI message with tool calls
	var lastAI schema.Message
	for i := len(state.Messages) - 1; i >= 0; i-- {
		if state.Messages[i].Role == schema.RoleAI && len(state.Messages[i].ToolCalls) > 0 {
			lastAI = state.Messages[i]
			break
		}
	}

	if len(lastAI.ToolCalls) == 0 {
		return AgentState{}, nil // no tool calls to dispatch
	}

	var toolMsgs []schema.Message
	for _, tc := range lastAI.ToolCalls {
		t := tools.FindTool(agentTools, tc.Name)
		if t == nil {
			toolMsgs = append(toolMsgs, schema.NewToolMessage(
				fmt.Sprintf("Error: tool %q not found", tc.Name), tc.ID, tc.Name,
			))
			continue
		}
		result, err := t.Run(ctx, string(tc.Arguments))
		if err != nil {
			result = "Error: " + err.Error()
		}
		toolMsgs = append(toolMsgs, schema.NewToolMessage(result, tc.ID, tc.Name))
	}

	return AgentState{Messages: toolMsgs}, nil
}

// ---------------------------------------------------------------------------
// Conditional routing function
// ---------------------------------------------------------------------------

func routeAgent(_ context.Context, state AgentState) string {
	return state.Next // "tools" | "end"
}

// ---------------------------------------------------------------------------
// StateReducer
// ---------------------------------------------------------------------------

func reducer(current, update AgentState) AgentState {
	current.Messages = append(current.Messages, update.Messages...)
	if update.Next != "" {
		current.Next = update.Next
	}
	return current
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

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
	// 2. Build LLM
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
	// 3. Build graph
	// -------------------------------------------------------------------------
	g := graph.NewStateGraph(reducer).WithName("AgentGraph")

	g.MustAddNode("agent", makeAgentNode(model))
	g.MustAddNode("tools", toolsNode)

	g.MustAddEdge(graph.START, "agent")
	g.MustAddConditionalEdges("agent", routeAgent, map[string]string{
		"tools": "tools",
		"end":   graph.END,
	})
	g.MustAddEdge("tools", "agent") // loop: agent can call tools multiple times

	// -------------------------------------------------------------------------
	// 4. Compile with checkpointer
	// -------------------------------------------------------------------------
	checkpointer := graph.NewMemoryCheckpointer[AgentState]()
	compiled, err := g.Compile(
		graph.WithCheckpointer[AgentState](checkpointer),
		graph.WithMaxSteps[AgentState](20),
	)
	if err != nil {
		log.Fatalf("compile: %v", err)
	}

	// -------------------------------------------------------------------------
	// 5. Run a query with streaming
	// -------------------------------------------------------------------------
	query := "What is 1234 * 5678? Also search for what DuckDuckGo is."
	fmt.Printf("Query: %s\n\n", query)

	initialState := AgentState{
		Messages: []schema.Message{schema.NewHumanMessage(query)},
	}

	threadID := "example-thread-1"

	eventStream := compiled.Stream(ctx, initialState,
		graph.WithThreadID[AgentState](threadID),
	)

	for event := range eventStream {
		switch event.Type {
		case graph.GraphEventNodeStart:
			fmt.Printf("[→ NODE START] %s\n", event.Node)
		case graph.GraphEventNodeEnd:
			fmt.Printf("[✓ NODE END] %s\n", event.Node)
			// Print last message added by node
			msgs := event.State.Messages
			if len(msgs) > 0 {
				last := msgs[len(msgs)-1]
				content := last.Content
				if len(content) > 120 {
					content = content[:120] + "..."
				}
				if content != "" {
					fmt.Printf("   → %s: %s\n", last.Role, content)
				}
				if len(last.ToolCalls) > 0 {
					for _, tc := range last.ToolCalls {
						fmt.Printf("   → tool_call: %s(%s)\n", tc.Name, string(tc.Arguments))
					}
				}
			}
		case graph.GraphEventCheckpoint:
			fmt.Printf("[💾 CHECKPOINT] thread=%s\n", threadID)
		case graph.GraphEventEnd:
			fmt.Printf("\n[✅ DONE]\n")
			// Print the final answer
			for i := len(event.State.Messages) - 1; i >= 0; i-- {
				msg := event.State.Messages[i]
				if msg.Role == schema.RoleAI && msg.Content != "" {
					fmt.Printf("\nFinal Answer: %s\n", msg.Content)
					break
				}
			}
		case graph.GraphEventError:
			log.Fatalf("[ERROR] %v", event.Err)
		}
	}

	// -------------------------------------------------------------------------
	// 6. Demonstrate checkpoint listing
	// -------------------------------------------------------------------------
	fmt.Println("\n--- Checkpoint history ---")
	history, _ := checkpointer.List(ctx, threadID)
	for i, cp := range history {
		msgCount := len(cp.State.Messages)
		fmt.Printf("  [%d] %s — %d messages\n", i+1, cp.CreatedAt.Format("15:04:05.000"), msgCount)
	}

	// -------------------------------------------------------------------------
	// 7. Demonstrate human-in-the-loop via Interrupt
	// -------------------------------------------------------------------------
	fmt.Println("\n--- Human-in-the-loop demo ---")
	hiloGraph := graph.NewStateGraph(reducer)
	hiloGraph.MustAddNode("ask_human", func(ctx context.Context, state AgentState) (AgentState, error) {
		// Simulates a node that pauses for human approval
		fmt.Println("[Node: ask_human] Requesting human approval before proceeding...")
		return state, graph.NewInterrupt("Waiting for human approval")
	})
	hiloGraph.MustAddNode("final", func(_ context.Context, state AgentState) (AgentState, error) {
		return AgentState{
			Messages: []schema.Message{schema.NewAIMessage("Human approved! Proceeding.")},
		}, nil
	})
	hiloGraph.MustAddEdge(graph.START, "ask_human")
	hiloGraph.MustAddEdge("ask_human", "final")
	hiloGraph.MustAddEdge("final", graph.END)

	hiloCheckpointer := graph.NewMemoryCheckpointer[AgentState]()
	hiloCompiled, _ := hiloGraph.Compile(
		graph.WithCheckpointer[AgentState](hiloCheckpointer),
	)

	hiloThread := "hilo-thread-1"
	startState := AgentState{Messages: []schema.Message{schema.NewHumanMessage("Please do the thing.")}}

	// First run — will be interrupted
	_, interruptErr := hiloCompiled.Invoke(ctx, startState, graph.WithThreadID[AgentState](hiloThread))
	if interruptErr != nil {
		fmt.Printf("Run paused: %v\n", interruptErr)
	}

	// Simulate human approval, then resume from checkpoint
	fmt.Println("[Human] Approved! Resuming...")
	savedCp, _ := hiloCheckpointer.Load(ctx, hiloThread)
	if savedCp != nil {
		finalState, err := hiloCompiled.Invoke(ctx, savedCp.State, graph.WithThreadID[AgentState](hiloThread))
		if err != nil {
			log.Printf("Resume error: %v", err)
		} else {
			lastMsg := finalState.Messages[len(finalState.Messages)-1]
			fmt.Printf("Resumed result: %s\n", lastMsg.Content)
		}
	}

	// -------------------------------------------------------------------------
	// 8. Parallel branches demo
	// -------------------------------------------------------------------------
	fmt.Println("\n--- Parallel branches demo ---")
	parallelGraph := graph.NewStateGraph(func(cur, upd AgentState) AgentState {
		cur.Messages = append(cur.Messages, upd.Messages...)
		return cur
	})

	addSummaryNode := func(name, prefix string) {
		parallelGraph.MustAddNode(name, func(_ context.Context, state AgentState) (AgentState, error) {
			// Simulate work: just tag the message
			return AgentState{
				Messages: []schema.Message{
					schema.NewAIMessage(prefix + ": processed in parallel"),
				},
			}, nil
		})
	}

	addSummaryNode("branch_a", "Branch A")
	addSummaryNode("branch_b", "Branch B")
	addSummaryNode("branch_c", "Branch C")
	parallelGraph.MustAddNode("collector", func(_ context.Context, state AgentState) (AgentState, error) {
		// Collect results from branches
		var parts []string
		for _, m := range state.Messages {
			if m.Role == schema.RoleAI {
				parts = append(parts, m.Content)
			}
		}
		return AgentState{
			Messages: []schema.Message{
				schema.NewAIMessage("Collected: " + strings.Join(parts, " | ")),
			},
		}, nil
	})

	parallelGraph.MustAddEdge(graph.START, "branch_a")
	// Fan-out to three branches in parallel
	_ = parallelGraph.AddParallelEdges("branch_a", []string{"branch_b", "branch_c"})
	parallelGraph.MustAddEdge("branch_b", "collector")
	parallelGraph.MustAddEdge("branch_c", "collector")
	parallelGraph.MustAddEdge("collector", graph.END)

	pCompiled, err := parallelGraph.Compile()
	if err != nil {
		log.Printf("parallel compile error: %v", err)
	} else {
		pResult, err := pCompiled.Invoke(ctx, AgentState{
			Messages: []schema.Message{schema.NewHumanMessage("run parallel")},
		})
		if err != nil {
			log.Printf("parallel run error: %v", err)
		} else {
			data, _ := json.MarshalIndent(pResult.Messages, "", "  ")
			fmt.Printf("Parallel result:\n%s\n", data)
		}
	}
}
