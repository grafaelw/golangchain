// Example: sentinel_errors
//
// Demonstrates the sentinel errors introduced for common failure modes
// across the library.  These errors allow callers to use errors.Is() to
// take different recovery paths depending on the failure reason.
//
// Sentinels:
//
//   - tools.ErrToolNotFound — returned by FindTool when the requested tool
//     name does not exist in the agent's tool list.
//
//   - agent.ErrAgentMaxIter  — returned by AgentExecutor when the agent
//     exceeds its maximum iteration limit without producing a final answer.
//
//   - graph.ErrGraphMaxSteps — returned by CompiledGraph when the execution
//     hits the configured step limit.
//
//     Run this example with:
//     go run ./examples/sentinel_errors
package main

import (
	"errors"
	"fmt"

	"github.com/grafaelw/golangchain/agent"
	"github.com/grafaelw/golangchain/graph"
	"github.com/grafaelw/golangchain/tools"
)

func main() {
	// -------------------------------------------------------------------------
	// 1. tools.ErrToolNotFound
	// -------------------------------------------------------------------------
	fmt.Println("--- 1. tools.ErrToolNotFound ---")
	calc := tools.Calculator{}
	allTools := []tools.Tool{calc}

	found := tools.FindTool(allTools, "weather")
	if found == nil {
		fmt.Printf("  FindTool(\"weather\") -> nil\n")
		fmt.Printf("  error: %v\n", tools.ErrToolNotFound)
	}
	fmt.Printf("  errors.Is(err, tools.ErrToolNotFound) = %v\n",
		errors.Is(tools.ErrToolNotFound, tools.ErrToolNotFound))

	// Typical usage inside an agent loop:
	toolErr := fmt.Errorf("tool %q: %w", "weather", tools.ErrToolNotFound)
	fmt.Printf("  Wrapped: %v\n", toolErr)
	fmt.Printf("  errors.Is(wrapped, tools.ErrToolNotFound) = %v\n",
		errors.Is(toolErr, tools.ErrToolNotFound))

	// -------------------------------------------------------------------------
	// 2. agent.ErrAgentMaxIter
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 2. agent.ErrAgentMaxIter ---")
	fmt.Printf("  Sentinel: %v\n", agent.ErrAgentMaxIter)
	fmt.Printf("  errors.Is(agent.ErrAgentMaxIter, agent.ErrAgentMaxIter) = %v\n",
		errors.Is(agent.ErrAgentMaxIter, agent.ErrAgentMaxIter))

	agentErr := fmt.Errorf("agent stopped after 10 iterations: %w", agent.ErrAgentMaxIter)
	fmt.Printf("  Wrapped: %v\n", agentErr)
	fmt.Printf("  errors.Is(wrapped, agent.ErrAgentMaxIter) = %v\n",
		errors.Is(agentErr, agent.ErrAgentMaxIter))

	// -------------------------------------------------------------------------
	// 3. graph.ErrGraphMaxSteps
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 3. graph.ErrGraphMaxSteps ---")
	fmt.Printf("  Sentinel: %v\n", graph.ErrGraphMaxSteps)
	fmt.Printf("  errors.Is(graph.ErrGraphMaxSteps, graph.ErrGraphMaxSteps) = %v\n",
		errors.Is(graph.ErrGraphMaxSteps, graph.ErrGraphMaxSteps))

	graphErr := fmt.Errorf("graph exceeded step limit: %w", graph.ErrGraphMaxSteps)
	fmt.Printf("  Wrapped: %v\n", graphErr)
	fmt.Printf("  errors.Is(wrapped, graph.ErrGraphMaxSteps) = %v\n",
		errors.Is(graphErr, graph.ErrGraphMaxSteps))

	// -------------------------------------------------------------------------
	// 4. Practical: classify errors to decide recovery strategy
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 4. Error classification ---")
	try := func(err error) {
		switch {
		case errors.Is(err, tools.ErrToolNotFound):
			fmt.Printf("  → Remove unknown tool from agent config\n")
		case errors.Is(err, agent.ErrAgentMaxIter):
			fmt.Printf("  → Increase MaxIter or simplify the prompt\n")
		case errors.Is(err, graph.ErrGraphMaxSteps):
			fmt.Printf("  → Increase MaxSteps or reduce node count\n")
		default:
			fmt.Printf("  → Unknown error: %v\n", err)
		}
	}

	try(agentErr)
	try(toolErr)
	try(graphErr)
	try(fmt.Errorf("network timeout"))

	fmt.Println("\n✅ All sentinel errors verified.")
}
