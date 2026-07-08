// Package graph implements a LangGraph-equivalent StateGraph engine for Go.
//
// A StateGraph is a directed graph where nodes are functions, edges are
// transitions, and the generic state type S is merged via a user-supplied
// StateReducer at every step.
//
// # Core concepts
//
//   - [StateGraph][S]        — mutable builder; not thread-safe during construction.
//   - [CompiledGraph][S]     — immutable, thread-safe, executable graph.
//   - [NodeFunc][S]          — func(ctx, state S) (S, error).
//   - [StateReducer][S]      — func(current, update S) S — merges partial updates.
//   - [Checkpointer][S]      — interface for persisting and loading graph state.
//     Ships with [MemoryCheckpointer] (in-process) and [FileCheckpointer]
//     (one JSON file per checkpoint under a directory, chronological).
//   - [Interrupt]            — sentinel error returned by a node to pause execution.
//
// # Edges
//
//   - [StateGraph.AddEdge]              — unconditional transition
//   - [StateGraph.AddConditionalEdges]  — dynamic routing via a condition function
//   - [StateGraph.AddParallelEdges]     — fan-out to concurrent branches
//
// # Example: agent loop
//
//	g := graph.NewStateGraph(myReducer)
//	g.MustAddNode("agent", agentNode)
//	g.MustAddNode("tools", toolsNode)
//	g.MustAddEdge(graph.START, "agent")
//	g.MustAddConditionalEdges("agent", routerFn, map[string]string{
//	    "use_tools": "tools",
//	    "done":      graph.END,
//	})
//	g.MustAddEdge("tools", "agent") // loop back
//
//	compiled, _ := g.Compile(
//	    graph.WithCheckpointer(graph.NewMemoryCheckpointer[MyState]()),
//	    graph.WithMaxSteps[MyState](50),
//	)
//
//	// Blocking
//	finalState, _ := compiled.Invoke(ctx, initialState, graph.WithThreadID[MyState]("thread-1"))
//
//	// Streaming
//	for event := range compiled.Stream(ctx, initialState) {
//	    fmt.Println(event.Type, event.Node)
//	}
//
// # Human-in-the-loop
//
// Any node can return [NewInterrupt] to pause execution. The graph saves state
// via the [Checkpointer]. Resume by invoking again with the saved state:
//
//	_, err := compiled.Invoke(ctx, state, graph.WithThreadID[S]("t1"))
//	if interrupt, ok := err.(*graph.Interrupt); ok {
//	    // human reviews…
//	    saved, _ := checkpointer.Load(ctx, "t1")
//	    finalState, _ = compiled.Invoke(ctx, saved.State, graph.WithThreadID[S]("t1"))
//	}
package graph
