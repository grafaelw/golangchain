package graph

import "context"

// ---------------------------------------------------------------------------
// SubgraphNode — use a CompiledGraph as a node in a parent graph
// ---------------------------------------------------------------------------

// SubgraphNode wraps a CompiledGraph as a NodeFunc so it can be used as a
// regular node in a parent graph. The subgraph executes its own internal
// flow (with its own checkpointing, max steps, etc.) and returns the final
// state, which the parent graph merges via its reducer.
func SubgraphNode[S any](subgraph *CompiledGraph[S]) NodeFunc[S] {
	return func(ctx context.Context, state S) (S, error) {
		return subgraph.Invoke(ctx, state)
	}
}
