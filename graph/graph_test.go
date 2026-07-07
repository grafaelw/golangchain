package graph_test

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grafaelw/golangchain/graph"
)

// ---------------------------------------------------------------------------
// Simple state type used across tests
// ---------------------------------------------------------------------------

type testState struct {
	Count int
	Log   []string
	Next  string
}

func reducer(cur, upd testState) testState {
	cur.Count += upd.Count
	cur.Log = append(cur.Log, upd.Log...)
	if upd.Next != "" {
		cur.Next = upd.Next
	}
	return cur
}

// ---------------------------------------------------------------------------
// Basic graph compilation
// ---------------------------------------------------------------------------

func TestCompile_NoEntryEdge(t *testing.T) {
	g := graph.NewStateGraph(reducer)
	g.MustAddNode("a", func(_ context.Context, s testState) (testState, error) { return s, nil })
	// No edge from START — should fail
	_, err := g.Compile()
	if err == nil {
		t.Error("expected compile error for missing START edge")
	}
}

func TestCompile_UnknownDestination(t *testing.T) {
	g := graph.NewStateGraph(reducer)
	g.MustAddNode("a", func(_ context.Context, s testState) (testState, error) { return s, nil })
	g.MustAddEdge(graph.START, "a")
	g.MustAddEdge("a", "nonexistent") // unknown destination
	_, err := g.Compile()
	if err == nil {
		t.Error("expected compile error for unknown destination")
	}
}

func TestCompile_DuplicateNode(t *testing.T) {
	g := graph.NewStateGraph(reducer)
	if err := g.AddNode("a", func(_ context.Context, s testState) (testState, error) { return s, nil }); err != nil {
		t.Fatalf("first AddNode: %v", err)
	}
	if err := g.AddNode("a", func(_ context.Context, s testState) (testState, error) { return s, nil }); err == nil {
		t.Error("expected error for duplicate node name")
	}
}

func TestCompile_ReservedNames(t *testing.T) {
	g := graph.NewStateGraph(reducer)
	if err := g.AddNode(graph.START, func(_ context.Context, s testState) (testState, error) { return s, nil }); err == nil {
		t.Error("expected error for reserved node name START")
	}
	if err := g.AddNode(graph.END, func(_ context.Context, s testState) (testState, error) { return s, nil }); err == nil {
		t.Error("expected error for reserved node name END")
	}
}

// ---------------------------------------------------------------------------
// Linear graph: START → a → b → END
// ---------------------------------------------------------------------------

func TestInvoke_Linear(t *testing.T) {
	g := graph.NewStateGraph(reducer)
	g.MustAddNode("a", func(_ context.Context, s testState) (testState, error) {
		return testState{Count: 1, Log: []string{"a"}}, nil
	})
	g.MustAddNode("b", func(_ context.Context, s testState) (testState, error) {
		return testState{Count: 2, Log: []string{"b"}}, nil
	})
	g.MustAddEdge(graph.START, "a")
	g.MustAddEdge("a", "b")
	g.MustAddEdge("b", graph.END)

	compiled, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}

	result, err := compiled.Invoke(context.Background(), testState{})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if result.Count != 3 {
		t.Errorf("Count: want 3, got %d", result.Count)
	}
	if len(result.Log) != 2 {
		t.Errorf("Log: want 2 entries, got %v", result.Log)
	}
}

// ---------------------------------------------------------------------------
// Conditional routing: START → router → (even | odd) → END
// ---------------------------------------------------------------------------

func TestInvoke_ConditionalEdges(t *testing.T) {
	type numState struct {
		N      int
		Branch string
	}
	numReducer := func(cur, upd numState) numState {
		if upd.N != 0 {
			cur.N = upd.N
		}
		if upd.Branch != "" {
			cur.Branch = upd.Branch
		}
		return cur
	}

	g := graph.NewStateGraph(numReducer)
	g.MustAddNode("check", func(_ context.Context, s numState) (numState, error) {
		if s.N%2 == 0 {
			return numState{Branch: "even"}, nil
		}
		return numState{Branch: "odd"}, nil
	})
	g.MustAddNode("even_node", func(_ context.Context, s numState) (numState, error) {
		return numState{N: s.N * 10}, nil
	})
	g.MustAddNode("odd_node", func(_ context.Context, s numState) (numState, error) {
		return numState{N: s.N + 1}, nil
	})

	g.MustAddEdge(graph.START, "check")
	g.MustAddConditionalEdges("check",
		func(_ context.Context, s numState) string { return s.Branch },
		map[string]string{"even": "even_node", "odd": "odd_node"},
	)
	g.MustAddEdge("even_node", graph.END)
	g.MustAddEdge("odd_node", graph.END)

	compiled, _ := g.Compile()

	// Even input
	r1, err := compiled.Invoke(context.Background(), numState{N: 4})
	if err != nil {
		t.Fatalf("even Invoke: %v", err)
	}
	if r1.N != 40 {
		t.Errorf("even: want N=40, got %d", r1.N)
	}

	// Odd input
	r2, err := compiled.Invoke(context.Background(), numState{N: 3})
	if err != nil {
		t.Fatalf("odd Invoke: %v", err)
	}
	if r2.N != 4 {
		t.Errorf("odd: want N=4, got %d", r2.N)
	}
}

// ---------------------------------------------------------------------------
// Conditional edge with wildcard fallback
// ---------------------------------------------------------------------------

func TestConditionalEdge_WildcardFallback(t *testing.T) {
	g := graph.NewStateGraph(reducer)
	g.MustAddNode("router", func(_ context.Context, s testState) (testState, error) {
		return testState{Next: "unknown_key"}, nil
	})
	g.MustAddNode("fallback", func(_ context.Context, s testState) (testState, error) {
		return testState{Log: []string{"fallback_hit"}}, nil
	})
	g.MustAddEdge(graph.START, "router")
	g.MustAddConditionalEdges("router",
		func(_ context.Context, s testState) string { return s.Next },
		map[string]string{"*": "fallback"},
	)
	g.MustAddEdge("fallback", graph.END)

	compiled, _ := g.Compile()
	result, err := compiled.Invoke(context.Background(), testState{})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if len(result.Log) == 0 || result.Log[0] != "fallback_hit" {
		t.Errorf("wildcard fallback not triggered: %v", result.Log)
	}
}

// ---------------------------------------------------------------------------
// Cyclic graph: START → agent → (tools|END), tools → agent (loop)
// ---------------------------------------------------------------------------

func TestInvoke_Cycle(t *testing.T) {
	type cycleState struct {
		Iter int
		Done bool
	}
	cycleReducer := func(cur, upd cycleState) cycleState {
		cur.Iter += upd.Iter
		if upd.Done {
			cur.Done = true
		}
		return cur
	}

	g := graph.NewStateGraph(cycleReducer)
	g.MustAddNode("agent", func(_ context.Context, s cycleState) (cycleState, error) {
		if s.Iter >= 3 {
			return cycleState{Done: true}, nil
		}
		return cycleState{Iter: 1}, nil
	})
	g.MustAddEdge(graph.START, "agent")
	g.MustAddConditionalEdges("agent",
		func(_ context.Context, s cycleState) string {
			if s.Done {
				return "done"
			}
			return "continue"
		},
		map[string]string{
			"done":     graph.END,
			"continue": "agent", // cycle back
		},
	)

	compiled, err := g.Compile(graph.WithMaxSteps[cycleState](20))
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	result, err := compiled.Invoke(context.Background(), cycleState{})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if !result.Done {
		t.Error("expected Done=true after loop")
	}
	if result.Iter < 3 {
		t.Errorf("expected Iter >= 3, got %d", result.Iter)
	}
}

// ---------------------------------------------------------------------------
// MaxSteps guard
// ---------------------------------------------------------------------------

func TestInvoke_MaxStepsExceeded(t *testing.T) {
	g := graph.NewStateGraph(reducer)
	g.MustAddNode("loop", func(_ context.Context, s testState) (testState, error) {
		return testState{Count: 1}, nil
	})
	g.MustAddEdge(graph.START, "loop")
	g.MustAddEdge("loop", "loop") // infinite loop

	compiled, _ := g.Compile(graph.WithMaxSteps[testState](5))
	_, err := compiled.Invoke(context.Background(), testState{})
	if err == nil {
		t.Error("expected max-steps error")
	}
}

// ---------------------------------------------------------------------------
// Node error propagation
// ---------------------------------------------------------------------------

func TestInvoke_NodeError(t *testing.T) {
	g := graph.NewStateGraph(reducer)
	g.MustAddNode("bad", func(_ context.Context, s testState) (testState, error) {
		return s, fmt.Errorf("node exploded")
	})
	g.MustAddEdge(graph.START, "bad")
	g.MustAddEdge("bad", graph.END)

	compiled, _ := g.Compile()
	_, err := compiled.Invoke(context.Background(), testState{})
	if err == nil {
		t.Error("expected error from node")
	}
}

// ---------------------------------------------------------------------------
// Context cancellation
// ---------------------------------------------------------------------------

func TestInvoke_ContextCancelled(t *testing.T) {
	g := graph.NewStateGraph(reducer)
	g.MustAddNode("slow", func(ctx context.Context, s testState) (testState, error) {
		select {
		case <-ctx.Done():
			return s, ctx.Err()
		case <-time.After(5 * time.Second):
			return testState{Count: 1}, nil
		}
	})
	g.MustAddEdge(graph.START, "slow")
	g.MustAddEdge("slow", graph.END)

	compiled, _ := g.Compile()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := compiled.Invoke(ctx, testState{})
	if err == nil {
		t.Error("expected cancellation error")
	}
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

func TestStream_Events(t *testing.T) {
	g := graph.NewStateGraph(reducer)
	g.MustAddNode("a", func(_ context.Context, s testState) (testState, error) {
		return testState{Log: []string{"a"}}, nil
	})
	g.MustAddEdge(graph.START, "a")
	g.MustAddEdge("a", graph.END)

	compiled, _ := g.Compile()
	ch := compiled.Stream(context.Background(), testState{})

	var events []graph.GraphEventType
	for ev := range ch {
		events = append(events, ev.Type)
	}

	has := func(t graph.GraphEventType) bool {
		for _, e := range events {
			if e == t {
				return true
			}
		}
		return false
	}

	if !has(graph.GraphEventNodeStart) {
		t.Error("missing NodeStart event")
	}
	if !has(graph.GraphEventNodeEnd) {
		t.Error("missing NodeEnd event")
	}
	if !has(graph.GraphEventEnd) {
		t.Error("missing End event")
	}
}

// ---------------------------------------------------------------------------
// Checkpointing
// ---------------------------------------------------------------------------

func TestCheckpointing_SaveAndLoad(t *testing.T) {
	cp := graph.NewMemoryCheckpointer[testState]()
	ctx := context.Background()

	checkpoint := graph.Checkpoint[testState]{
		ThreadID:  "t1",
		State:     testState{Count: 42, Log: []string{"saved"}},
		CreatedAt: time.Now(),
	}

	if err := cp.Save(ctx, "t1", checkpoint); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := cp.Load(ctx, "t1")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded == nil {
		t.Fatal("expected non-nil checkpoint")
	}
	if loaded.State.Count != 42 {
		t.Errorf("Count: want 42, got %d", loaded.State.Count)
	}
}

func TestCheckpointing_List(t *testing.T) {
	cp := graph.NewMemoryCheckpointer[testState]()
	ctx := context.Background()

	for i := 0; i < 3; i++ {
		cp.Save(ctx, "t1", graph.Checkpoint[testState]{
			ThreadID: "t1",
			State:    testState{Count: i},
		})
	}

	list, err := cp.List(ctx, "t1")
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(list) != 3 {
		t.Errorf("want 3 checkpoints, got %d", len(list))
	}
}

func TestCheckpointing_LoadEmpty(t *testing.T) {
	cp := graph.NewMemoryCheckpointer[testState]()
	loaded, err := cp.Load(context.Background(), "nonexistent")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded != nil {
		t.Error("expected nil for nonexistent thread")
	}
}

func TestCheckpointing_Delete(t *testing.T) {
	cp := graph.NewMemoryCheckpointer[testState]()
	ctx := context.Background()
	cp.Save(ctx, "t1", graph.Checkpoint[testState]{ThreadID: "t1", State: testState{}})
	cp.Delete(ctx, "t1")
	loaded, _ := cp.Load(ctx, "t1")
	if loaded != nil {
		t.Error("expected nil after delete")
	}
}

func TestCheckpointing_GraphIntegration(t *testing.T) {
	cp := graph.NewMemoryCheckpointer[testState]()
	g := graph.NewStateGraph(reducer)
	g.MustAddNode("step", func(_ context.Context, s testState) (testState, error) {
		return testState{Count: 1, Log: []string{"stepped"}}, nil
	})
	g.MustAddEdge(graph.START, "step")
	g.MustAddEdge("step", graph.END)

	compiled, _ := g.Compile(graph.WithCheckpointer[testState](cp))
	_, err := compiled.Invoke(context.Background(), testState{},
		graph.WithThreadID[testState]("thread-abc"),
	)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}

	// At least one checkpoint should be saved
	list, _ := cp.List(context.Background(), "thread-abc")
	if len(list) == 0 {
		t.Error("expected at least one checkpoint saved")
	}
}

// ---------------------------------------------------------------------------
// Human-in-the-loop via Interrupt
// ---------------------------------------------------------------------------

func TestInterrupt_PausesExecution(t *testing.T) {
	g := graph.NewStateGraph(reducer)
	var ran atomic.Int32
	g.MustAddNode("pause", func(_ context.Context, s testState) (testState, error) {
		ran.Add(1)
		return s, graph.NewInterrupt("need approval")
	})
	g.MustAddNode("after", func(_ context.Context, s testState) (testState, error) {
		ran.Add(10)
		return testState{Log: []string{"after"}}, nil
	})
	g.MustAddEdge(graph.START, "pause")
	g.MustAddEdge("pause", "after")
	g.MustAddEdge("after", graph.END)

	cp := graph.NewMemoryCheckpointer[testState]()
	compiled, _ := g.Compile(graph.WithCheckpointer[testState](cp))

	ctx := context.Background()
	_, err := compiled.Invoke(ctx, testState{}, graph.WithThreadID[testState]("hilo"))
	// Should return the interrupt error
	if err == nil {
		t.Error("expected interrupt error")
	}
	interrupt, ok := err.(*graph.Interrupt)
	if !ok {
		t.Fatalf("expected *graph.Interrupt, got %T: %v", err, err)
	}
	if interrupt.Message != "need approval" {
		t.Errorf("message: want %q got %q", "need approval", interrupt.Message)
	}
	// "after" node should not have run
	if ran.Load() != 1 {
		t.Errorf("only pause node should have run, ran=%d", ran.Load())
	}
}

// ---------------------------------------------------------------------------
// Serialise / Deserialise checkpoint
// ---------------------------------------------------------------------------

func TestCheckpoint_Serialise(t *testing.T) {
	cp := graph.Checkpoint[testState]{
		ThreadID: "t1",
		State:    testState{Count: 7, Log: []string{"x"}},
	}
	data, err := graph.SerialiseCheckpoint(cp)
	if err != nil {
		t.Fatalf("SerialiseCheckpoint: %v", err)
	}
	restored, err := graph.DeserialiseCheckpoint[testState](data)
	if err != nil {
		t.Fatalf("DeserialiseCheckpoint: %v", err)
	}
	if restored.State.Count != 7 {
		t.Errorf("Count: want 7, got %d", restored.State.Count)
	}
}

// ---------------------------------------------------------------------------
// WithName
// ---------------------------------------------------------------------------

func TestWithName(t *testing.T) {
	g := graph.NewStateGraph(reducer).WithName("MyGraph")
	g.MustAddNode("n", func(_ context.Context, s testState) (testState, error) { return s, nil })
	g.MustAddEdge(graph.START, "n")
	g.MustAddEdge("n", graph.END)
	compiled, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	// Invoke should work — graph name flows through to callbacks
	_, err = compiled.Invoke(context.Background(), testState{})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
}

// ---------------------------------------------------------------------------
// MustAddEdge panics on error
// ---------------------------------------------------------------------------

func TestMustAddEdge_PanicsOnError(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for adding edge from END")
		}
	}()
	g := graph.NewStateGraph(reducer)
	g.MustAddEdge(graph.END, "anywhere") // should panic
}

// ---------------------------------------------------------------------------
// Conditional edge with missing key and no wildcard → error
// ---------------------------------------------------------------------------

func TestConditionalEdge_MissingKey_NoWildcard(t *testing.T) {
	g := graph.NewStateGraph(reducer)
	g.MustAddNode("n", func(_ context.Context, s testState) (testState, error) {
		return testState{Next: "unknown"}, nil
	})
	g.MustAddEdge(graph.START, "n")
	g.MustAddConditionalEdges("n",
		func(_ context.Context, s testState) string { return s.Next },
		map[string]string{"known": graph.END},
	)

	compiled, _ := g.Compile()
	_, err := compiled.Invoke(context.Background(), testState{})
	if err == nil {
		t.Error("expected error for unknown routing key without wildcard")
	}
}
