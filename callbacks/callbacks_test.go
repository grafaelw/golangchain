package callbacks_test

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/grafaelw/golangchain/callbacks"
	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// recordHandler — captures every event for assertions
// ---------------------------------------------------------------------------

type recordHandler struct {
	callbacks.NoOpHandler
	mu     sync.Mutex
	events []string
}

func (r *recordHandler) record(event string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, event)
}

func (r *recordHandler) OnLLMStart(_ context.Context, model string, _ []schema.Message) {
	r.record("llm_start:" + model)
}
func (r *recordHandler) OnLLMEnd(_ context.Context, model string, _ *schema.Generation) {
	r.record("llm_end:" + model)
}
func (r *recordHandler) OnChainStart(_ context.Context, name string, _ map[string]any) {
	r.record("chain_start:" + name)
}
func (r *recordHandler) OnChainEnd(_ context.Context, name string, _ map[string]any) {
	r.record("chain_end:" + name)
}
func (r *recordHandler) OnToolStart(_ context.Context, name, _ string) {
	r.record("tool_start:" + name)
}
func (r *recordHandler) OnToolEnd(_ context.Context, name, _ string) {
	r.record("tool_end:" + name)
}
func (r *recordHandler) OnAgentAction(_ context.Context, a schema.AgentAction) {
	r.record("agent_action:" + a.Tool)
}
func (r *recordHandler) OnAgentFinish(_ context.Context, f schema.AgentFinish) {
	r.record("agent_finish:" + f.Output)
}
func (r *recordHandler) OnGraphNodeStart(_ context.Context, graph, node string) {
	r.record("node_start:" + graph + "." + node)
}
func (r *recordHandler) OnGraphNodeEnd(_ context.Context, graph, node string) {
	r.record("node_end:" + graph + "." + node)
}
func (r *recordHandler) OnError(_ context.Context, source string, err error) {
	r.record("error:" + source)
}

func (r *recordHandler) has(event string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, e := range r.events {
		if e == event {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// CallbackManager tests
// ---------------------------------------------------------------------------

func TestCallbackManager_FanOut(t *testing.T) {
	h1 := &recordHandler{}
	h2 := &recordHandler{}
	cm := callbacks.NewCallbackManager(h1, h2)

	ctx := context.Background()
	gen := &schema.Generation{Text: "hello"}
	cm.OnLLMStart(ctx, "gpt-4o", nil)
	cm.OnLLMEnd(ctx, "gpt-4o", gen)

	for _, h := range []*recordHandler{h1, h2} {
		if !h.has("llm_start:gpt-4o") {
			t.Errorf("handler missing llm_start event")
		}
		if !h.has("llm_end:gpt-4o") {
			t.Errorf("handler missing llm_end event")
		}
	}
}

func TestCallbackManager_Add(t *testing.T) {
	h := &recordHandler{}
	cm := callbacks.NewCallbackManager()
	cm.Add(h)

	cm.OnChainStart(context.Background(), "MyChain", nil)
	if !h.has("chain_start:MyChain") {
		t.Error("dynamically added handler missed event")
	}
}

func TestCallbackManager_AllEvents(t *testing.T) {
	h := &recordHandler{}
	cm := callbacks.NewCallbackManager(h)
	ctx := context.Background()

	cm.OnLLMStart(ctx, "m", nil)
	cm.OnLLMEnd(ctx, "m", &schema.Generation{})
	cm.OnLLMStream(ctx, "m", schema.StreamChunk{})
	cm.OnChainStart(ctx, "c", nil)
	cm.OnChainEnd(ctx, "c", nil)
	cm.OnToolStart(ctx, "t", "")
	cm.OnToolEnd(ctx, "t", "")
	cm.OnAgentAction(ctx, schema.AgentAction{Tool: "calc"})
	cm.OnAgentFinish(ctx, schema.AgentFinish{Output: "done"})
	cm.OnGraphNodeStart(ctx, "g", "n")
	cm.OnGraphNodeEnd(ctx, "g", "n")
	cm.OnError(ctx, "src", fmt.Errorf("boom"))

	expected := []string{
		"llm_start:m", "llm_end:m",
		"chain_start:c", "chain_end:c",
		"tool_start:t", "tool_end:t",
		"agent_action:calc", "agent_finish:done",
		"node_start:g.n", "node_end:g.n",
		"error:src",
	}
	for _, e := range expected {
		if !h.has(e) {
			t.Errorf("missing event %q", e)
		}
	}
}

func TestCallbackManager_Concurrent(t *testing.T) {
	h := &recordHandler{}
	cm := callbacks.NewCallbackManager(h)
	ctx := context.Background()

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			cm.OnChainStart(ctx, "concurrent", nil)
		}()
	}
	wg.Wait()

	h.mu.Lock()
	count := len(h.events)
	h.mu.Unlock()
	if count != 50 {
		t.Errorf("want 50 events, got %d", count)
	}
}

// ---------------------------------------------------------------------------
// NoOpHandler — all methods should be callable without panic
// ---------------------------------------------------------------------------

func TestNoOpHandler_AllMethods(t *testing.T) {
	var h callbacks.NoOpHandler
	ctx := context.Background()
	// Just ensure no panics
	h.OnLLMStart(ctx, "m", nil)
	h.OnLLMEnd(ctx, "m", nil)
	h.OnLLMStream(ctx, "m", schema.StreamChunk{})
	h.OnChainStart(ctx, "c", nil)
	h.OnChainEnd(ctx, "c", nil)
	h.OnToolStart(ctx, "t", "")
	h.OnToolEnd(ctx, "t", "")
	h.OnAgentAction(ctx, schema.AgentAction{})
	h.OnAgentFinish(ctx, schema.AgentFinish{})
	h.OnGraphNodeStart(ctx, "g", "n")
	h.OnGraphNodeEnd(ctx, "g", "n")
	h.OnGraphCheckpoint(ctx, "g", "tid")
	h.OnError(ctx, "src", nil)
}

// ---------------------------------------------------------------------------
// LoggingHandler
// ---------------------------------------------------------------------------

func TestLoggingHandler_Logs(t *testing.T) {
	var logged []string
	lh := callbacks.NewLoggingHandler(func(format string, args ...any) {
		logged = append(logged, fmt.Sprintf(format, args...))
	})

	ctx := context.Background()
	lh.OnLLMStart(ctx, "gpt-4o", []schema.Message{schema.NewHumanMessage("hi")})
	lh.OnLLMEnd(ctx, "gpt-4o", &schema.Generation{StopReason: "stop"})
	lh.OnChainStart(ctx, "MyChain", nil)
	lh.OnChainEnd(ctx, "MyChain", nil)
	lh.OnToolStart(ctx, "calc", "2+2")
	lh.OnToolEnd(ctx, "calc", "4")
	lh.OnAgentAction(ctx, schema.AgentAction{Tool: "search", ToolInput: "go lang"})
	lh.OnAgentFinish(ctx, schema.AgentFinish{Output: "answer"})
	lh.OnGraphNodeStart(ctx, "graph", "agent")
	lh.OnGraphNodeEnd(ctx, "graph", "agent")
	lh.OnError(ctx, "chain", fmt.Errorf("fail"))

	if len(logged) != 11 {
		t.Errorf("want 11 log lines, got %d: %v", len(logged), logged)
	}
}
