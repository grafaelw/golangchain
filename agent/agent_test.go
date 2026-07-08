package agent_test

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/grafaelw/golangchain/agent"
	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/schema"
	"github.com/grafaelw/golangchain/tools"
)

// ---------------------------------------------------------------------------
// mockLLM for agent tests
// ---------------------------------------------------------------------------

type mockLLM struct {
	responses []schema.Generation // popped FIFO
	idx       int
}

func newMockLLM(responses ...schema.Generation) *mockLLM {
	return &mockLLM{responses: responses}
}

func (m *mockLLM) Generate(_ context.Context, _ []schema.Message, _ ...llm.Option) (*schema.Generation, error) {
	if m.idx >= len(m.responses) {
		return &schema.Generation{Text: "default answer", Message: schema.NewAIMessage("default answer")}, nil
	}
	r := m.responses[m.idx]
	m.idx++
	return &r, nil
}

func (m *mockLLM) Stream(_ context.Context, _ []schema.Message, _ ...llm.Option) (<-chan schema.StreamChunk, error) {
	gen, _ := m.Generate(context.Background(), nil)
	ch := make(chan schema.StreamChunk, 2)
	ch <- schema.StreamChunk{Text: gen.Text}
	ch <- schema.StreamChunk{Done: true}
	close(ch)
	return ch, nil
}

func (m *mockLLM) ModelName() string { return "mock" }

// ---------------------------------------------------------------------------
// mockTool
// ---------------------------------------------------------------------------

type mockTool struct {
	name   string
	output string
	calls  int
}

func (t *mockTool) Name() string            { return t.name }
func (t *mockTool) Description() string     { return "mock tool: " + t.name }
func (t *mockTool) Schema() json.RawMessage { return json.RawMessage(`{"type":"object"}`) }
func (t *mockTool) Run(_ context.Context, _ string) (string, error) {
	t.calls++
	return t.output, nil
}

// verify mockTool implements tools.Tool at compile time
var _ tools.Tool = (*mockTool)(nil)

// ---------------------------------------------------------------------------
// ReAct output parser (via ReActAgent.Plan)
// ---------------------------------------------------------------------------

func TestReActAgent_FinalAnswer(t *testing.T) {
	response := "Thought: I know this.\nFinal Answer: Paris"
	mock := newMockLLM(schema.Generation{
		Text:    response,
		Message: schema.NewAIMessage(response),
	})

	a := agent.NewReActAgent(mock, nil)
	actions, finish, err := a.Plan(context.Background(), []schema.Message{
		schema.NewHumanMessage("Capital of France?"),
	}, nil)

	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	if finish == nil {
		t.Fatal("expected AgentFinish, got nil")
	}
	if finish.Output != "Paris" {
		t.Errorf("want Paris, got %q", finish.Output)
	}
	if len(actions) != 0 {
		t.Errorf("expected no actions when finish is returned")
	}
}

func TestReActAgent_ToolAction(t *testing.T) {
	response := "Thought: Need to calculate.\nAction: calculator\nAction Input: 2 + 2"
	mock := newMockLLM(schema.Generation{
		Text:    response,
		Message: schema.NewAIMessage(response),
	})

	a := agent.NewReActAgent(mock, nil)
	actions, finish, err := a.Plan(context.Background(), []schema.Message{
		schema.NewHumanMessage("What is 2+2?"),
	}, nil)

	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	if finish != nil {
		t.Fatal("expected nil AgentFinish for tool action")
	}
	if len(actions) != 1 {
		t.Fatalf("want 1 action, got %d", len(actions))
	}
	if actions[0].Tool != "calculator" {
		t.Errorf("tool: want calculator, got %q", actions[0].Tool)
	}
	if actions[0].ToolInput != "2 + 2" {
		t.Errorf("tool input: want '2 + 2', got %q", actions[0].ToolInput)
	}
}

func TestReActAgent_UnparsedTextTreatedAsFinalAnswer(t *testing.T) {
	response := "Just some text without a ReAct format"
	mock := newMockLLM(schema.Generation{
		Text:    response,
		Message: schema.NewAIMessage(response),
	})

	a := agent.NewReActAgent(mock, nil)
	_, finish, err := a.Plan(context.Background(), []schema.Message{
		schema.NewHumanMessage("hello"),
	}, nil)

	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	if finish == nil {
		t.Fatal("unparsed text should become final answer")
	}
}

func TestReActAgent_Name(t *testing.T) {
	a := agent.NewReActAgent(nil, nil)
	if a.Name() != "ReActAgent" {
		t.Errorf("Name: want ReActAgent, got %q", a.Name())
	}
}

// ---------------------------------------------------------------------------
// ToolCallingAgent
// ---------------------------------------------------------------------------

func TestToolCallingAgent_FinalAnswer(t *testing.T) {
	// LLM returns plain text (no tool calls) → final answer
	mock := newMockLLM(schema.Generation{
		Text:    "The answer is 42.",
		Message: schema.NewAIMessage("The answer is 42."),
	})

	calc := &tools.Calculator{}
	a := agent.NewToolCallingAgent(mock, []tools.Tool{calc}, "Be helpful.")

	actions, finish, err := a.Plan(context.Background(), []schema.Message{
		schema.NewHumanMessage("What is 6 * 7?"),
	}, nil)

	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	if finish == nil {
		t.Fatal("expected AgentFinish")
	}
	if finish.Output != "The answer is 42." {
		t.Errorf("output: want 'The answer is 42.', got %q", finish.Output)
	}
	if len(actions) != 0 {
		t.Error("expected no actions")
	}
}

func TestToolCallingAgent_ToolCall(t *testing.T) {
	// LLM returns a tool call
	mock := newMockLLM(schema.Generation{
		Text: "",
		Message: schema.Message{
			Role: schema.RoleAI,
			ToolCalls: []schema.ToolCall{
				{ID: "tc1", Name: "calculator", Arguments: []byte(`{"expression":"2+2"}`)},
			},
		},
	})

	calc := &tools.Calculator{}
	a := agent.NewToolCallingAgent(mock, []tools.Tool{calc}, "")

	actions, finish, err := a.Plan(context.Background(), []schema.Message{
		schema.NewHumanMessage("2 + 2?"),
	}, nil)

	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	if finish != nil {
		t.Fatal("expected nil finish for tool call")
	}
	if len(actions) != 1 {
		t.Fatalf("want 1 action, got %d", len(actions))
	}
	if actions[0].Tool != "calculator" {
		t.Errorf("tool: want calculator, got %q", actions[0].Tool)
	}
}

func TestToolCallingAgent_Name(t *testing.T) {
	a := agent.NewToolCallingAgent(nil, nil, "")
	if a.Name() != "ToolCallingAgent" {
		t.Errorf("Name: want ToolCallingAgent, got %q", a.Name())
	}
}

// ---------------------------------------------------------------------------
// AgentExecutor — full loop
// ---------------------------------------------------------------------------

func TestAgentExecutor_Run_FinalAnswer(t *testing.T) {
	// Agent returns final answer immediately (no tools needed)
	response := "Final Answer: Amsterdam"
	mock := newMockLLM(schema.Generation{
		Text:    response,
		Message: schema.NewAIMessage(response),
	})

	a := agent.NewReActAgent(mock, nil)
	executor := agent.NewAgentExecutor(a, nil)

	answer, err := executor.Run(context.Background(), "Capital of the Netherlands?")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if answer != "Amsterdam" {
		t.Errorf("want Amsterdam, got %q", answer)
	}
}

func TestAgentExecutor_Run_WithTool(t *testing.T) {
	// First call returns tool action, second call returns final answer
	toolResponse := "Thought: Let me calculate.\nAction: calculator\nAction Input: 6 * 7"
	finalResponse := "Final Answer: 42"

	mock := newMockLLM(
		schema.Generation{Text: toolResponse, Message: schema.NewAIMessage(toolResponse)},
		schema.Generation{Text: finalResponse, Message: schema.NewAIMessage(finalResponse)},
	)

	calc := tools.Calculator{}
	a := agent.NewReActAgent(mock, []tools.Tool{calc})
	executor := agent.NewAgentExecutor(a, []tools.Tool{calc})

	answer, err := executor.Run(context.Background(), "What is 6 * 7?")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if answer != "42" {
		t.Errorf("want 42, got %q", answer)
	}
}

func TestAgentExecutor_MaxIter(t *testing.T) {
	// Agent always returns tool action, never finishes → should hit MaxIter
	alwaysTool := "Thought: hmm\nAction: calculator\nAction Input: 1+1"
	mock := &mockLLM{}
	for i := 0; i < 20; i++ {
		mock.responses = append(mock.responses, schema.Generation{
			Text:    alwaysTool,
			Message: schema.NewAIMessage(alwaysTool),
		})
	}
	calc := tools.Calculator{}
	a := agent.NewReActAgent(mock, []tools.Tool{calc})
	executor := agent.NewAgentExecutor(a, []tools.Tool{calc}, agent.WithMaxIter(3))

	_, err := executor.Run(context.Background(), "keep going")
	if err == nil {
		t.Error("expected max-iter error")
	}
}

func TestAgentExecutor_UnknownTool(t *testing.T) {
	response := "Thought: Use a ghost tool.\nAction: ghost_tool\nAction Input: boo"
	finalResp := "Final Answer: done"
	mock := newMockLLM(
		schema.Generation{Text: response, Message: schema.NewAIMessage(response)},
		schema.Generation{Text: finalResp, Message: schema.NewAIMessage(finalResp)},
	)
	a := agent.NewReActAgent(mock, nil)
	executor := agent.NewAgentExecutor(a, nil)

	// Should not crash; tool error becomes observation
	_, err := executor.Run(context.Background(), "test")
	// Either runs to answer or errors at max iter; the key is it doesn't panic
	_ = err
}

// ---------------------------------------------------------------------------
// AgentEvent streaming
// ---------------------------------------------------------------------------

func TestAgentExecutor_Stream_Events(t *testing.T) {
	finalResponse := "Final Answer: Done"
	mock := newMockLLM(schema.Generation{
		Text:    finalResponse,
		Message: schema.NewAIMessage(finalResponse),
	})
	a := agent.NewReActAgent(mock, nil)
	executor := agent.NewAgentExecutor(a, nil)

	var events []agent.EventType
	for ev := range executor.Stream(context.Background(), "hello") {
		events = append(events, ev.Type)
	}

	hasFinal := false
	for _, e := range events {
		if e == agent.EventFinalAnswer {
			hasFinal = true
		}
	}
	if !hasFinal {
		t.Errorf("expected EventFinalAnswer in stream, got: %v", events)
	}
}

func TestAgentExecutor_Stream_ToolEvents(t *testing.T) {
	toolResp := "Thought: calc it.\nAction: calculator\nAction Input: 3+3"
	finalResp := "Final Answer: 6"
	mock := newMockLLM(
		schema.Generation{Text: toolResp, Message: schema.NewAIMessage(toolResp)},
		schema.Generation{Text: finalResp, Message: schema.NewAIMessage(finalResp)},
	)
	calc := tools.Calculator{}
	a := agent.NewReActAgent(mock, []tools.Tool{calc})
	executor := agent.NewAgentExecutor(a, []tools.Tool{calc})

	eventTypes := map[agent.EventType]int{}
	for ev := range executor.Stream(context.Background(), "3+3?") {
		eventTypes[ev.Type]++
	}

	for _, et := range []agent.EventType{agent.EventToolCall, agent.EventToolResult, agent.EventFinalAnswer} {
		if eventTypes[et] == 0 {
			t.Errorf("missing event type %q", et)
		}
	}
}

// ---------------------------------------------------------------------------
// ExecutorOption helpers
// ---------------------------------------------------------------------------

func TestExecutorOptions(t *testing.T) {
	mock := newMockLLM(schema.Generation{
		Text:    "Final Answer: ok",
		Message: schema.NewAIMessage("Final Answer: ok"),
	})
	a := agent.NewReActAgent(mock, nil)
	// Just verifies options don't cause panics
	executor := agent.NewAgentExecutor(a, nil,
		agent.WithMaxIter(5),
		agent.WithVerbose(true),
	)
	_, err := executor.Run(context.Background(), "test")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Error propagation
// ---------------------------------------------------------------------------

func TestAgentExecutor_LLMError(t *testing.T) {
	// LLM always errors
	errLLM := &errLLMStub{}
	a := agent.NewReActAgent(errLLM, nil)
	executor := agent.NewAgentExecutor(a, nil)
	_, err := executor.Run(context.Background(), "hello")
	if err == nil {
		t.Error("expected error when LLM always fails")
	}
}

type errLLMStub struct{}

func (e *errLLMStub) Generate(_ context.Context, _ []schema.Message, _ ...llm.Option) (*schema.Generation, error) {
	return nil, fmt.Errorf("LLM unavailable")
}
func (e *errLLMStub) Stream(_ context.Context, _ []schema.Message, _ ...llm.Option) (<-chan schema.StreamChunk, error) {
	return nil, fmt.Errorf("LLM unavailable")
}
func (e *errLLMStub) ModelName() string { return "error-model" }
