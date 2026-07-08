package memory_test

import (
	"context"
	"testing"

	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/memory"
	"github.com/grafaelw/golangchain/schema"
)

var ctx = context.Background()

// ---------------------------------------------------------------------------
// ConversationBufferMemory
// ---------------------------------------------------------------------------

func TestBufferMemory_SaveAndLoad(t *testing.T) {
	m := memory.NewConversationBufferMemory()

	if err := m.SaveContext(ctx, "Hello", "Hi there!"); err != nil {
		t.Fatalf("SaveContext: %v", err)
	}
	if err := m.SaveContext(ctx, "How are you?", "I'm fine."); err != nil {
		t.Fatalf("SaveContext: %v", err)
	}

	vars, err := m.LoadMemoryVariables(ctx)
	if err != nil {
		t.Fatalf("LoadMemoryVariables: %v", err)
	}

	hist, ok := vars["history"]
	if !ok {
		t.Fatal("expected 'history' key in vars")
	}
	msgs := hist.([]schema.Message)
	if len(msgs) != 4 {
		t.Fatalf("want 4 messages, got %d", len(msgs))
	}
	if msgs[0].Role != schema.RoleHuman || msgs[0].Content != "Hello" {
		t.Errorf("msgs[0] mismatch: %+v", msgs[0])
	}
	if msgs[1].Role != schema.RoleAI || msgs[1].Content != "Hi there!" {
		t.Errorf("msgs[1] mismatch: %+v", msgs[1])
	}
}

func TestBufferMemory_Messages(t *testing.T) {
	m := memory.NewConversationBufferMemory()
	m.SaveContext(ctx, "q1", "a1")
	msgs := m.Messages()
	if len(msgs) != 2 {
		t.Fatalf("want 2, got %d", len(msgs))
	}
}

func TestBufferMemory_Clear(t *testing.T) {
	m := memory.NewConversationBufferMemory()
	m.SaveContext(ctx, "q", "a")
	if err := m.Clear(ctx); err != nil {
		t.Fatalf("Clear: %v", err)
	}
	msgs := m.Messages()
	if len(msgs) != 0 {
		t.Errorf("want 0 messages after clear, got %d", len(msgs))
	}
}

func TestBufferMemory_ReturnString(t *testing.T) {
	m := memory.NewConversationBufferMemory()
	m.ReturnMessages = false
	m.SaveContext(ctx, "Hello", "Hi")

	vars, err := m.LoadMemoryVariables(ctx)
	if err != nil {
		t.Fatalf("LoadMemoryVariables: %v", err)
	}
	hist := vars["history"]
	s, ok := hist.(string)
	if !ok {
		t.Fatalf("expected string, got %T", hist)
	}
	if s == "" {
		t.Error("expected non-empty string history")
	}
}

func TestBufferMemory_CustomHistoryKey(t *testing.T) {
	m := memory.NewConversationBufferMemory()
	m.HistoryKey = "chat_history"
	m.SaveContext(ctx, "x", "y")

	vars, _ := m.LoadMemoryVariables(ctx)
	if _, ok := vars["chat_history"]; !ok {
		t.Error("expected 'chat_history' key")
	}
}

// ---------------------------------------------------------------------------
// ConversationWindowMemory
// ---------------------------------------------------------------------------

func TestWindowMemory_Window(t *testing.T) {
	m := memory.NewConversationWindowMemory(2)

	// Add 3 turns — only last 2 should be retained
	for i, pair := range [][2]string{
		{"turn1-q", "turn1-a"},
		{"turn2-q", "turn2-a"},
		{"turn3-q", "turn3-a"},
	} {
		_ = i
		m.SaveContext(ctx, pair[0], pair[1])
	}

	msgs := m.Messages()
	// 2 turns * 2 messages = 4
	if len(msgs) != 4 {
		t.Fatalf("want 4 messages (last 2 turns), got %d", len(msgs))
	}
	if msgs[0].Content != "turn2-q" {
		t.Errorf("first retained message should be turn2-q, got %q", msgs[0].Content)
	}
}

func TestWindowMemory_BelowWindow(t *testing.T) {
	m := memory.NewConversationWindowMemory(5)
	m.SaveContext(ctx, "q", "a")
	msgs := m.Messages()
	if len(msgs) != 2 {
		t.Fatalf("want 2, got %d", len(msgs))
	}
}

func TestWindowMemory_LoadReturnsMessages(t *testing.T) {
	m := memory.NewConversationWindowMemory(3)
	m.SaveContext(ctx, "hello", "world")

	vars, err := m.LoadMemoryVariables(ctx)
	if err != nil {
		t.Fatalf("LoadMemoryVariables: %v", err)
	}
	hist := vars["history"].([]schema.Message)
	if len(hist) != 2 {
		t.Errorf("want 2, got %d", len(hist))
	}
}

func TestWindowMemory_ReturnString(t *testing.T) {
	m := memory.NewConversationWindowMemory(3)
	m.ReturnMessages = false
	m.SaveContext(ctx, "q", "a")

	vars, _ := m.LoadMemoryVariables(ctx)
	_, ok := vars["history"].(string)
	if !ok {
		t.Error("expected string history when ReturnMessages=false")
	}
}

func TestWindowMemory_Clear(t *testing.T) {
	m := memory.NewConversationWindowMemory(3)
	m.SaveContext(ctx, "q", "a")
	m.Clear(ctx)
	if len(m.Messages()) != 0 {
		t.Error("expected empty messages after clear")
	}
}

// ---------------------------------------------------------------------------
// ConversationSummaryMemory
// ---------------------------------------------------------------------------

// mockSummaryLLM is a stub LLM used to test ConversationSummaryMemory
// without making real API calls. Implements llm.LLM.
type mockSummaryLLM struct {
	summary string
}

func (m *mockSummaryLLM) Generate(_ context.Context, msgs []schema.Message, _ ...llm.Option) (*schema.Generation, error) {
	return &schema.Generation{Text: m.summary, Message: schema.NewAIMessage(m.summary)}, nil
}
func (m *mockSummaryLLM) Stream(_ context.Context, _ []schema.Message, _ ...llm.Option) (<-chan schema.StreamChunk, error) {
	ch := make(chan schema.StreamChunk, 1)
	ch <- schema.StreamChunk{Text: m.summary, Done: true}
	close(ch)
	return ch, nil
}
func (m *mockSummaryLLM) ModelName() string { return "mock" }

func TestSummaryMemory_Basic(t *testing.T) {
	llm := &mockSummaryLLM{summary: "User asked about Go, AI explained."}
	m := memory.NewConversationSummaryMemory(llm)
	m.MaxRecent = 2 // compress after 2 turns

	// Save 3 turns → should trigger compression after 3rd
	m.SaveContext(ctx, "What is Go?", "A compiled language.")
	m.SaveContext(ctx, "What is a goroutine?", "A lightweight thread.")
	m.SaveContext(ctx, "Thanks", "You're welcome.")

	// After compression the recent messages are cleared and summary is set
	msgs := m.Messages()
	// Should have a system message with the summary
	if len(msgs) == 0 {
		t.Fatal("expected at least one message")
	}
	// First message should be system summary
	if msgs[0].Role != schema.RoleSystem {
		t.Errorf("first message should be system summary, got %q", msgs[0].Role)
	}
}

func TestSummaryMemory_Clear(t *testing.T) {
	llm := &mockSummaryLLM{summary: "summary"}
	m := memory.NewConversationSummaryMemory(llm)
	m.SaveContext(ctx, "q", "a")
	m.Clear(ctx)
	msgs := m.Messages()
	if len(msgs) != 0 {
		t.Errorf("expected empty after clear, got %d", len(msgs))
	}
}

func TestSummaryMemory_LoadMemoryVariables(t *testing.T) {
	llm := &mockSummaryLLM{summary: "x"}
	m := memory.NewConversationSummaryMemory(llm)
	m.SaveContext(ctx, "q", "a")
	vars, err := m.LoadMemoryVariables(ctx)
	if err != nil {
		t.Fatalf("LoadMemoryVariables: %v", err)
	}
	if _, ok := vars["history"]; !ok {
		t.Error("expected 'history' key")
	}
}
