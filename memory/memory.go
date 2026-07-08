// Package memory provides conversation memory implementations for golangchain.
// Memories store and retrieve conversation history so LLMChains and agents
// can maintain context across multiple turns.
package memory

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// Memory interface
// ---------------------------------------------------------------------------

// Memory is the interface all memory types implement.
// LoadMemoryVariables returns variables to inject into the prompt (typically
// the conversation history under a well-known key like "history").
// SaveContext persists an input/output turn after the LLM responds.
type Memory interface {
	// LoadMemoryVariables returns variables for prompt injection.
	// The keys returned depend on the implementation (e.g. "history").
	LoadMemoryVariables(ctx context.Context) (map[string]any, error)

	// SaveContext records one conversation turn.
	SaveContext(ctx context.Context, humanInput, aiOutput string) error

	// Messages returns the raw conversation history as schema.Message slice.
	Messages() []schema.Message

	// Clear wipes all stored history.
	Clear(ctx context.Context) error
}

// ---------------------------------------------------------------------------
// ConversationBufferMemory — keeps the full history
// ---------------------------------------------------------------------------

// ConversationBufferMemory stores the entire conversation as a flat list of
// messages. It is the simplest and most common memory type.
//
//	mem := memory.NewConversationBufferMemory()
//	mem.SaveContext(ctx, "Hi there", "Hello! How can I help?")
//	vars, _ := mem.LoadMemoryVariables(ctx)
//	// vars["history"] is []schema.Message
type ConversationBufferMemory struct {
	mu             sync.RWMutex
	messages       []schema.Message
	HistoryKey     string // key used in LoadMemoryVariables (default: "history")
	HumanPrefix    string // default: "Human"
	AIPrefix       string // default: "AI"
	ReturnMessages bool   // if true, return []schema.Message; if false, return string
}

// NewConversationBufferMemory creates a ConversationBufferMemory with defaults.
func NewConversationBufferMemory() *ConversationBufferMemory {
	return &ConversationBufferMemory{
		HistoryKey:     "history",
		HumanPrefix:    "Human",
		AIPrefix:       "AI",
		ReturnMessages: true,
	}
}

func (m *ConversationBufferMemory) Messages() []schema.Message {
	m.mu.RLock()
	defer m.mu.RUnlock()
	cp := make([]schema.Message, len(m.messages))
	copy(cp, m.messages)
	return cp
}

func (m *ConversationBufferMemory) LoadMemoryVariables(_ context.Context) (map[string]any, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.ReturnMessages {
		cp := make([]schema.Message, len(m.messages))
		copy(cp, m.messages)
		return map[string]any{m.HistoryKey: cp}, nil
	}

	// Return as a formatted string for simple PromptTemplates
	var sb strings.Builder
	for _, msg := range m.messages {
		switch msg.Role {
		case schema.RoleHuman:
			sb.WriteString(m.HumanPrefix + ": " + msg.Content + "\n")
		case schema.RoleAI:
			sb.WriteString(m.AIPrefix + ": " + msg.Content + "\n")
		}
	}
	return map[string]any{m.HistoryKey: strings.TrimRight(sb.String(), "\n")}, nil
}

func (m *ConversationBufferMemory) SaveContext(_ context.Context, humanInput, aiOutput string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.messages = append(m.messages,
		schema.NewHumanMessage(humanInput),
		schema.NewAIMessage(aiOutput),
	)
	return nil
}

func (m *ConversationBufferMemory) Clear(_ context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.messages = nil
	return nil
}

// ---------------------------------------------------------------------------
// ConversationWindowMemory — keeps the last k turns
// ---------------------------------------------------------------------------

// ConversationWindowMemory keeps only the most recent k human+AI turns.
// Older messages are silently dropped.
//
//	mem := memory.NewConversationWindowMemory(5) // keep last 5 turns
type ConversationWindowMemory struct {
	mu             sync.RWMutex
	messages       []schema.Message
	K              int // number of turns to retain (1 turn = 1 human + 1 AI message)
	HistoryKey     string
	ReturnMessages bool
}

// NewConversationWindowMemory creates a window memory that keeps the last k turns.
func NewConversationWindowMemory(k int) *ConversationWindowMemory {
	return &ConversationWindowMemory{
		K:              k,
		HistoryKey:     "history",
		ReturnMessages: true,
	}
}

func (m *ConversationWindowMemory) Messages() []schema.Message {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.window()
}

func (m *ConversationWindowMemory) window() []schema.Message {
	max := m.K * 2 // each turn = 2 messages
	if len(m.messages) <= max {
		cp := make([]schema.Message, len(m.messages))
		copy(cp, m.messages)
		return cp
	}
	cp := make([]schema.Message, max)
	copy(cp, m.messages[len(m.messages)-max:])
	return cp
}

func (m *ConversationWindowMemory) LoadMemoryVariables(_ context.Context) (map[string]any, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	msgs := m.window()
	if m.ReturnMessages {
		return map[string]any{m.HistoryKey: msgs}, nil
	}
	var sb strings.Builder
	for _, msg := range msgs {
		sb.WriteString(string(msg.Role) + ": " + msg.Content + "\n")
	}
	return map[string]any{m.HistoryKey: strings.TrimRight(sb.String(), "\n")}, nil
}

func (m *ConversationWindowMemory) SaveContext(_ context.Context, humanInput, aiOutput string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.messages = append(m.messages,
		schema.NewHumanMessage(humanInput),
		schema.NewAIMessage(aiOutput),
	)
	return nil
}

func (m *ConversationWindowMemory) Clear(_ context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.messages = nil
	return nil
}

// ---------------------------------------------------------------------------
// ConversationSummaryMemory — summarises history via an LLM
// ---------------------------------------------------------------------------

// ConversationSummaryMemory uses an LLM to incrementally summarise the
// conversation as it grows. Older content is compressed; only the summary
// and the most recent raw turns are kept.
//
//	mem := memory.NewConversationSummaryMemory(myLLM)
type ConversationSummaryMemory struct {
	mu         sync.Mutex
	llm        llm.LLM
	summary    string
	recent     []schema.Message
	MaxRecent  int // raw turns to keep before compressing (default 4)
	HistoryKey string
}

// NewConversationSummaryMemory creates a summary memory that uses model to
// compress conversation history.
func NewConversationSummaryMemory(model llm.LLM) *ConversationSummaryMemory {
	return &ConversationSummaryMemory{
		llm:        model,
		MaxRecent:  4,
		HistoryKey: "history",
	}
}

func (m *ConversationSummaryMemory) Messages() []schema.Message {
	m.mu.Lock()
	defer m.mu.Unlock()
	var msgs []schema.Message
	if m.summary != "" {
		msgs = append(msgs, schema.NewSystemMessage("Conversation summary so far:\n"+m.summary))
	}
	msgs = append(msgs, m.recent...)
	return msgs
}

func (m *ConversationSummaryMemory) LoadMemoryVariables(_ context.Context) (map[string]any, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	msgs := m.buildMessages()
	return map[string]any{m.HistoryKey: msgs}, nil
}

func (m *ConversationSummaryMemory) buildMessages() []schema.Message {
	var msgs []schema.Message
	if m.summary != "" {
		msgs = append(msgs, schema.NewSystemMessage("Previous conversation summary:\n"+m.summary))
	}
	msgs = append(msgs, m.recent...)
	return msgs
}

func (m *ConversationSummaryMemory) SaveContext(ctx context.Context, humanInput, aiOutput string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.recent = append(m.recent,
		schema.NewHumanMessage(humanInput),
		schema.NewAIMessage(aiOutput),
	)
	// If we have exceeded MaxRecent turns, summarise
	if len(m.recent) > m.MaxRecent*2 {
		if err := m.compress(ctx); err != nil {
			return fmt.Errorf("memory: summary compress: %w", err)
		}
	}
	return nil
}

func (m *ConversationSummaryMemory) compress(ctx context.Context) error {
	var sb strings.Builder
	if m.summary != "" {
		sb.WriteString("Existing summary:\n")
		sb.WriteString(m.summary)
		sb.WriteString("\n\nNew conversation to add:\n")
	}
	for _, msg := range m.recent {
		sb.WriteString(string(msg.Role) + ": " + msg.Content + "\n")
	}
	sb.WriteString("\nProvide a concise updated summary of the full conversation above.")

	gen, err := m.llm.Generate(ctx, []schema.Message{
		schema.NewHumanMessage(sb.String()),
	})
	if err != nil {
		return err
	}
	m.summary = strings.TrimSpace(gen.Text)
	m.recent = nil
	return nil
}

func (m *ConversationSummaryMemory) Clear(_ context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.summary = ""
	m.recent = nil
	return nil
}
