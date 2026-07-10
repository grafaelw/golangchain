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
	LoadMemoryVariables(ctx context.Context) (map[string]any, error)
	SaveContext(ctx context.Context, humanInput, aiOutput string) error
	Messages() []schema.Message
	Clear(ctx context.Context) error
}

// MessagesMemory extends Memory with the ability to save multi-message turns
// that include tool calls, tool results, and intermediate agent steps.
// Implementations that do not support this fall back to SaveContext.
type MessagesMemory interface {
	Memory
	// SaveMessages records a turn containing arbitrary messages (human,
	// AI with tool calls, tool results). This preserves full agent
	// interaction history for accurate conversation replay.
	SaveMessages(ctx context.Context, msgs []schema.Message) error
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

func (m *ConversationBufferMemory) SaveMessages(_ context.Context, msgs []schema.Message) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.messages = append(m.messages, msgs...)
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

func (m *ConversationWindowMemory) SaveMessages(_ context.Context, msgs []schema.Message) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.messages = append(m.messages, msgs...)
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

func (m *ConversationSummaryMemory) SaveMessages(ctx context.Context, msgs []schema.Message) error {
	m.mu.Lock()
	m.recent = append(m.recent, msgs...)
	exceeded := len(m.recent) > m.MaxRecent*2
	m.mu.Unlock()

	if exceeded {
		return m.compress(ctx)
	}
	return nil
}

// ---------------------------------------------------------------------------
// TokenBufferMemory — keeps a window bounded by token count
// ---------------------------------------------------------------------------

// TokenBufferMemory keeps the conversation trimmed to a maximum token count.
// Unlike ConversationWindowMemory (which counts messages), this trims by
// estimated token count so it respects the model's context window.
//
//	// Keep at most 2000 tokens of history
//	mem := memory.NewTokenBufferMemory(2000)
type TokenBufferMemory struct {
	mu             sync.RWMutex
	messages       []schema.Message
	MaxTokens      int
	HistoryKey     string
	ReturnMessages bool
}

// NewTokenBufferMemory creates a token-buffer memory with the given limit.
func NewTokenBufferMemory(maxTokens int) *TokenBufferMemory {
	return &TokenBufferMemory{
		MaxTokens:      maxTokens,
		HistoryKey:     "history",
		ReturnMessages: true,
	}
}

func (m *TokenBufferMemory) Messages() []schema.Message {
	m.mu.RLock()
	defer m.mu.RUnlock()
	cp := make([]schema.Message, len(m.messages))
	copy(cp, m.messages)
	return cp
}

func (m *TokenBufferMemory) LoadMemoryVariables(_ context.Context) (map[string]any, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	trimmed := m.trim()
	if m.ReturnMessages {
		return map[string]any{m.HistoryKey: trimmed}, nil
	}
	var sb strings.Builder
	for _, msg := range trimmed {
		sb.WriteString(string(msg.Role) + ": " + msg.Content + "\n")
	}
	return map[string]any{m.HistoryKey: strings.TrimRight(sb.String(), "\n")}, nil
}

func (m *TokenBufferMemory) SaveContext(_ context.Context, humanInput, aiOutput string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.messages = append(m.messages,
		schema.NewHumanMessage(humanInput),
		schema.NewAIMessage(aiOutput),
	)
	return nil
}

func (m *TokenBufferMemory) Clear(_ context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.messages = nil
	return nil
}

func (m *TokenBufferMemory) SaveMessages(_ context.Context, msgs []schema.Message) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.messages = append(m.messages, msgs...)
	return nil
}

// trim returns the messages trimmed to fit MaxTokens, dropping from the front.
func (m *TokenBufferMemory) trim() []schema.Message {
	if m.MaxTokens <= 0 {
		cp := make([]schema.Message, len(m.messages))
		copy(cp, m.messages)
		return cp
	}
	total := 0
	for _, msg := range m.messages {
		total += estimateTokens(msg.Content)
	}
	if total <= m.MaxTokens {
		cp := make([]schema.Message, len(m.messages))
		copy(cp, m.messages)
		return cp
	}
	// Drop from the front until we fit.
	start := 0
	for start < len(m.messages) {
		start++
		if start >= len(m.messages) {
			break
		}
		remaining := 0
		for _, m := range m.messages[start:] {
			remaining += estimateTokens(m.Content)
		}
		if remaining <= m.MaxTokens {
			break
		}
	}
	cp := make([]schema.Message, len(m.messages)-start)
	copy(cp, m.messages[start:])
	return cp
}

// estimateTokens is a fast approximation: ~4 characters per token.
func estimateTokens(s string) int {
	return len(s) / 4
}

// ---------------------------------------------------------------------------
// ConversationEntityMemory — entity-aware conversation memory
// ---------------------------------------------------------------------------

// ConversationEntityMemory uses an LLM to extract named entities from each
// turn and maintain summaries per entity. When loading, it injects relevant
// entity context alongside the recent conversation.
//
//	mem := memory.NewConversationEntityMemory(myLLM)
type ConversationEntityMemory struct {
	mu         sync.Mutex
	llm        llm.LLM
	messages   []schema.Message
	entities   map[string]string // entity name → summary
	HistoryKey string
	MaxRecent  int // raw turns to keep (default 4)
}

// NewConversationEntityMemory creates an entity memory.
func NewConversationEntityMemory(model llm.LLM) *ConversationEntityMemory {
	return &ConversationEntityMemory{
		llm:        model,
		entities:   make(map[string]string),
		HistoryKey: "history",
		MaxRecent:  4,
	}
}

func (m *ConversationEntityMemory) Messages() []schema.Message {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.buildMessages()
}

func (m *ConversationEntityMemory) LoadMemoryVariables(_ context.Context) (map[string]any, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return map[string]any{m.HistoryKey: m.buildMessages()}, nil
}

func (m *ConversationEntityMemory) buildMessages() []schema.Message {
	var msgs []schema.Message

	entityCtx := m.entityContext()
	if entityCtx != "" {
		msgs = append(msgs, schema.NewSystemMessage("Entities mentioned in conversation:\n"+entityCtx))
	}

	recent := m.recent()
	msgs = append(msgs, recent...)
	return msgs
}

func (m *ConversationEntityMemory) entityContext() string {
	if len(m.entities) == 0 {
		return ""
	}
	var sb strings.Builder
	for name, summary := range m.entities {
		fmt.Fprintf(&sb, "%s: %s\n", name, summary)
	}
	return strings.TrimSpace(sb.String())
}

func (m *ConversationEntityMemory) recent() []schema.Message {
	n := len(m.messages)
	if m.MaxRecent > 0 {
		start := n - m.MaxRecent*2
		if start < 0 {
			start = 0
		}
		cp := make([]schema.Message, n-start)
		copy(cp, m.messages[start:])
		return cp
	}
	cp := make([]schema.Message, n)
	copy(cp, m.messages)
	return cp
}

func (m *ConversationEntityMemory) SaveContext(ctx context.Context, humanInput, aiOutput string) error {
	m.mu.Lock()
	m.messages = append(m.messages,
		schema.NewHumanMessage(humanInput),
		schema.NewAIMessage(aiOutput),
	)
	m.mu.Unlock()

	return m.extractEntities(ctx, humanInput+"\n"+aiOutput)
}

func (m *ConversationEntityMemory) Clear(_ context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.messages = nil
	m.entities = make(map[string]string)
	return nil
}

func (m *ConversationEntityMemory) SaveMessages(ctx context.Context, msgs []schema.Message) error {
	m.mu.Lock()
	m.messages = append(m.messages, msgs...)
	m.mu.Unlock()

	var sb strings.Builder
	for _, msg := range msgs {
		sb.WriteString(msg.Content)
		sb.WriteString("\n")
	}
	return m.extractEntities(ctx, sb.String())
}

func (m *ConversationEntityMemory) extractEntities(ctx context.Context, text string) error {
	prompt := fmt.Sprintf(`Extract all named entities (people, places, organisations, products, concepts) from the text below.
For each entity, provide a concise one-sentence summary of what was said about it.
Respond in this format:
ENTITY: <name>
SUMMARY: <one-sentence summary>
ENTITY: <name>
SUMMARY: <one-sentence summary>

If no entities are mentioned, respond with "NONE".

Text:
%s`, text)

	gen, err := m.llm.Generate(ctx, []schema.Message{
		schema.NewHumanMessage(prompt),
	})
	if err != nil {
		return fmt.Errorf("memory: entity extract: %w", err)
	}

	parsed := parseEntities(strings.TrimSpace(gen.Text))
	m.mu.Lock()
	for name, summary := range parsed {
		if existing, ok := m.entities[name]; ok {
			m.entities[name] = existing + " " + summary
		} else {
			m.entities[name] = summary
		}
	}
	m.mu.Unlock()
	return nil
}

// EntitySummary returns a copy of the entity->summary map.
func (m *ConversationEntityMemory) EntitySummary() map[string]string {
	m.mu.Lock()
	defer m.mu.Unlock()
	cp := make(map[string]string, len(m.entities))
	for k, v := range m.entities {
		cp[k] = v
	}
	return cp
}

func parseEntities(text string) map[string]string {
	if text == "NONE" || text == "" {
		return nil
	}
	result := make(map[string]string)
	var currentEntity, currentSummary string
	lines := strings.Split(text, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(strings.ToUpper(line), "ENTITY:") {
			if currentEntity != "" && currentSummary != "" {
				result[currentEntity] = currentSummary
			}
			currentEntity = strings.TrimSpace(line[7:]) // strip "ENTITY:"
			currentSummary = ""
		} else if strings.HasPrefix(strings.ToUpper(line), "SUMMARY:") {
			currentSummary = strings.TrimSpace(line[8:])
		}
	}
	if currentEntity != "" && currentSummary != "" {
		result[currentEntity] = currentSummary
	}
	return result
}
