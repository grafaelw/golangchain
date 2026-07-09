// Package schema defines the core shared types used throughout golangchain.
// All packages depend on schema; schema itself has no internal dependencies.
package schema

import (
	"encoding/json"
	"fmt"
)

// ---------------------------------------------------------------------------
// Message roles
// ---------------------------------------------------------------------------

// Role identifies the participant in a conversation turn.
type Role string

const (
	RoleSystem   Role = "system"
	RoleHuman    Role = "human"
	RoleAI       Role = "ai"
	RoleTool     Role = "tool"
	RoleFunction Role = "function" // legacy OpenAI function role
)

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

// Message is a single conversation turn.
type Message struct {
	Role       Role       `json:"role"`
	Content    string     `json:"content"`
	Name       string     `json:"name,omitempty"`         // for tool/function messages
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`   // AI → tool call requests
	ToolCallID string     `json:"tool_call_id,omitempty"` // tool result message
}

// NewSystemMessage constructs a system message.
func NewSystemMessage(content string) Message {
	return Message{Role: RoleSystem, Content: content}
}

// NewHumanMessage constructs a human (user) message.
func NewHumanMessage(content string) Message {
	return Message{Role: RoleHuman, Content: content}
}

// NewAIMessage constructs an AI (assistant) message.
func NewAIMessage(content string) Message {
	return Message{Role: RoleAI, Content: content}
}

// NewToolMessage constructs a tool result message.
func NewToolMessage(content, toolCallID, name string) Message {
	return Message{Role: RoleTool, Content: content, ToolCallID: toolCallID, Name: name}
}

// String returns a compact representation suitable for debugging.
func (m Message) String() string {
	if m.Name != "" {
		return fmt.Sprintf("[%s:%s] %s", m.Role, m.Name, truncateStr(m.Content, 120))
	}
	if len(m.ToolCalls) > 0 {
		names := make([]string, len(m.ToolCalls))
		for i, tc := range m.ToolCalls {
			names[i] = tc.Name
		}
		return fmt.Sprintf("[%s→tools=%v] %s", m.Role, names, truncateStr(m.Content, 120))
	}
	return fmt.Sprintf("[%s] %s", m.Role, truncateStr(m.Content, 120))
}

// ---------------------------------------------------------------------------
// Tool calls
// ---------------------------------------------------------------------------

// ToolCall represents a request from the model to invoke a tool.
type ToolCall struct {
	ID        string          `json:"id"`
	Type      string          `json:"type"` // always "function" for now
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"` // JSON-encoded arguments
}

// String returns a compact tool call representation.
func (tc ToolCall) String() string {
	return fmt.Sprintf("ToolCall(%s, args=%s)", tc.Name, string(tc.Arguments))
}

// ToolDef describes a tool available for the model to call.
type ToolDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"` // JSON Schema object
}

// ---------------------------------------------------------------------------
// Generations
// ---------------------------------------------------------------------------

// Generation is the output of a single LLM call.
type Generation struct {
	Text       string     `json:"text"`
	Message    Message    `json:"message"`
	StopReason string     `json:"stop_reason,omitempty"`
	Usage      TokenUsage `json:"usage"`
}

// String returns the generation text, truncated for display.
func (g Generation) String() string {
	if g.StopReason != "" {
		return fmt.Sprintf("Generation(stop=%s, tokens=%s): %s",
			g.StopReason, g.Usage.String(), truncateStr(g.Text, 200))
	}
	return truncateStr(g.Text, 200)
}

// StreamChunk is a single token or partial value emitted during streaming.
// At the LLM layer Text carries incremental output; higher-level Runnables
// use the Value field for typed pipeline data.
type StreamChunk struct {
	// Text is the incremental text fragment (LLM streaming).
	Text string
	// Value is the typed partial value (Runnable pipeline streaming).
	Value any
	// ToolCallDelta carries partial tool-call data during streaming.
	ToolCallDelta *ToolCallDelta
	// Done is true on the final chunk (carries Usage).
	Done  bool
	Usage TokenUsage
	// Err carries any error that terminated the stream.
	Err error
}

// ToolCallDelta carries a partial tool call during streaming.
type ToolCallDelta struct {
	Index     int
	ID        string
	Name      string
	Arguments string // partial JSON
}

// TokenUsage holds token consumption statistics for a call.
type TokenUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// String returns token usage as "↑P↓C" (prompt/completion).
func (u TokenUsage) String() string {
	if u.TotalTokens == 0 {
		return "0 tok"
	}
	return fmt.Sprintf("↑%d↓%d tok", u.PromptTokens, u.CompletionTokens)
}

// ---------------------------------------------------------------------------
// Documents (for RAG)
// ---------------------------------------------------------------------------

// Document is a piece of text with associated metadata, used in RAG pipelines.
type Document struct {
	PageContent string         `json:"page_content"`
	Metadata    map[string]any `json:"metadata,omitempty"`
	Score       float64        `json:"score,omitempty"`
}

// String returns page content truncated to 200 characters, prefixed with
// score when available.
func (d Document) String() string {
	if d.Score != 0 {
		return fmt.Sprintf("Doc(%.4f): %s", d.Score, truncateStr(d.PageContent, 200))
	}
	return fmt.Sprintf("Doc: %s", truncateStr(d.PageContent, 200))
}

// ---------------------------------------------------------------------------
// Agent types
// ---------------------------------------------------------------------------

// AgentAction represents a decision by the agent to invoke a tool.
type AgentAction struct {
	Tool      string `json:"tool"`
	ToolInput string `json:"tool_input"`
	Log       string `json:"log"`
}

// String returns "tool(input)".
func (a AgentAction) String() string {
	return fmt.Sprintf("AgentAction(%s(%s))", a.Tool, a.ToolInput)
}

// AgentFinish represents the agent's final answer.
type AgentFinish struct {
	Output string `json:"output"`
	Log    string `json:"log"`
}

// String returns the output truncated.
func (f AgentFinish) String() string {
	return fmt.Sprintf("AgentFinish: %s", truncateStr(f.Output, 200))
}

// AgentStep pairs an action with the observation returned by the tool.
type AgentStep struct {
	Action      AgentAction `json:"action"`
	Observation string      `json:"observation"`
}

// String returns "tool → observation".
func (s AgentStep) String() string {
	return fmt.Sprintf("AgentStep: %s → %s", s.Action.String(), truncateStr(s.Observation, 120))
}

func truncateStr(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
