// Package schema defines the core shared types used throughout golangchain.
// All packages depend on schema; schema itself has no internal dependencies.
package schema

import "encoding/json"

// ---------------------------------------------------------------------------
// Message roles
// ---------------------------------------------------------------------------

// Role identifies the participant in a conversation turn.
type Role string

const (
	RoleSystem    Role = "system"
	RoleHuman     Role = "human"
	RoleAI        Role = "ai"
	RoleTool      Role = "tool"
	RoleFunction  Role = "function" // legacy OpenAI function role
)

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

// Message is a single conversation turn.
type Message struct {
	Role       Role       `json:"role"`
	Content    string     `json:"content"`
	Name       string     `json:"name,omitempty"`        // for tool/function messages
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`  // AI → tool call requests
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
	// Text is the raw string output (for completion models).
	Text string `json:"text"`
	// Message is the structured output (for chat models).
	Message Message `json:"message"`
	// StopReason indicates why generation stopped (e.g. "stop", "length", "tool_calls").
	StopReason string `json:"stop_reason,omitempty"`
	// Usage contains token usage statistics.
	Usage TokenUsage `json:"usage"`
}

// StreamChunk is a single token or partial text emitted during streaming.
type StreamChunk struct {
	// Text is the incremental text fragment.
	Text string
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

// ---------------------------------------------------------------------------
// Documents (for RAG)
// ---------------------------------------------------------------------------

// Document is a piece of text with associated metadata, used in RAG pipelines.
type Document struct {
	PageContent string         `json:"page_content"`
	Metadata    map[string]any `json:"metadata,omitempty"`
	// Score is populated by similarity search results.
	Score float64 `json:"score,omitempty"`
}

// ---------------------------------------------------------------------------
// Agent types
// ---------------------------------------------------------------------------

// AgentAction represents a decision by the agent to invoke a tool.
type AgentAction struct {
	Tool      string `json:"tool"`
	ToolInput string `json:"tool_input"` // JSON or plain string
	Log       string `json:"log"`        // agent's reasoning / thought
}

// AgentFinish represents the agent's final answer.
type AgentFinish struct {
	Output string `json:"output"`
	Log    string `json:"log"`
}

// AgentStep pairs an action with the observation returned by the tool.
type AgentStep struct {
	Action      AgentAction `json:"action"`
	Observation string      `json:"observation"`
}
