// Package schema defines the core shared types used throughout golangchain.
// All packages depend on schema; schema itself has no internal dependencies.
package schema

import (
	"encoding/json"
	"fmt"
	"strings"
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
	Role         Role          `json:"role"`
	Content      string        `json:"content"`
	ContentParts []ContentPart `json:"content_parts,omitempty"` // multimodal content blocks
	Name         string        `json:"name,omitempty"`          // for tool/function messages
	ToolCalls    []ToolCall    `json:"tool_calls,omitempty"`    // AI → tool call requests
	ToolCallID   string        `json:"tool_call_id,omitempty"`  // tool result message
}

// ContentPart is a multimodal content block (text, image, audio).
// When ContentParts is non-empty it takes precedence over Content
// for providers that support multimodal input (GPT-4V, Claude 3, Gemini).
type ContentPart struct {
	Type     string    `json:"type"` // "text", "image_url", "image_base64"
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
	Data     string    `json:"data,omitempty"`      // base64-encoded content
	MimeType string    `json:"mime_type,omitempty"` // e.g. "image/png"
}

// ImageURL references an image accessible by URL.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"` // "auto" (default), "low", "high"
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
	prefix := ""
	if m.Name != "" {
		prefix = fmt.Sprintf("[%s:%s]", m.Role, m.Name)
	} else if len(m.ToolCalls) > 0 {
		names := make([]string, len(m.ToolCalls))
		for i, tc := range m.ToolCalls {
			names[i] = tc.Name
		}
		prefix = fmt.Sprintf("[%s→tools=%v]", m.Role, names)
	} else {
		prefix = fmt.Sprintf("[%s]", m.Role)
	}

	if len(m.ContentParts) > 0 {
		texts := make([]string, 0, len(m.ContentParts))
		imageCount := 0
		for _, part := range m.ContentParts {
			switch part.Type {
			case "text":
				texts = append(texts, part.Text)
			case "image_url", "image_base64":
				imageCount++
			default:
				texts = append(texts, fmt.Sprintf("<%s>", part.Type))
			}
		}
		combined := strings.Join(texts, " | ")
		if imageCount > 0 {
			combined += fmt.Sprintf(" [+%d image(s)]", imageCount)
		}
		return fmt.Sprintf("%s %s", prefix, truncateStr(combined, 120))
	}
	return fmt.Sprintf("%s %s", prefix, truncateStr(m.Content, 120))
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
	Text          string     `json:"text"`
	Message       Message    `json:"message"`
	StopReason    string     `json:"stop_reason,omitempty"`
	Usage         TokenUsage `json:"usage"`
	EstimatedCost float64    `json:"estimated_cost,omitempty"` // USD
}

// String returns the generation text, truncated for display.
func (g Generation) String() string {
	cost := ""
	if g.EstimatedCost > 0 {
		cost = fmt.Sprintf(", $%.6f", g.EstimatedCost)
	}
	if g.StopReason != "" {
		return fmt.Sprintf("Generation(stop=%s, tokens=%s%s): %s",
			g.StopReason, g.Usage.String(), cost, truncateStr(g.Text, 200))
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
	// ToolCalls carries completed tool calls on the Done chunk.
	ToolCalls []ToolCall
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

// CostPerToken holds USD pricing per 1000 tokens for a model.
// Multiply 1K-token price * tokens / 1000 to get the dollar cost.
type CostPerToken struct {
	PromptPrice     float64
	CompletionPrice float64
}

// Multiply returns the estimated cost for the given token counts.
func (c CostPerToken) Multiply(promptTokens, completionTokens int) float64 {
	return c.PromptPrice*float64(promptTokens)/1000.0 +
		c.CompletionPrice*float64(completionTokens)/1000.0
}

// ModelPricing is a lookup for known model prices (USD per 1K tokens).
// Providers use this to set Generation.EstimatedCost automatically.
//
//nolint:lll
var ModelPricing = map[string]CostPerToken{
	// OpenAI
	"gpt-4o":                 {0.0025, 0.010},
	"gpt-4o-mini":            {0.00015, 0.0006},
	"gpt-4-turbo":            {0.01, 0.03},
	"gpt-4":                  {0.03, 0.06},
	"gpt-4-32k":              {0.06, 0.12},
	"gpt-3.5-turbo":          {0.0005, 0.0015},
	"gpt-3.5-turbo-instruct": {0.0015, 0.002},
	// OpenAI embedding
	"text-embedding-3-small": {0.00002, 0},
	"text-embedding-3-large": {0.00013, 0},
	"text-embedding-ada-002": {0.0001, 0},
	// Anthropic
	"claude-3-5-sonnet": {0.003, 0.015},
	"claude-3-5-haiku":  {0.0008, 0.004},
	"claude-3-opus":     {0.015, 0.075},
	"claude-3-sonnet":   {0.003, 0.015},
	"claude-3-haiku":    {0.00025, 0.00125},
	// Google
	"gemini-2.5-flash": {0.00015, 0.0006},
	"gemini-2.5-pro":   {0.00125, 0.010},
	"gemini-2.0-flash": {0.00015, 0.0006},
	"gemini-1.5-pro":   {0.00125, 0.005},
	"gemini-1.5-flash": {0.000075, 0.0003},
}

// EstimateCost looks up the model in ModelPricing and returns the
// dollar cost. Returns 0 if the model is not in the pricing table.
func (u TokenUsage) EstimateCost(model string) float64 {
	p, ok := ModelPricing[model]
	if !ok {
		return 0
	}
	return p.Multiply(u.PromptTokens, u.CompletionTokens)
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
