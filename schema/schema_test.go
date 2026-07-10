package schema_test

import (
	"encoding/json"
	"testing"

	"github.com/grafaelw/golangchain/schema"
)

func TestMessageConstructors(t *testing.T) {
	tests := []struct {
		name    string
		msg     schema.Message
		role    schema.Role
		content string
	}{
		{"system", schema.NewSystemMessage("be helpful"), schema.RoleSystem, "be helpful"},
		{"human", schema.NewHumanMessage("hello"), schema.RoleHuman, "hello"},
		{"ai", schema.NewAIMessage("world"), schema.RoleAI, "world"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.msg.Role != tt.role {
				t.Errorf("role: want %q got %q", tt.role, tt.msg.Role)
			}
			if tt.msg.Content != tt.content {
				t.Errorf("content: want %q got %q", tt.content, tt.msg.Content)
			}
		})
	}
}

func TestNewToolMessage(t *testing.T) {
	msg := schema.NewToolMessage("result", "call-123", "calculator")
	if msg.Role != schema.RoleTool {
		t.Errorf("role: want %q got %q", schema.RoleTool, msg.Role)
	}
	if msg.Content != "result" {
		t.Errorf("content mismatch")
	}
	if msg.ToolCallID != "call-123" {
		t.Errorf("ToolCallID mismatch")
	}
	if msg.Name != "calculator" {
		t.Errorf("Name mismatch")
	}
}

func TestMessageJSONRoundtrip(t *testing.T) {
	orig := schema.Message{
		Role:    schema.RoleAI,
		Content: "hello",
		ToolCalls: []schema.ToolCall{
			{ID: "tc1", Type: "function", Name: "calc", Arguments: json.RawMessage(`{"x":1}`)},
		},
	}
	data, err := json.Marshal(orig)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got schema.Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.Role != orig.Role || got.Content != orig.Content {
		t.Errorf("roundtrip mismatch: got %+v", got)
	}
	if len(got.ToolCalls) != 1 || got.ToolCalls[0].Name != "calc" {
		t.Errorf("ToolCalls roundtrip failed")
	}
}

func TestTokenUsage(t *testing.T) {
	u := schema.TokenUsage{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30}
	if u.TotalTokens != 30 {
		t.Errorf("TotalTokens: want 30 got %d", u.TotalTokens)
	}
}

func TestDocumentFields(t *testing.T) {
	doc := schema.Document{
		PageContent: "Go is great",
		Metadata:    map[string]any{"source": "manual"},
		Score:       0.95,
	}
	if doc.PageContent != "Go is great" {
		t.Errorf("PageContent mismatch")
	}
	if doc.Metadata["source"] != "manual" {
		t.Errorf("Metadata mismatch")
	}
	if doc.Score != 0.95 {
		t.Errorf("Score mismatch")
	}
}

func TestAgentTypes(t *testing.T) {
	action := schema.AgentAction{Tool: "search", ToolInput: "golang", Log: "thinking"}
	step := schema.AgentStep{Action: action, Observation: "lots of results"}
	if step.Action.Tool != "search" {
		t.Errorf("AgentStep.Action.Tool mismatch")
	}
	if step.Observation != "lots of results" {
		t.Errorf("AgentStep.Observation mismatch")
	}
	finish := schema.AgentFinish{Output: "done", Log: "final"}
	if finish.Output != "done" {
		t.Errorf("AgentFinish.Output mismatch")
	}
}

// ---------------------------------------------------------------------------
// ContentPart (multimodal)
// ---------------------------------------------------------------------------

func TestContentPart_TextOnly(t *testing.T) {
	msg := schema.Message{
		Role: schema.RoleHuman,
		ContentParts: []schema.ContentPart{
			{Type: "text", Text: "Hello world"},
		},
	}
	if len(msg.ContentParts) != 1 {
		t.Fatalf("want 1 part, got %d", len(msg.ContentParts))
	}
	if msg.ContentParts[0].Text != "Hello world" {
		t.Errorf("text mismatch")
	}
}

func TestContentPart_ImageURL(t *testing.T) {
	msg := schema.Message{
		Role: schema.RoleHuman,
		ContentParts: []schema.ContentPart{
			{Type: "text", Text: "What is this?"},
			{
				Type:     "image_url",
				ImageURL: &schema.ImageURL{URL: "https://example.com/img.png", Detail: "high"},
			},
		},
	}
	if msg.ContentParts[1].ImageURL.URL != "https://example.com/img.png" {
		t.Errorf("URL mismatch")
	}
	if msg.ContentParts[1].ImageURL.Detail != "high" {
		t.Errorf("detail mismatch")
	}
}

func TestContentPart_String(t *testing.T) {
	msg := schema.Message{
		Role: schema.RoleHuman,
		ContentParts: []schema.ContentPart{
			{Type: "text", Text: "Describe"},
			{Type: "image_url", ImageURL: &schema.ImageURL{URL: "http://x.com/a.png"}},
		},
	}
	s := msg.String()
	if s == "" {
		t.Error("expected non-empty string")
	}
}

// ---------------------------------------------------------------------------
// Cost tracking
// ---------------------------------------------------------------------------

func TestCostPerToken_Multiply(t *testing.T) {
	c := schema.CostPerToken{PromptPrice: 0.01, CompletionPrice: 0.03}
	cost := c.Multiply(1000, 500)
	if cost != 0.025 {
		t.Errorf("want 0.025, got %f", cost)
	}
}

func TestTokenUsage_EstimateCost_KnownModel(t *testing.T) {
	u := schema.TokenUsage{PromptTokens: 1000, CompletionTokens: 1000}
	cost := u.EstimateCost("gpt-4o")
	if cost <= 0 {
		t.Error("expected positive cost")
	}
	// 1000 prompt $0.0025 + 1000 completion $0.010 = $0.0125
	if cost != 0.0125 {
		t.Errorf("want 0.0125, got %f", cost)
	}
}

func TestTokenUsage_EstimateCost_UnknownModel(t *testing.T) {
	u := schema.TokenUsage{PromptTokens: 1000, CompletionTokens: 500}
	cost := u.EstimateCost("unknown-model-v2")
	if cost != 0 {
		t.Errorf("want 0, got %f", cost)
	}
}

func TestModelPricing_HasEntries(t *testing.T) {
	if len(schema.ModelPricing) < 10 {
		t.Errorf("want at least 10 pricing entries, got %d", len(schema.ModelPricing))
	}
	for name, price := range schema.ModelPricing {
		if price.PromptPrice < 0 || price.CompletionPrice < 0 {
			t.Errorf("model %q has negative pricing", name)
		}
	}
}

func TestGeneration_EstimatedCost(t *testing.T) {
	gen := &schema.Generation{
		Text:          "answer",
		EstimatedCost: 0.0125,
		Usage:         schema.TokenUsage{PromptTokens: 100, CompletionTokens: 50},
	}
	if gen.EstimatedCost != 0.0125 {
		t.Errorf("EstimatedCost mismatch")
	}
}

// ---------------------------------------------------------------------------
// StreamChunk ToolCalls
// ---------------------------------------------------------------------------

func TestStreamChunk_ToolCalls(t *testing.T) {
	tc := schema.ToolCall{ID: "call-1", Name: "calc", Arguments: json.RawMessage(`{"x":1}`)}
	chunk := schema.StreamChunk{
		Done:      true,
		ToolCalls: []schema.ToolCall{tc},
	}
	if len(chunk.ToolCalls) != 1 {
		t.Fatalf("want 1 tool call, got %d", len(chunk.ToolCalls))
	}
	if chunk.ToolCalls[0].Name != "calc" {
		t.Errorf("name mismatch")
	}
}
