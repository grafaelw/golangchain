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
