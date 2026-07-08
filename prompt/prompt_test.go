package prompt_test

import (
	"testing"

	"github.com/grafaelw/golangchain/prompt"
	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// PromptTemplate
// ---------------------------------------------------------------------------

func TestPromptTemplate_Format(t *testing.T) {
	pt, err := prompt.NewPromptTemplate("Hello, {{.Name}}! You are {{.Age}} years old.")
	if err != nil {
		t.Fatalf("NewPromptTemplate: %v", err)
	}
	got, err := pt.Format(map[string]any{"Name": "Alice", "Age": 30})
	if err != nil {
		t.Fatalf("Format: %v", err)
	}
	want := "Hello, Alice! You are 30 years old."
	if got != want {
		t.Errorf("want %q got %q", want, got)
	}
}

func TestPromptTemplate_MissingKey(t *testing.T) {
	pt := prompt.MustNewPromptTemplate("Hello, {{.Name}}!")
	_, err := pt.Format(map[string]any{})
	if err == nil {
		t.Error("expected error for missing key, got nil")
	}
}

func TestPromptTemplate_InvalidTemplate(t *testing.T) {
	_, err := prompt.NewPromptTemplate("{{.Unclosed")
	if err == nil {
		t.Error("expected parse error for invalid template")
	}
}

func TestMustPromptTemplate_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for invalid template")
		}
	}()
	prompt.MustNewPromptTemplate("{{.Unclosed")
}

func TestPromptTemplate_Template(t *testing.T) {
	raw := "Hello {{.X}}"
	pt := prompt.MustNewPromptTemplate(raw)
	if pt.Template() != raw {
		t.Errorf("Template() want %q got %q", raw, pt.Template())
	}
}

// ---------------------------------------------------------------------------
// MessageTemplate helpers
// ---------------------------------------------------------------------------

func TestMessageTemplateHelpers(t *testing.T) {
	sys, err := prompt.NewSystemMessageTemplate("You are {{.Persona}}.")
	if err != nil {
		t.Fatalf("NewSystemMessageTemplate: %v", err)
	}
	if sys.Role != schema.RoleSystem {
		t.Errorf("system role mismatch")
	}

	human, err := prompt.NewHumanMessageTemplate("Question: {{.Q}}")
	if err != nil {
		t.Fatalf("NewHumanMessageTemplate: %v", err)
	}
	if human.Role != schema.RoleHuman {
		t.Errorf("human role mismatch")
	}

	ai, err := prompt.NewAIMessageTemplate("Answer: {{.A}}")
	if err != nil {
		t.Fatalf("NewAIMessageTemplate: %v", err)
	}
	if ai.Role != schema.RoleAI {
		t.Errorf("ai role mismatch")
	}
}

func TestMustHelpers(t *testing.T) {
	// Should not panic
	_ = prompt.MustSystem("System: {{.X}}")
	_ = prompt.MustHuman("Human: {{.X}}")
	_ = prompt.MustAI("AI: {{.X}}")
}

// ---------------------------------------------------------------------------
// ChatPromptTemplate
// ---------------------------------------------------------------------------

func TestChatPromptTemplate_Basic(t *testing.T) {
	cpt := prompt.MustNewChatPromptTemplate(
		prompt.MustSystem("You are {{.Persona}}."),
		prompt.MustHuman("{{.Question}}"),
	)

	msgs, err := cpt.FormatMessages(map[string]any{
		"Persona":  "a Go expert",
		"Question": "What is a goroutine?",
	})
	if err != nil {
		t.Fatalf("FormatMessages: %v", err)
	}
	if len(msgs) != 2 {
		t.Fatalf("want 2 messages, got %d", len(msgs))
	}
	if msgs[0].Role != schema.RoleSystem {
		t.Errorf("msg[0] role: want system got %q", msgs[0].Role)
	}
	if msgs[0].Content != "You are a Go expert." {
		t.Errorf("msg[0] content mismatch: %q", msgs[0].Content)
	}
	if msgs[1].Role != schema.RoleHuman {
		t.Errorf("msg[1] role: want human got %q", msgs[1].Role)
	}
	if msgs[1].Content != "What is a goroutine?" {
		t.Errorf("msg[1] content mismatch: %q", msgs[1].Content)
	}
}

func TestChatPromptTemplate_WithPlaceholder(t *testing.T) {
	cpt := prompt.MustNewChatPromptTemplate(
		prompt.MustSystem("Be helpful."),
		prompt.NewMessagePlaceholder("history"),
		prompt.MustHuman("{{.Question}}"),
	)

	history := []schema.Message{
		schema.NewHumanMessage("What is Go?"),
		schema.NewAIMessage("A compiled language."),
	}

	msgs, err := cpt.FormatMessages(map[string]any{
		"history":  history,
		"Question": "Tell me more.",
	})
	if err != nil {
		t.Fatalf("FormatMessages: %v", err)
	}
	// system + 2 history + 1 human = 4
	if len(msgs) != 4 {
		t.Fatalf("want 4 messages, got %d", len(msgs))
	}
	if msgs[1].Content != "What is Go?" {
		t.Errorf("history[0] content mismatch")
	}
	if msgs[3].Content != "Tell me more." {
		t.Errorf("last message content mismatch")
	}
}

func TestChatPromptTemplate_PlaceholderMissingIsSkipped(t *testing.T) {
	cpt := prompt.MustNewChatPromptTemplate(
		prompt.NewMessagePlaceholder("history"),
		prompt.MustHuman("{{.Question}}"),
	)
	// history not provided — should silently skip the placeholder
	msgs, err := cpt.FormatMessages(map[string]any{"Question": "hello"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("want 1 message, got %d", len(msgs))
	}
}

func TestChatPromptTemplate_WrongPlaceholderType(t *testing.T) {
	cpt := prompt.MustNewChatPromptTemplate(
		prompt.NewMessagePlaceholder("history"),
		prompt.MustHuman("{{.Q}}"),
	)
	_, err := cpt.FormatMessages(map[string]any{
		"history": "this is a string, not []schema.Message",
		"Q":       "hello",
	})
	if err == nil {
		t.Error("expected type error for wrong placeholder value")
	}
}

func TestNewChatPromptTemplate_InvalidSlotType(t *testing.T) {
	_, err := prompt.NewChatPromptTemplate(42) // unsupported type
	if err == nil {
		t.Error("expected error for unsupported slot type")
	}
}

// ---------------------------------------------------------------------------
// FewShotPromptTemplate
// ---------------------------------------------------------------------------

func TestFewShotPromptTemplate_Format(t *testing.T) {
	suffix := prompt.MustNewPromptTemplate("Input: {{.Input}}\nOutput:")
	fs := &prompt.FewShotPromptTemplate{
		Prefix:        "Classify the sentiment:",
		ExamplePrefix: "Input:",
		ExampleSuffix: "Output:",
		Examples: []prompt.Example{
			{Input: "I love Go", Output: "positive"},
			{Input: "I hate bugs", Output: "negative"},
		},
		SuffixTemplate: suffix,
	}
	result, err := fs.Format(map[string]any{"Input": "Go is okay"})
	if err != nil {
		t.Fatalf("Format: %v", err)
	}
	if result == "" {
		t.Error("expected non-empty result")
	}
	// Should contain the prefix and both examples
	for _, needle := range []string{"Classify", "I love Go", "positive", "I hate bugs", "Go is okay"} {
		if !contains(result, needle) {
			t.Errorf("result missing %q", needle)
		}
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || len(s) > 0 && containsStr(s, sub))
}

func containsStr(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
