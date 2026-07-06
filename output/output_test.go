package output_test

import (
	"testing"

	"github.com/grafaelw/golangchain/output"
)

// ---------------------------------------------------------------------------
// StrOutputParser
// ---------------------------------------------------------------------------

func TestStrOutputParser(t *testing.T) {
	p := output.StrOutputParser{}
	got, err := p.Parse("  hello world  ")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "hello world" {
		t.Errorf("want %q got %q", "hello world", got)
	}
	if p.FormatInstructions() != "" {
		t.Error("StrOutputParser should return empty format instructions")
	}
}

func TestStrOutputParser_Empty(t *testing.T) {
	p := output.StrOutputParser{}
	got, err := p.Parse("")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "" {
		t.Errorf("want empty string, got %q", got)
	}
}

// ---------------------------------------------------------------------------
// JSONOutputParser
// ---------------------------------------------------------------------------

func TestJSONOutputParser_Plain(t *testing.T) {
	p := output.JSONOutputParser{}
	got, err := p.Parse(`{"name":"Alice","age":30}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got["name"] != "Alice" {
		t.Errorf("name mismatch: %v", got["name"])
	}
	if got["age"].(float64) != 30 {
		t.Errorf("age mismatch: %v", got["age"])
	}
}

func TestJSONOutputParser_CodeFence(t *testing.T) {
	p := output.JSONOutputParser{}
	input := "```json\n{\"key\":\"value\"}\n```"
	got, err := p.Parse(input)
	if err != nil {
		t.Fatalf("code fence strip failed: %v", err)
	}
	if got["key"] != "value" {
		t.Errorf("key mismatch: %v", got["key"])
	}
}

func TestJSONOutputParser_UppercaseCodeFence(t *testing.T) {
	p := output.JSONOutputParser{}
	input := "```JSON\n{\"x\":1}\n```"
	got, err := p.Parse(input)
	if err != nil {
		t.Fatalf("uppercase fence: %v", err)
	}
	if got["x"].(float64) != 1 {
		t.Errorf("x mismatch")
	}
}

func TestJSONOutputParser_PlainFence(t *testing.T) {
	p := output.JSONOutputParser{}
	input := "```\n{\"y\":2}\n```"
	got, err := p.Parse(input)
	if err != nil {
		t.Fatalf("plain fence: %v", err)
	}
	if got["y"].(float64) != 2 {
		t.Errorf("y mismatch")
	}
}

func TestJSONOutputParser_Invalid(t *testing.T) {
	p := output.JSONOutputParser{}
	_, err := p.Parse("not json at all")
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestJSONOutputParser_FormatInstructions(t *testing.T) {
	p := output.JSONOutputParser{}
	if p.FormatInstructions() == "" {
		t.Error("FormatInstructions should not be empty")
	}
}

// ---------------------------------------------------------------------------
// StructOutputParser
// ---------------------------------------------------------------------------

type reply struct {
	Answer string `json:"answer"`
	Score  int    `json:"score"`
}

func TestStructOutputParser_Valid(t *testing.T) {
	p := output.NewStructOutputParser[reply]()
	got, err := p.Parse(`{"answer":"Paris","score":10}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Answer != "Paris" {
		t.Errorf("Answer mismatch: %q", got.Answer)
	}
	if got.Score != 10 {
		t.Errorf("Score mismatch: %d", got.Score)
	}
}

func TestStructOutputParser_CodeFence(t *testing.T) {
	p := output.NewStructOutputParser[reply]()
	got, err := p.Parse("```json\n{\"answer\":\"London\",\"score\":5}\n```")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Answer != "London" {
		t.Errorf("Answer mismatch: %q", got.Answer)
	}
}

func TestStructOutputParser_Invalid(t *testing.T) {
	p := output.NewStructOutputParser[reply]()
	_, err := p.Parse("garbage")
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

// ---------------------------------------------------------------------------
// ListOutputParser
// ---------------------------------------------------------------------------

func TestListOutputParser_Newline(t *testing.T) {
	p := output.NewListOutputParser(output.SepNewline)
	got, err := p.Parse("apple\nbanana\ncherry")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("want 3 items, got %d: %v", len(got), got)
	}
	if got[0] != "apple" || got[1] != "banana" || got[2] != "cherry" {
		t.Errorf("items mismatch: %v", got)
	}
}

func TestListOutputParser_Comma(t *testing.T) {
	p := output.NewListOutputParser(output.SepComma)
	got, err := p.Parse("red, green, blue")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("want 3 items, got %d: %v", len(got), got)
	}
}

func TestListOutputParser_EmptyLines(t *testing.T) {
	p := output.NewListOutputParser(output.SepNewline)
	got, err := p.Parse("a\n\nb\n\nc")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Empty lines should be stripped
	if len(got) != 3 {
		t.Fatalf("want 3 items, got %d: %v", len(got), got)
	}
}

func TestListOutputParser_FormatInstructions(t *testing.T) {
	pNL := output.NewListOutputParser(output.SepNewline)
	pC := output.NewListOutputParser(output.SepComma)
	if pNL.FormatInstructions() == "" {
		t.Error("newline parser format instructions empty")
	}
	if pC.FormatInstructions() == "" {
		t.Error("comma parser format instructions empty")
	}
}

// ---------------------------------------------------------------------------
// BoolOutputParser
// ---------------------------------------------------------------------------

func TestBoolOutputParser_True(t *testing.T) {
	p := output.BoolOutputParser{}
	for _, s := range []string{"yes", "YES", "Yes", "true", "True", "1", "y"} {
		got, err := p.Parse(s)
		if err != nil {
			t.Errorf("Parse(%q): unexpected error %v", s, err)
		}
		if !got {
			t.Errorf("Parse(%q): want true", s)
		}
	}
}

func TestBoolOutputParser_False(t *testing.T) {
	p := output.BoolOutputParser{}
	for _, s := range []string{"no", "NO", "false", "False", "0", "n"} {
		got, err := p.Parse(s)
		if err != nil {
			t.Errorf("Parse(%q): unexpected error %v", s, err)
		}
		if got {
			t.Errorf("Parse(%q): want false", s)
		}
	}
}

func TestBoolOutputParser_Unknown(t *testing.T) {
	p := output.BoolOutputParser{}
	_, err := p.Parse("maybe")
	if err == nil {
		t.Error("expected error for unknown value")
	}
}

func TestBoolOutputParser_FormatInstructions(t *testing.T) {
	p := output.BoolOutputParser{}
	if p.FormatInstructions() == "" {
		t.Error("FormatInstructions should not be empty")
	}
}

// ---------------------------------------------------------------------------
// AsAny adapter
// ---------------------------------------------------------------------------

func TestAsAny_StrParser(t *testing.T) {
	adapted := output.AsAny(output.StrOutputParser{})
	got, err := adapted.Parse("  trimmed  ")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.(string) != "trimmed" {
		t.Errorf("want %q got %q", "trimmed", got)
	}
}

func TestAsAny_StructParser(t *testing.T) {
	adapted := output.AsAny(output.NewStructOutputParser[reply]())
	got, err := adapted.Parse(`{"answer":"yes","score":1}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	r := got.(reply)
	if r.Answer != "yes" {
		t.Errorf("want answer=yes, got %q", r.Answer)
	}
}
