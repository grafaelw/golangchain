// Package output provides output parsers that transform raw LLM text into
// typed Go values. Parsers implement the chain.OutputParser interface and
// can be composed directly in LLMChain or used standalone.
package output

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ---------------------------------------------------------------------------
// OutputParser interface (mirrored in chain to avoid circular imports)
// ---------------------------------------------------------------------------

// Parser transforms a raw string into a typed value.
// The type parameter T is the target Go type.
type Parser[T any] interface {
	Parse(text string) (T, error)
	// FormatInstructions returns a hint to append to the prompt so the model
	// knows what format to produce (optional — may return "").
	FormatInstructions() string
}

// ---------------------------------------------------------------------------
// StrOutputParser
// ---------------------------------------------------------------------------

// StrOutputParser passes the raw string through unchanged.
// Use it as the terminal parser in chains where you want the LLM's text as-is.
type StrOutputParser struct{}

func (StrOutputParser) Parse(text string) (string, error) { return strings.TrimSpace(text), nil }
func (StrOutputParser) FormatInstructions() string        { return "" }

// ---------------------------------------------------------------------------
// JSONOutputParser — parses into map[string]any
// ---------------------------------------------------------------------------

// JSONOutputParser unmarshals the LLM output into a map[string]any.
// It is tolerant of markdown code fences (```json ... ```) that models often emit.
type JSONOutputParser struct{}

func (JSONOutputParser) Parse(text string) (map[string]any, error) {
	text = stripCodeFence(text)
	var out map[string]any
	if err := json.Unmarshal([]byte(text), &out); err != nil {
		return nil, fmt.Errorf("output: JSON parse: %w (raw: %q)", err, truncate(text, 200))
	}
	return out, nil
}

func (JSONOutputParser) FormatInstructions() string {
	return "Respond with valid JSON only. Do not include markdown code fences."
}

// ---------------------------------------------------------------------------
// StructOutputParser — parses into a typed struct via JSON
// ---------------------------------------------------------------------------

// StructOutputParser[T] unmarshals LLM output JSON into a strongly-typed
// Go struct T.
//
//	type Reply struct {
//	    Answer string `json:"answer"`
//	    Score  int    `json:"score"`
//	}
//	parser := output.NewStructOutputParser[Reply]()
//	result, err := parser.Parse(llmText)
type StructOutputParser[T any] struct{}

// NewStructOutputParser constructs a StructOutputParser for type T.
func NewStructOutputParser[T any]() *StructOutputParser[T] {
	return &StructOutputParser[T]{}
}

func (p *StructOutputParser[T]) Parse(text string) (T, error) {
	var zero T
	text = stripCodeFence(text)
	if err := json.Unmarshal([]byte(text), &zero); err != nil {
		return zero, fmt.Errorf("output: struct parse: %w (raw: %q)", err, truncate(text, 200))
	}
	return zero, nil
}

func (p *StructOutputParser[T]) FormatInstructions() string {
	return "Respond with valid JSON matching the required schema. Do not include markdown code fences."
}

// ---------------------------------------------------------------------------
// ListOutputParser — parses a newline- or comma-separated list into []string
// ---------------------------------------------------------------------------

// ListSeparator controls how ListOutputParser splits the text.
type ListSeparator string

const (
	SepNewline ListSeparator = "newline"
	SepComma   ListSeparator = "comma"
)

// ListOutputParser splits the LLM output into a slice of strings.
type ListOutputParser struct {
	Separator ListSeparator
}

// NewListOutputParser constructs a ListOutputParser.
func NewListOutputParser(sep ListSeparator) *ListOutputParser {
	return &ListOutputParser{Separator: sep}
}

func (p *ListOutputParser) Parse(text string) ([]string, error) {
	text = strings.TrimSpace(text)
	var parts []string
	switch p.Separator {
	case SepComma:
		parts = strings.Split(text, ",")
	default: // newline
		parts = strings.Split(text, "\n")
	}
	out := make([]string, 0, len(parts))
	for _, s := range parts {
		s = strings.TrimSpace(s)
		if s != "" {
			out = append(out, s)
		}
	}
	return out, nil
}

func (p *ListOutputParser) FormatInstructions() string {
	switch p.Separator {
	case SepComma:
		return "Respond with a comma-separated list of values, nothing else."
	default:
		return "Respond with one item per line, nothing else."
	}
}

// ---------------------------------------------------------------------------
// BoolOutputParser
// ---------------------------------------------------------------------------

// BoolOutputParser interprets the LLM output as a boolean.
// It recognises "yes", "true", "1" (case-insensitive) as true.
type BoolOutputParser struct{}

func (BoolOutputParser) Parse(text string) (bool, error) {
	t := strings.ToLower(strings.TrimSpace(text))
	switch t {
	case "yes", "true", "1", "y":
		return true, nil
	case "no", "false", "0", "n":
		return false, nil
	default:
		return false, fmt.Errorf("output: bool parse: unrecognised value %q", t)
	}
}

func (BoolOutputParser) FormatInstructions() string {
	return "Respond with exactly 'yes' or 'no', nothing else."
}

// ---------------------------------------------------------------------------
// RegexParser — extracts a named capture group from the text
// ---------------------------------------------------------------------------



// ---------------------------------------------------------------------------
// AnyParser — adapts a typed Parser[T] to the chain.OutputParser interface
// ---------------------------------------------------------------------------

// AnyParser wraps a Parser[T] so it implements interface{ Parse(string) (any, error) },
// which is what chain.NewLLMChain expects.
//
//	chain.NewLLMChain(prompt, model, output.AsAny(output.StrOutputParser{}))
func AsAny[T any](p Parser[T]) interface{ Parse(string) (any, error) } {
	return &anyParser[T]{inner: p}
}

type anyParser[T any] struct{ inner Parser[T] }

func (a *anyParser[T]) Parse(text string) (any, error) { return a.inner.Parse(text) }

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// stripCodeFence removes leading/trailing ``` or ```json fences that models
// often wrap JSON output in.
func stripCodeFence(s string) string {
	s = strings.TrimSpace(s)
	// Remove opening fence
	for _, prefix := range []string{"```json", "```JSON", "```"} {
		if strings.HasPrefix(s, prefix) {
			s = s[len(prefix):]
			break
		}
	}
	// Remove closing fence
	s = strings.TrimSuffix(s, "```")
	return strings.TrimSpace(s)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
