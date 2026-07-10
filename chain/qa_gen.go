package chain

import (
	"context"
	"fmt"
	"strings"

	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/schema"
)

// QAGenerationChain generates question-answer pairs from a document.
// Useful for building evaluation datasets from raw text.
//
//	gen := chain.NewQAGenerationChain(model)
//	pairs, _ := gen.Invoke(ctx, "Go was released by Google in 2009...")
//	// pairs is []map[string]string with "question" and "answer"
type QAGenerationChain struct {
	LLM        llm.LLM
	LLMOptions []llm.Option
	Prompt     string
	NumPairs   int
	Name       string
}

// DefaultQAGenPrompt generates QA pairs from text.
const DefaultQAGenPrompt = `Generate {{ .num }} question-answer pairs from the text below. Each pair should be self-contained and answerable solely from the text. Output ONLY valid JSON as a list of {"question": "...", "answer": "..."} objects.

Text:
{{ .text }}

JSON:`

// NewQAGenerationChain creates a QA generator.
func NewQAGenerationChain(model llm.LLM, opts ...llm.Option) *QAGenerationChain {
	return &QAGenerationChain{
		LLM:        model,
		LLMOptions: opts,
		Prompt:     DefaultQAGenPrompt,
		NumPairs:   3,
		Name:       "QAGenerationChain",
	}
}

// QAPair is a single generated question-answer pair.
type QAPair struct {
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

func (c *QAGenerationChain) Invoke(ctx context.Context, input any) (any, error) {
	text := fmt.Sprint(input)
	p := strings.ReplaceAll(c.Prompt, "{{ .num }}", fmt.Sprint(c.NumPairs))
	p = strings.ReplaceAll(p, "{{ .text }}", text)

	gen, err := c.LLM.Generate(ctx, []schema.Message{schema.NewHumanMessage(p)}, c.LLMOptions...)
	if err != nil {
		return nil, fmt.Errorf("%s: generate: %w", c.Name, err)
	}

	// Parse the JSON array of QA pairs
	result := parseQAPairs(gen.Text)
	if len(result) == 0 {
		return nil, fmt.Errorf("%s: no QA pairs generated from response: %s", c.Name, trunc(gen.Text, 200))
	}
	return result, nil
}

func (c *QAGenerationChain) Stream(ctx context.Context, input any) (<-chan schema.StreamChunk, error) {
	out, err := c.Invoke(ctx, input)
	ch := make(chan schema.StreamChunk, 1)
	if err != nil {
		ch <- schema.StreamChunk{Err: err}
		close(ch)
		return ch, nil
	}
	ch <- schema.StreamChunk{Value: out, Done: true}
	close(ch)
	return ch, nil
}

func (c *QAGenerationChain) Pipe(next Runnable) Runnable {
	return &pipeRunnable{first: c, second: next}
}
func (c *QAGenerationChain) Batch(ctx context.Context, inputs []any) ([]any, error) {
	return RunBatch(ctx, c, inputs)
}

// parseQAPairs extracts QA pairs from LLM output (handles code fences, trailing commas).
func parseQAPairs(raw string) []QAPair {
	raw = strings.TrimSpace(raw)
	raw = strings.TrimPrefix(raw, "```json")
	raw = strings.TrimPrefix(raw, "```")
	raw = strings.TrimSuffix(raw, "```")
	raw = strings.TrimSpace(raw)

	// Simple manual JSON array parser for robustness
	var pairs []QAPair
	inString := false
	escaped := false
	depth := 0
	var current strings.Builder

	for i := 0; i < len(raw); i++ {
		ch := raw[i]
		if escaped {
			current.WriteByte(ch)
			escaped = false
			continue
		}
		if ch == '\\' {
			current.WriteByte(ch)
			escaped = true
			continue
		}
		if ch == '"' {
			inString = !inString
			current.WriteByte(ch)
			continue
		}
		if !inString {
			if ch == '{' {
				depth++
				if depth == 1 {
					current.Reset()
					continue
				}
			}
			if ch == '}' {
				depth--
				if depth == 0 {
					current.WriteByte('}')
					obj := current.String()
					q := extractJSONField(obj, "question")
					a := extractJSONField(obj, "answer")
					if q != "" && a != "" {
						pairs = append(pairs, QAPair{Question: q, Answer: a})
					}
					current.Reset()
					continue
				}
			}
		}
		if depth > 0 {
			current.WriteByte(ch)
		}
	}
	return pairs
}

func extractJSONField(obj, field string) string {
	key := `"` + field + `":`
	idx := strings.Index(obj, key)
	if idx < 0 {
		return ""
	}
	start := idx + len(key)
	// skip whitespace
	for start < len(obj) && (obj[start] == ' ' || obj[start] == '\t' || obj[start] == '\n') {
		start++
	}
	if start >= len(obj) || obj[start] != '"' {
		return ""
	}
	start++ // skip opening quote
	escaped := false
	var sb strings.Builder
	for i := start; i < len(obj); i++ {
		if escaped {
			sb.WriteByte(obj[i])
			escaped = false
			continue
		}
		if obj[i] == '\\' {
			escaped = true
			continue
		}
		if obj[i] == '"' {
			break
		}
		sb.WriteByte(obj[i])
	}
	return sb.String()
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
