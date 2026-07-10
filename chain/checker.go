package chain

import (
	"context"
	"fmt"
	"strings"

	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/schema"
)

// LLMCheckerChain generates an answer, then asks the LLM to verify and
// correct itself. The final output is the improved, self-checked answer.
//
//	checker := chain.NewLLMCheckerChain(model)
//	ans, _ := checker.Invoke(ctx, "Explain quantum entanglement.")
type LLMCheckerChain struct {
	LLM         llm.LLM
	LLMOptions  []llm.Option
	DraftPrompt string
	CheckPrompt string
	Name        string
}

// NewLLMCheckerChain creates a self-checking chain with default prompts.
func NewLLMCheckerChain(model llm.LLM, opts ...llm.Option) *LLMCheckerChain {
	return &LLMCheckerChain{
		LLM:         model,
		LLMOptions:  opts,
		DraftPrompt: "Answer the following question:\n{{ .question }}",
		CheckPrompt: `You are given a question and a draft answer. Verify the answer for correctness, completeness, and clarity. If it needs improvement, rewrite it. Return ONLY the final improved answer.

Question: {{ .question }}

Draft answer: {{ .draft }}

Improved answer:`,
		Name: "LLMCheckerChain",
	}
}

func (c *LLMCheckerChain) Invoke(ctx context.Context, input any) (any, error) {
	question := fmt.Sprint(input)

	// Draft phase
	draftPrompt := strings.ReplaceAll(c.DraftPrompt, "{{ .question }}", question)
	draftGen, err := c.LLM.Generate(ctx, []schema.Message{schema.NewHumanMessage(draftPrompt)}, c.LLMOptions...)
	if err != nil {
		return nil, fmt.Errorf("%s: draft: %w", c.Name, err)
	}
	draft := strings.TrimSpace(draftGen.Text)

	// Check phase
	checkPrompt := c.CheckPrompt
	checkPrompt = strings.ReplaceAll(checkPrompt, "{{ .question }}", question)
	checkPrompt = strings.ReplaceAll(checkPrompt, "{{ .draft }}", draft)
	checkGen, err := c.LLM.Generate(ctx, []schema.Message{schema.NewHumanMessage(checkPrompt)}, c.LLMOptions...)
	if err != nil {
		return nil, fmt.Errorf("%s: check: %w", c.Name, err)
	}

	return map[string]any{
		"question": question,
		"draft":    draft,
		"answer":   strings.TrimSpace(checkGen.Text),
	}, nil
}

func (c *LLMCheckerChain) Stream(ctx context.Context, input any) (<-chan schema.StreamChunk, error) {
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

func (c *LLMCheckerChain) Pipe(next Runnable) Runnable { return &pipeRunnable{first: c, second: next} }
func (c *LLMCheckerChain) Batch(ctx context.Context, inputs []any) ([]any, error) {
	return RunBatch(ctx, c, inputs)
}
