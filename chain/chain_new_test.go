package chain_test

import (
	"context"
	"testing"

	"github.com/grafaelw/golangchain/chain"
	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/schema"
)

// mockRouteLLM classifies input by examining the text.
type mockRouteLLM struct {
	response string
}

func (m *mockRouteLLM) Generate(_ context.Context, _ []schema.Message, _ ...llm.Option) (*schema.Generation, error) {
	return &schema.Generation{Text: m.response, Message: schema.NewAIMessage(m.response)}, nil
}
func (m *mockRouteLLM) Stream(_ context.Context, _ []schema.Message, _ ...llm.Option) (<-chan schema.StreamChunk, error) {
	ch := make(chan schema.StreamChunk, 1)
	ch <- schema.StreamChunk{Text: m.response, Done: true}
	close(ch)
	return ch, nil
}
func (m *mockRouteLLM) ModelName() string { return "mock" }

func TestLLMRouterChain_Routes(t *testing.T) {
	llm := &mockRouteLLM{response: "math"}
	mathChain := chain.NewFuncRunnable("math", func(_ context.Context, in any) (any, error) {
		return "[MATH] " + in.(string), nil
	})
	scienceChain := chain.NewFuncRunnable("science", func(_ context.Context, in any) (any, error) {
		return "[SCIENCE] " + in.(string), nil
	})
	router := chain.NewLLMRouterChain("test", llm,
		map[string]chain.Runnable{"math": mathChain, "science": scienceChain},
		"math: for math questions", "science: for science questions",
	)
	got, err := router.Invoke(context.Background(), "2+2")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.(string) != "[MATH] 2+2" {
		t.Errorf("want [MATH] 2+2, got %q", got)
	}
}

func TestLLMRouterChain_UnknownRoute(t *testing.T) {
	llm := &mockRouteLLM{response: "history"}
	router := chain.NewLLMRouterChain("test", llm,
		map[string]chain.Runnable{},
		"math: for math",
	)
	_, err := router.Invoke(context.Background(), "question")
	if err == nil {
		t.Fatal("expected error for unknown route")
	}
}

// ---------------------------------------------------------------------------
// LLMCheckerChain
// ---------------------------------------------------------------------------

func TestLLMCheckerChain(t *testing.T) {
	llm := &mockRouteLLM{response: "Verified: 2+2=4"}
	checker := chain.NewLLMCheckerChain(llm)
	got, err := checker.Invoke(context.Background(), "What is 2+2?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	res, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("expected map[string]any, got %T", got)
	}
	if res["answer"] != "Verified: 2+2=4" {
		t.Errorf("unexpected answer: %v", res["answer"])
	}
}

// ---------------------------------------------------------------------------
// QAGenerationChain
// ---------------------------------------------------------------------------

type mockQALLM struct {
	response string
}

func (m *mockQALLM) Generate(_ context.Context, _ []schema.Message, _ ...llm.Option) (*schema.Generation, error) {
	return &schema.Generation{Text: m.response, Message: schema.NewAIMessage(m.response)}, nil
}
func (m *mockQALLM) Stream(_ context.Context, _ []schema.Message, _ ...llm.Option) (<-chan schema.StreamChunk, error) {
	ch := make(chan schema.StreamChunk, 1)
	ch <- schema.StreamChunk{Text: m.response, Done: true}
	close(ch)
	return ch, nil
}
func (m *mockQALLM) ModelName() string { return "mock" }

func TestQAGenerationChain(t *testing.T) {
	llm := &mockQALLM{response: `[{"question": "What is Go?", "answer": "A programming language by Google."}]`}
	gen := chain.NewQAGenerationChain(llm)
	got, err := gen.Invoke(context.Background(), "Go is a programming language created at Google in 2009.")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	pairs, ok := got.([]chain.QAPair)
	if !ok {
		t.Fatalf("expected []chain.QAPair, got %T", got)
	}
	if len(pairs) != 1 {
		t.Errorf("want 1 pair, got %d", len(pairs))
	}
	if pairs[0].Question != "What is Go?" {
		t.Errorf("question mismatch: %q", pairs[0].Question)
	}
}

func TestQAGenerationChain_NoPairs(t *testing.T) {
	llm := &mockQALLM{response: `no pairs here`}
	gen := chain.NewQAGenerationChain(llm)
	_, err := gen.Invoke(context.Background(), "text")
	if err == nil {
		t.Fatal("expected error for no pairs generated")
	}
}
