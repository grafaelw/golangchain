package chain_test

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/grafaelw/golangchain/chain"
	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/output"
	"github.com/grafaelw/golangchain/prompt"
	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// mockLLM — deterministic stub for tests
// ---------------------------------------------------------------------------

type mockLLM struct {
	response string
	err      error
	calls    int
}

func (m *mockLLM) Generate(_ context.Context, _ []schema.Message, _ ...llm.Option) (*schema.Generation, error) {
	m.calls++
	if m.err != nil {
		return nil, m.err
	}
	return &schema.Generation{
		Text:    m.response,
		Message: schema.NewAIMessage(m.response),
	}, nil
}

func (m *mockLLM) Stream(_ context.Context, _ []schema.Message, _ ...llm.Option) (<-chan schema.StreamChunk, error) {
	if m.err != nil {
		return nil, m.err
	}
	ch := make(chan schema.StreamChunk, 3)
	// Emit response in 3 chunks
	words := strings.Fields(m.response)
	go func() {
		defer close(ch)
		for i, w := range words {
			sep := ""
			if i > 0 {
				sep = " "
			}
			ch <- schema.StreamChunk{Text: sep + w}
		}
		ch <- schema.StreamChunk{Done: true}
	}()
	return ch, nil
}

func (m *mockLLM) ModelName() string { return "mock-model" }

// ---------------------------------------------------------------------------
// FuncRunnable
// ---------------------------------------------------------------------------

func TestFuncRunnable_Invoke(t *testing.T) {
	r := chain.NewFuncRunnable("double", func(_ context.Context, input any) (any, error) {
		return input.(int) * 2, nil
	})
	got, err := r.Invoke(context.Background(), 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.(int) != 10 {
		t.Errorf("want 10, got %v", got)
	}
}

func TestFuncRunnable_Stream(t *testing.T) {
	r := chain.NewFuncRunnable("echo", func(_ context.Context, input any) (any, error) {
		return input, nil
	})
	ch, err := r.Stream(context.Background(), "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var last schema.StreamChunk
	for c := range ch {
		last = c
	}
	if !last.Done {
		t.Error("expected Done=true on last chunk")
	}
	if last.Value.(string) != "hello" {
		t.Errorf("want hello, got %v", last.Value)
	}
}

func TestFuncRunnable_Error(t *testing.T) {
	r := chain.NewFuncRunnable("fail", func(_ context.Context, _ any) (any, error) {
		return nil, fmt.Errorf("intentional error")
	})
	_, err := r.Invoke(context.Background(), nil)
	if err == nil {
		t.Error("expected error")
	}
}

// ---------------------------------------------------------------------------
// pipeRunnable (via Pipe)
// ---------------------------------------------------------------------------

func TestPipe_TwoSteps(t *testing.T) {
	add1 := chain.NewFuncRunnable("add1", func(_ context.Context, in any) (any, error) {
		return in.(int) + 1, nil
	})
	mul2 := chain.NewFuncRunnable("mul2", func(_ context.Context, in any) (any, error) {
		return in.(int) * 2, nil
	})
	// (5 + 1) * 2 = 12
	piped := add1.Pipe(mul2)
	got, err := piped.Invoke(context.Background(), 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.(int) != 12 {
		t.Errorf("want 12, got %v", got)
	}
}

func TestPipe_ThreeSteps(t *testing.T) {
	a := chain.NewFuncRunnable("a", func(_ context.Context, in any) (any, error) { return in.(int) + 1, nil })
	b := chain.NewFuncRunnable("b", func(_ context.Context, in any) (any, error) { return in.(int) * 3, nil })
	c := chain.NewFuncRunnable("c", func(_ context.Context, in any) (any, error) { return in.(int) - 2, nil })
	// (2+1)*3-2 = 7
	piped := a.Pipe(b).Pipe(c)
	got, err := piped.Invoke(context.Background(), 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.(int) != 7 {
		t.Errorf("want 7, got %v", got)
	}
}

func TestPipe_Stream(t *testing.T) {
	a := chain.NewFuncRunnable("a", func(_ context.Context, in any) (any, error) { return in.(int) + 10, nil })
	b := chain.NewFuncRunnable("b", func(_ context.Context, in any) (any, error) { return in.(int) * 2, nil })
	piped := a.Pipe(b)

	ch, err := piped.Stream(context.Background(), 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var result any
	for chunk := range ch {
		if chunk.Done {
			result = chunk.Value
		}
	}
	if result.(int) != 30 { // (5+10)*2
		t.Errorf("want 30, got %v", result)
	}
}

func TestPipe_EarlyStepError(t *testing.T) {
	fail := chain.NewFuncRunnable("fail", func(_ context.Context, _ any) (any, error) {
		return nil, fmt.Errorf("step failed")
	})
	noop := chain.NewFuncRunnable("noop", func(_ context.Context, in any) (any, error) { return in, nil })
	piped := fail.Pipe(noop)
	_, err := piped.Invoke(context.Background(), "x")
	if err == nil {
		t.Error("expected error from first step")
	}
}

// ---------------------------------------------------------------------------
// LLMRunnable
// ---------------------------------------------------------------------------

func TestLLMRunnable_Invoke(t *testing.T) {
	mock := &mockLLM{response: "Paris"}
	r := chain.NewLLMRunnable(mock)

	msgs := []schema.Message{schema.NewHumanMessage("Capital of France?")}
	got, err := r.Invoke(context.Background(), msgs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	gen := got.(*schema.Generation)
	if gen.Text != "Paris" {
		t.Errorf("want Paris, got %q", gen.Text)
	}
}

func TestLLMRunnable_StringInput(t *testing.T) {
	mock := &mockLLM{response: "42"}
	r := chain.NewLLMRunnable(mock)
	got, err := r.Invoke(context.Background(), "What is 6*7?")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.(*schema.Generation).Text != "42" {
		t.Errorf("want 42")
	}
}

func TestLLMRunnable_Stream(t *testing.T) {
	mock := &mockLLM{response: "hello world"}
	r := chain.NewLLMRunnable(mock)

	ch, err := r.Stream(context.Background(), "say hi")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var text string
	for chunk := range ch {
		if chunk.Err != nil {
			t.Fatalf("stream error: %v", chunk.Err)
		}
		if s, ok := chunk.Value.(string); ok {
			text += s
		}
	}
	if !strings.Contains(text, "hello") {
		t.Errorf("expected 'hello' in stream, got %q", text)
	}
}

// ---------------------------------------------------------------------------
// LLMChain
// ---------------------------------------------------------------------------

func TestLLMChain_Invoke(t *testing.T) {
	chatPrompt := prompt.MustNewChatPromptTemplate(
		prompt.MustSystem("You are helpful."),
		prompt.MustHuman("{{.question}}"),
	)
	mock := &mockLLM{response: "  The answer is 42.  "}

	c := chain.NewLLMChain(chatPrompt, mock, output.AsAny(output.StrOutputParser{}))
	got, err := c.Invoke(context.Background(), map[string]any{"question": "what is the answer?"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.(string) != "The answer is 42." {
		t.Errorf("want trimmed answer, got %q", got)
	}
}

func TestLLMChain_Stream(t *testing.T) {
	chatPrompt := prompt.MustNewChatPromptTemplate(
		prompt.MustHuman("{{.q}}"),
	)
	mock := &mockLLM{response: "hello world test"}

	c := chain.NewLLMChain(chatPrompt, mock, output.AsAny(output.StrOutputParser{}))
	ch, err := c.Stream(context.Background(), map[string]any{"q": "say something"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var chunks []string
	var finalParsed any
	for chunk := range ch {
		if chunk.Err != nil {
			t.Fatalf("stream error: %v", chunk.Err)
		}
		if chunk.Done {
			finalParsed = chunk.Value
		} else {
			if s, ok := chunk.Value.(string); ok {
				chunks = append(chunks, s)
			}
		}
	}
	if finalParsed == nil {
		t.Error("expected final parsed value")
	}
	if len(chunks) == 0 {
		t.Error("expected incremental chunks")
	}
}

func TestLLMChain_LLMError(t *testing.T) {
	chatPrompt := prompt.MustNewChatPromptTemplate(prompt.MustHuman("{{.q}}"))
	mock := &mockLLM{err: fmt.Errorf("API down")}
	c := chain.NewLLMChain(chatPrompt, mock, output.AsAny(output.StrOutputParser{}))
	_, err := c.Invoke(context.Background(), map[string]any{"q": "hello"})
	if err == nil {
		t.Error("expected error when LLM fails")
	}
}

func TestLLMChain_JSONOutput(t *testing.T) {
	chatPrompt := prompt.MustNewChatPromptTemplate(prompt.MustHuman("{{.q}}"))
	mock := &mockLLM{response: `{"city":"Amsterdam"}`}
	c := chain.NewLLMChain(chatPrompt, mock, output.AsAny(output.JSONOutputParser{}))
	got, err := c.Invoke(context.Background(), map[string]any{"q": "give me JSON"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	m := got.(map[string]any)
	if m["city"] != "Amsterdam" {
		t.Errorf("city mismatch: %v", m["city"])
	}
}

func TestLLMChain_Pipe(t *testing.T) {
	chatPrompt := prompt.MustNewChatPromptTemplate(prompt.MustHuman("{{.q}}"))
	mock := &mockLLM{response: "hello"}

	c := chain.NewLLMChain(chatPrompt, mock, output.AsAny(output.StrOutputParser{}))
	upper := chain.NewFuncRunnable("upper", func(_ context.Context, in any) (any, error) {
		return strings.ToUpper(in.(string)), nil
	})
	piped := c.Pipe(upper)
	got, err := piped.Invoke(context.Background(), map[string]any{"q": "say hi"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.(string) != "HELLO" {
		t.Errorf("want HELLO, got %q", got)
	}
}

// ---------------------------------------------------------------------------
// SequentialChain
// ---------------------------------------------------------------------------

func TestSequentialChain_Invoke(t *testing.T) {
	step1 := chain.NewFuncRunnable("s1", func(_ context.Context, in any) (any, error) {
		return in.(int) + 5, nil
	})
	step2 := chain.NewFuncRunnable("s2", func(_ context.Context, in any) (any, error) {
		return in.(int) * 2, nil
	})
	seq := chain.NewSequentialChain("seq", step1, step2)
	got, err := seq.Invoke(context.Background(), 10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.(int) != 30 { // (10+5)*2
		t.Errorf("want 30, got %v", got)
	}
}

func TestSequentialChain_Stream(t *testing.T) {
	step1 := chain.NewFuncRunnable("s1", func(_ context.Context, in any) (any, error) {
		return in.(int) + 1, nil
	})
	step2 := chain.NewFuncRunnable("s2", func(_ context.Context, in any) (any, error) {
		return in.(int) * 3, nil
	})
	seq := chain.NewSequentialChain("seq", step1, step2)
	ch, err := seq.Stream(context.Background(), 4)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var last schema.StreamChunk
	for c := range ch {
		last = c
	}
	if last.Value.(int) != 15 { // (4+1)*3
		t.Errorf("want 15, got %v", last.Value)
	}
}

func TestSequentialChain_StepError(t *testing.T) {
	ok := chain.NewFuncRunnable("ok", func(_ context.Context, in any) (any, error) { return in, nil })
	fail := chain.NewFuncRunnable("fail", func(_ context.Context, _ any) (any, error) {
		return nil, fmt.Errorf("step 2 failed")
	})
	seq := chain.NewSequentialChain("seq", ok, fail)
	_, err := seq.Invoke(context.Background(), "x")
	if err == nil {
		t.Error("expected error from failing step")
	}
}

// ---------------------------------------------------------------------------
// MapChain
// ---------------------------------------------------------------------------

func TestMapChain_Invoke(t *testing.T) {
	branches := map[string]chain.Runnable{
		"double": chain.NewFuncRunnable("d", func(_ context.Context, in any) (any, error) {
			return in.(int) * 2, nil
		}),
		"triple": chain.NewFuncRunnable("t", func(_ context.Context, in any) (any, error) {
			return in.(int) * 3, nil
		}),
	}
	m := chain.NewMapChain("map", branches)
	got, err := m.Invoke(context.Background(), 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	result := got.(map[string]any)
	if result["double"].(int) != 10 {
		t.Errorf("double: want 10, got %v", result["double"])
	}
	if result["triple"].(int) != 15 {
		t.Errorf("triple: want 15, got %v", result["triple"])
	}
}

func TestMapChain_BranchError(t *testing.T) {
	branches := map[string]chain.Runnable{
		"ok": chain.NewFuncRunnable("ok", func(_ context.Context, in any) (any, error) { return in, nil }),
		"fail": chain.NewFuncRunnable("fail", func(_ context.Context, _ any) (any, error) {
			return nil, fmt.Errorf("branch failed")
		}),
	}
	m := chain.NewMapChain("map", branches)
	_, err := m.Invoke(context.Background(), "x")
	if err == nil {
		t.Error("expected error from failing branch")
	}
}

// ---------------------------------------------------------------------------
// RouterChain
// ---------------------------------------------------------------------------

func TestRouterChain_Routes(t *testing.T) {
	router := func(_ context.Context, in any) (string, error) {
		if in.(int) > 10 {
			return "big", nil
		}
		return "small", nil
	}
	chains := map[string]chain.Runnable{
		"big":   chain.NewFuncRunnable("big", func(_ context.Context, in any) (any, error) { return "BIG", nil }),
		"small": chain.NewFuncRunnable("small", func(_ context.Context, in any) (any, error) { return "SMALL", nil }),
	}
	rc := chain.NewRouterChain("router", router, chains, nil)

	for input, want := range map[int]string{5: "SMALL", 20: "BIG"} {
		got, err := rc.Invoke(context.Background(), input)
		if err != nil {
			t.Fatalf("Invoke(%d): %v", input, err)
		}
		if got.(string) != want {
			t.Errorf("input=%d: want %q got %q", input, want, got)
		}
	}
}

func TestRouterChain_Fallback(t *testing.T) {
	router := func(_ context.Context, _ any) (string, error) { return "unknown", nil }
	fallback := chain.NewFuncRunnable("fb", func(_ context.Context, _ any) (any, error) { return "fallback", nil })
	rc := chain.NewRouterChain("router", router, map[string]chain.Runnable{}, fallback)

	got, err := rc.Invoke(context.Background(), "x")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.(string) != "fallback" {
		t.Errorf("want fallback, got %q", got)
	}
}

func TestRouterChain_NoFallback_Error(t *testing.T) {
	router := func(_ context.Context, _ any) (string, error) { return "missing", nil }
	rc := chain.NewRouterChain("router", router, map[string]chain.Runnable{}, nil)
	_, err := rc.Invoke(context.Background(), "x")
	if err == nil {
		t.Error("expected error for unknown route with no fallback")
	}
}

// ---------------------------------------------------------------------------
// PromptTemplateFormatter adapter
// ---------------------------------------------------------------------------

func TestPromptTemplateFormatter(t *testing.T) {
	pt := prompt.MustNewPromptTemplate("Hello, {{.name}}!")
	f := chain.NewPromptTemplateFormatter(pt)
	msgs, err := f.FormatMessages(map[string]any{"name": "World"})
	if err != nil {
		t.Fatalf("FormatMessages: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("want 1 message, got %d", len(msgs))
	}
	if msgs[0].Role != schema.RoleHuman {
		t.Errorf("role: want human, got %q", msgs[0].Role)
	}
	if msgs[0].Content != "Hello, World!" {
		t.Errorf("content mismatch: %q", msgs[0].Content)
	}
}

// ---------------------------------------------------------------------------
// FallbackChain
// ---------------------------------------------------------------------------

func TestFallbackChain_Success(t *testing.T) {
	r := chain.NewFuncRunnable("ok", func(_ context.Context, in any) (any, error) {
		return "ok-" + in.(string), nil
	})
	fb := chain.NewFallbackChain("test", r)
	got, err := fb.Invoke(context.Background(), "input")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.(string) != "ok-input" {
		t.Errorf("want ok-input, got %q", got)
	}
}

func TestFallbackChain_Fallback(t *testing.T) {
	primary := chain.NewFuncRunnable("fail", func(_ context.Context, _ any) (any, error) {
		return nil, fmt.Errorf("boom")
	})
	backup := chain.NewFuncRunnable("ok", func(_ context.Context, _ any) (any, error) {
		return "fallback", nil
	})
	fb := chain.NewFallbackChain("test", primary, backup)
	got, err := fb.Invoke(context.Background(), "x")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.(string) != "fallback" {
		t.Errorf("want fallback, got %q", got)
	}
}

func TestFallbackChain_AllFail(t *testing.T) {
	r1 := chain.NewFuncRunnable("f1", func(_ context.Context, _ any) (any, error) {
		return nil, fmt.Errorf("fail1")
	})
	r2 := chain.NewFuncRunnable("f2", func(_ context.Context, _ any) (any, error) {
		return nil, fmt.Errorf("fail2")
	})
	fb := chain.NewFallbackChain("test", r1, r2)
	_, err := fb.Invoke(context.Background(), "x")
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestFallbackChain_Stream(t *testing.T) {
	// FallbackChain.Stream tries the Stream() error return from each runnable.
	// A FuncRunnable that returns an error from Invoke will emit it on the
	// channel, but the Stream() call itself succeeds. For true stream-level
	// fallback testing, use a runnable whose Stream() returns an error.
	backup := chain.NewFuncRunnable("ok", func(_ context.Context, _ any) (any, error) {
		return "stream-ok", nil
	})
	fb := chain.NewFallbackChain("test", backup)
	ch, err := fb.Stream(context.Background(), "x")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for c := range ch {
		if c.Err != nil {
			t.Fatalf("unexpected stream error: %v", c.Err)
		}
	}
}
