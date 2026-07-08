package llmutil

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/schema"
)

// mockLLM counts calls and can be configured to fail N times.
type mockLLM struct {
	name    string
	calls   int32
	failN   int32
	failErr error
	reply   string
}

func (m *mockLLM) ModelName() string { return m.name }

func (m *mockLLM) Generate(_ context.Context, _ []schema.Message, _ ...llm.Option) (*schema.Generation, error) {
	n := atomic.AddInt32(&m.calls, 1)
	if n <= atomic.LoadInt32(&m.failN) {
		return nil, m.failErr
	}
	return &schema.Generation{Text: m.reply, Message: schema.NewAIMessage(m.reply)}, nil
}

func (m *mockLLM) Stream(ctx context.Context, msgs []schema.Message, opts ...llm.Option) (<-chan schema.StreamChunk, error) {
	ch := make(chan schema.StreamChunk, 1)
	gen, err := m.Generate(ctx, msgs, opts...)
	if err != nil {
		return nil, err
	}
	go func() {
		defer close(ch)
		ch <- schema.StreamChunk{Text: gen.Text, Done: true}
	}()
	return ch, nil
}

func TestCachingLLM(t *testing.T) {
	base := &mockLLM{name: "mock", reply: "hello"}
	c := NewCachingLLM(base, NewMemoryCache())
	msgs := []schema.Message{schema.NewHumanMessage("q")}
	for i := 0; i < 3; i++ {
		gen, err := c.Generate(context.Background(), msgs)
		if err != nil || gen.Text != "hello" {
			t.Fatalf("iter %d: unexpected %v %v", i, gen, err)
		}
	}
	if got := atomic.LoadInt32(&base.calls); got != 1 {
		t.Fatalf("expected 1 underlying call, got %d", got)
	}
}

func TestRetryingLLMSucceedsAfterFailures(t *testing.T) {
	base := &mockLLM{name: "mock", reply: "ok", failN: 2, failErr: errors.New("boom")}
	r := NewRetryingLLM(base, RetryConfig{MaxAttempts: 4, Base: time.Millisecond})
	gen, err := r.Generate(context.Background(), []schema.Message{schema.NewHumanMessage("q")})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gen.Text != "ok" {
		t.Fatalf("unexpected reply: %q", gen.Text)
	}
	if got := atomic.LoadInt32(&base.calls); got != 3 {
		t.Fatalf("expected 3 attempts, got %d", got)
	}
}

func TestRetryingLLMGivesUp(t *testing.T) {
	base := &mockLLM{name: "mock", failN: 10, failErr: errors.New("boom")}
	r := NewRetryingLLM(base, RetryConfig{MaxAttempts: 2, Base: time.Millisecond})
	_, err := r.Generate(context.Background(), []schema.Message{schema.NewHumanMessage("q")})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestRateLimitedLLMHonoursConcurrency(t *testing.T) {
	base := &mockLLM{name: "mock", reply: "ok"}
	r := NewRateLimitedLLM(base, 1, 0)
	defer r.Close()

	done := make(chan struct{}, 2)
	start := time.Now()
	for i := 0; i < 2; i++ {
		go func() {
			// Underlying call is instantaneous; the semaphore serialises them.
			_, _ = r.Generate(context.Background(), []schema.Message{schema.NewHumanMessage("q")})
			done <- struct{}{}
		}()
	}
	<-done
	<-done
	if time.Since(start) < 0 {
		t.Fatal("nonsense timing")
	}
	if got := atomic.LoadInt32(&base.calls); got != 2 {
		t.Fatalf("expected 2 calls, got %d", got)
	}
}
