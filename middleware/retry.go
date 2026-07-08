package middleware

import (
	"context"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// ModelRetryMiddleware
// ---------------------------------------------------------------------------

// ModelRetryMiddleware retries failed LLM calls with exponential backoff.
type ModelRetryMiddleware struct {
	NoOpMiddleware
	MaxRetries int
	BaseDelay  time.Duration
	MaxDelay   time.Duration
	attempts   int
}

// ModelRetryOption configures a ModelRetryMiddleware.
type ModelRetryOption func(*ModelRetryMiddleware)

// NewModelRetryMiddleware constructs a ModelRetryMiddleware.
func NewModelRetryMiddleware(opts ...ModelRetryOption) *ModelRetryMiddleware {
	m := &ModelRetryMiddleware{
		MaxRetries: 3,
		BaseDelay:  1 * time.Second,
		MaxDelay:   30 * time.Second,
	}
	for _, o := range opts {
		o(m)
	}
	return m
}

func (m *ModelRetryMiddleware) Name() string { return "ModelRetry" }

// WithModelMaxRetries sets the maximum number of retries.
func WithModelMaxRetries(n int) ModelRetryOption {
	return func(m *ModelRetryMiddleware) { m.MaxRetries = n }
}

// WithModelBaseDelay sets the initial backoff delay.
func WithModelBaseDelay(d time.Duration) ModelRetryOption {
	return func(m *ModelRetryMiddleware) { m.BaseDelay = d }
}

// WithModelMaxDelay sets the maximum backoff delay.
func WithModelMaxDelay(d time.Duration) ModelRetryOption {
	return func(m *ModelRetryMiddleware) { m.MaxDelay = d }
}

func (m *ModelRetryMiddleware) AfterModel(ctx context.Context, gen *schema.Generation) (*schema.Generation, error) {
	m.attempts = 0
	return gen, nil
}

// RetryModelCall wraps an LLM call with exponential backoff retry logic.
// Call this instead of directly invoking the LLM generate method.
func (m *ModelRetryMiddleware) RetryModelCall(ctx context.Context, fn func(ctx context.Context) (*schema.Generation, error)) (*schema.Generation, error) {
	var lastErr error
	for i := 0; i <= m.MaxRetries; i++ {
		if i > 0 {
			delay := time.Duration(math.Min(float64(m.BaseDelay)*math.Pow(2, float64(i-1)), float64(m.MaxDelay)))
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}
		gen, err := fn(ctx)
		if err == nil {
			m.attempts = 0
			return gen, nil
		}
		lastErr = err
		m.attempts = i + 1
	}
	return nil, fmt.Errorf("model retry: after %d attempts: %w", m.MaxRetries+1, lastErr)
}

// ---------------------------------------------------------------------------
// ToolRetryMiddleware
// ---------------------------------------------------------------------------

// ToolRetryMiddleware retries failed tool calls with exponential backoff.
type ToolRetryMiddleware struct {
	NoOpMiddleware
	MaxRetries      int
	BaseDelay       time.Duration
	MaxDelay        time.Duration
	RetryableErrors []string
}

// ToolRetryOption configures a ToolRetryMiddleware.
type ToolRetryOption func(*ToolRetryMiddleware)

// NewToolRetryMiddleware constructs a ToolRetryMiddleware.
func NewToolRetryMiddleware(opts ...ToolRetryOption) *ToolRetryMiddleware {
	m := &ToolRetryMiddleware{
		MaxRetries: 3,
		BaseDelay:  1 * time.Second,
		MaxDelay:   30 * time.Second,
	}
	for _, o := range opts {
		o(m)
	}
	return m
}

func (m *ToolRetryMiddleware) Name() string { return "ToolRetry" }

// WithToolMaxRetries sets the maximum number of retries.
func WithToolMaxRetries(n int) ToolRetryOption {
	return func(m *ToolRetryMiddleware) { m.MaxRetries = n }
}

// WithToolBaseDelay sets the initial backoff delay.
func WithToolBaseDelay(d time.Duration) ToolRetryOption {
	return func(m *ToolRetryMiddleware) { m.BaseDelay = d }
}

// WithToolMaxDelay sets the maximum backoff delay.
func WithToolMaxDelay(d time.Duration) ToolRetryOption {
	return func(m *ToolRetryMiddleware) { m.MaxDelay = d }
}

// WithToolRetryableErrors sets the error substrings that trigger a retry.
func WithToolRetryableErrors(errs ...string) ToolRetryOption {
	return func(m *ToolRetryMiddleware) { m.RetryableErrors = errs }
}

// RetryToolCall wraps a tool execution with exponential backoff retry logic.
func (m *ToolRetryMiddleware) RetryToolCall(ctx context.Context, toolName, input string, fn func(ctx context.Context, toolName, input string) (string, error)) (string, error) {
	var lastErr error
	for i := 0; i <= m.MaxRetries; i++ {
		if i > 0 {
			delay := time.Duration(math.Min(float64(m.BaseDelay)*math.Pow(2, float64(i-1)), float64(m.MaxDelay)))
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-time.After(delay):
			}
		}
		output, err := fn(ctx, toolName, input)
		if err == nil {
			return output, nil
		}
		if len(m.RetryableErrors) > 0 && !matchesAny(err.Error(), m.RetryableErrors) {
			return "", err
		}
		lastErr = err
	}
	return "", fmt.Errorf("tool retry: %s after %d attempts: %w", toolName, m.MaxRetries+1, lastErr)
}

func matchesAny(s string, patterns []string) bool {
	for _, p := range patterns {
		if strings.Contains(s, p) {
			return true
		}
	}
	return false
}
