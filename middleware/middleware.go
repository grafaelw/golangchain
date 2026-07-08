package middleware

import (
	"context"

	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// Middleware interface
// ---------------------------------------------------------------------------

// Middleware hooks into the agent loop at multiple lifecycle points.
// Each method receives the current context, messages, and intermediate steps.
// Returning modified messages or steps allows middleware to transform the agent's state.
type Middleware interface {
	// BeforeModel is called before each LLM call. Return modified messages or an error
	// to short-circuit the agent loop.
	BeforeModel(ctx context.Context, messages []schema.Message, steps []schema.AgentStep) ([]schema.Message, error)

	// AfterModel is called after each LLM response. Return modified generation or nil.
	AfterModel(ctx context.Context, gen *schema.Generation) (*schema.Generation, error)

	// BeforeTool is called before each tool execution. Return modified input or error.
	BeforeTool(ctx context.Context, toolName string, input string) (string, error)

	// AfterTool is called after each tool execution. Return modified observation or error.
	AfterTool(ctx context.Context, toolName string, output string) (string, error)

	// Name returns a human-readable identifier for this middleware.
	Name() string
}

// NoOpMiddleware is an empty implementation for embedding.
type NoOpMiddleware struct{}

func (NoOpMiddleware) BeforeModel(_ context.Context, msgs []schema.Message, _ []schema.AgentStep) ([]schema.Message, error) {
	return msgs, nil
}
func (NoOpMiddleware) AfterModel(_ context.Context, gen *schema.Generation) (*schema.Generation, error) {
	return gen, nil
}
func (NoOpMiddleware) BeforeTool(_ context.Context, _ string, input string) (string, error) {
	return input, nil
}
func (NoOpMiddleware) AfterTool(_ context.Context, _ string, output string) (string, error) {
	return output, nil
}
func (NoOpMiddleware) Name() string { return "NoOp" }
