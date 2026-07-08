package middleware

import (
	"context"

	"github.com/grafaelw/golangchain/schema"
)

// chainMiddleware composes multiple middleware into a single Middleware.
type chainMiddleware struct {
	middlewares []Middleware
}

// Chain composes multiple middleware into a single Middleware.
// Each method is called in order; the output of one becomes the input of the next.
// The first error returned short-circuits the remaining middleware in the chain.
func Chain(middlewares ...Middleware) Middleware {
	return &chainMiddleware{middlewares: middlewares}
}

func (c *chainMiddleware) Name() string { return "Chain" }

func (c *chainMiddleware) BeforeModel(ctx context.Context, messages []schema.Message, steps []schema.AgentStep) ([]schema.Message, error) {
	var err error
	for _, mw := range c.middlewares {
		messages, err = mw.BeforeModel(ctx, messages, steps)
		if err != nil {
			return nil, err
		}
	}
	return messages, nil
}

func (c *chainMiddleware) AfterModel(ctx context.Context, gen *schema.Generation) (*schema.Generation, error) {
	var err error
	for _, mw := range c.middlewares {
		gen, err = mw.AfterModel(ctx, gen)
		if err != nil {
			return nil, err
		}
	}
	return gen, nil
}

func (c *chainMiddleware) BeforeTool(ctx context.Context, toolName string, input string) (string, error) {
	var err error
	for _, mw := range c.middlewares {
		input, err = mw.BeforeTool(ctx, toolName, input)
		if err != nil {
			return "", err
		}
	}
	return input, nil
}

func (c *chainMiddleware) AfterTool(ctx context.Context, toolName string, output string) (string, error) {
	var err error
	for _, mw := range c.middlewares {
		output, err = mw.AfterTool(ctx, toolName, output)
		if err != nil {
			return "", err
		}
	}
	return output, nil
}
