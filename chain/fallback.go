package chain

import (
	"context"
	"fmt"

	"github.com/grafaelw/golangchain/schema"
)

// FallbackChain tries runnables in order until one succeeds.
// If all runnables fail, the last error is returned.
//
//	primary := chain.NewLLMChain(...)
//	backup  := chain.NewLLMChain(...)  // cheaper model
//	fb := chain.NewFallbackChain("AskChain", primary, backup)
type FallbackChain struct {
	Name      string
	Runnables []Runnable
}

// NewFallbackChain creates a chain that tries runnables in fallback order.
func NewFallbackChain(name string, primary Runnable, fallbacks ...Runnable) *FallbackChain {
	return &FallbackChain{
		Name:      name,
		Runnables: append([]Runnable{primary}, fallbacks...),
	}
}

func (c *FallbackChain) Invoke(ctx context.Context, input any) (any, error) {
	var lastErr error
	for i, r := range c.Runnables {
		out, err := r.Invoke(ctx, input)
		if err == nil {
			return out, nil
		}
		lastErr = err
		if i < len(c.Runnables)-1 {
			lastErr = fmt.Errorf("%s: fallback[%d] failed (%w), trying next", c.Name, i, err)
		}
	}
	return nil, fmt.Errorf("%s: all %d runnables failed: %w", c.Name, len(c.Runnables), lastErr)
}

func (c *FallbackChain) Stream(ctx context.Context, input any) (<-chan schema.StreamChunk, error) {
	var lastErr error
	for i, r := range c.Runnables {
		ch, err := r.Stream(ctx, input)
		if err == nil {
			return ch, nil
		}
		lastErr = err
		if i < len(c.Runnables)-1 {
			lastErr = fmt.Errorf("%s: fallback[%d] stream failed (%w), trying next", c.Name, i, err)
		}
	}
	return nil, fmt.Errorf("%s: all %d runnables failed to stream: %w", c.Name, len(c.Runnables), lastErr)
}

func (c *FallbackChain) Pipe(next Runnable) Runnable {
	return &pipeRunnable{first: c, second: next}
}

func (c *FallbackChain) Batch(ctx context.Context, inputs []any) ([]any, error) {
	return RunBatch(ctx, c, inputs)
}
