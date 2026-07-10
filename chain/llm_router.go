package chain

import (
	"context"
	"fmt"
	"strings"

	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/schema"
)

// LLMRouterChain uses an LLM to classify the input and route to one of
// several named sub-chains. Unlike RouterChain (which uses a programmatic
// RouterFunc), this chain asks the LLM itself to pick the best route.
//
//	router := chain.NewLLMRouterChain("Classifier", model,
//	    map[string]chain.Runnable{
//	        "math":    mathChain,
//	        "science": scienceChain,
//	    },
//	    "math: for calculations, equations, math problems",
//	    "science: for physics, chemistry, biology questions",
//	)
//	ans, _ := router.Invoke(ctx, "What is the mass of an electron?")
type LLMRouterChain struct {
	LLM        llm.LLM
	LLMOptions []llm.Option
	Routes     map[string]Runnable
	RouteDescs []string // "name: description" strings for the classifier
	Name       string
}

// NewLLMRouterChain creates an LLM-based router.
func NewLLMRouterChain(name string, model llm.LLM, routes map[string]Runnable, routeDescs ...string) *LLMRouterChain {
	return &LLMRouterChain{
		LLM:        model,
		Routes:     routes,
		RouteDescs: routeDescs,
		Name:       name,
	}
}

func (c *LLMRouterChain) Invoke(ctx context.Context, input any) (any, error) {
	inputStr := fmt.Sprint(input)

	descs := strings.Join(c.RouteDescs, "\n")
	prompt := fmt.Sprintf(`Classify the following input into exactly one of the categories below. Reply with ONLY the category name, nothing else.

Categories:
%s

Input: %s

Category:`, descs, inputStr)

	gen, err := c.LLM.Generate(ctx, []schema.Message{schema.NewHumanMessage(prompt)}, c.LLMOptions...)
	if err != nil {
		return nil, fmt.Errorf("%s: classify: %w", c.Name, err)
	}

	route := strings.TrimSpace(gen.Text)
	r, ok := c.Routes[route]
	if !ok {
		return nil, fmt.Errorf("%s: no route %q (available: %v)", c.Name, route, routeNames(c.Routes))
	}
	return r.Invoke(ctx, input)
}

func (c *LLMRouterChain) Stream(ctx context.Context, input any) (<-chan schema.StreamChunk, error) {
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

func (c *LLMRouterChain) Pipe(next Runnable) Runnable { return &pipeRunnable{first: c, second: next} }
func (c *LLMRouterChain) Batch(ctx context.Context, inputs []any) ([]any, error) {
	return RunBatch(ctx, c, inputs)
}

func routeNames(m map[string]Runnable) []string {
	var names []string
	for k := range m {
		names = append(names, k)
	}
	return names
}
