// Package chain implements the Runnable interface (golangchain's LCEL
// equivalent) and the concrete chain types: LLMChain, SequentialChain,
// MapChain, and RouterChain.
//
// The composable pipeline API:
//
//	chain := prompt.Pipe(llmRunnable).Pipe(outputParser)
//	result, err := chain.Invoke(ctx, map[string]any{"question": "..."})
package chain

import (
	"context"
	"fmt"
	"sync"

	"github.com/grafaelw/golangchain/callbacks"
	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// Runnable — the core composable interface (LCEL equivalent)
// ---------------------------------------------------------------------------

// Runnable is the fundamental composable unit in golangchain. Every chain,
// LLM wrapper, prompt, and output parser implements Runnable, enabling
// type-safe composition via Pipe.
//
// Input and output types are any to allow heterogeneous pipelines. Type
// assertions are used at the boundaries between components.
type Runnable interface {
	// Invoke executes the runnable with the given input and returns output.
	Invoke(ctx context.Context, input any) (any, error)

	// Stream executes the runnable and yields incremental schema.StreamChunk
	// values. At the LLM layer Text carries tokens; higher-level runnables
	// populate Value for typed pipeline data. The channel is closed after
	// the final chunk (Done==true) or on error.
	Stream(ctx context.Context, input any) (<-chan schema.StreamChunk, error)

	// Pipe composes this Runnable with next, returning a new Runnable that
	// passes the output of this directly as input to next.
	Pipe(next Runnable) Runnable

	// Batch runs Invoke concurrently for each input, returning a slice of
	// outputs in the same order. Errors from individual inputs are propagated
	// as the first non-nil error. Implementations may override for custom
	// concurrency control.
	Batch(ctx context.Context, inputs []any) ([]any, error)
}

// ---------------------------------------------------------------------------
// RunBatch — concurrent batch execution for any Runnable
// ---------------------------------------------------------------------------

// RunBatch runs r.Invoke concurrently for every input and collects results
// in order. Concurrency is bounded by the length of inputs (no artificial cap).
// If any invocation fails the first error is returned.
func RunBatch(ctx context.Context, r Runnable, inputs []any) ([]any, error) {
	if len(inputs) == 0 {
		return nil, nil
	}
	results := make([]any, len(inputs))
	var wg sync.WaitGroup
	var mu sync.Mutex
	var firstErr error
	for i, input := range inputs {
		wg.Add(1)
		go func(idx int, in any) {
			defer wg.Done()
			out, err := r.Invoke(ctx, in)
			mu.Lock()
			defer mu.Unlock()
			if err != nil && firstErr == nil {
				firstErr = err
				return
			}
			results[idx] = out
		}(i, input)
	}
	wg.Wait()
	if firstErr != nil {
		return nil, firstErr
	}
	return results, nil
}

// Batch is the default Batch implementation for types that embed or delegate to Runnable.
func Batch(ctx context.Context, r Runnable, inputs []any) ([]any, error) {
	return RunBatch(ctx, r, inputs)
}

// ---------------------------------------------------------------------------
// pipeRunnable — internal composition of two Runnables
// ---------------------------------------------------------------------------

type pipeRunnable struct {
	first  Runnable
	second Runnable
}

func (p *pipeRunnable) Invoke(ctx context.Context, input any) (any, error) {
	mid, err := p.first.Invoke(ctx, input)
	if err != nil {
		return nil, err
	}
	return p.second.Invoke(ctx, mid)
}

func (p *pipeRunnable) Stream(ctx context.Context, input any) (<-chan schema.StreamChunk, error) {
	// For a pipe, only the last element streams; earlier ones block.
	mid, err := p.first.Invoke(ctx, input)
	if err != nil {
		return nil, err
	}
	return p.second.Stream(ctx, mid)
}

func (p *pipeRunnable) Pipe(next Runnable) Runnable {
	return &pipeRunnable{first: p, second: next}
}

func (p *pipeRunnable) Batch(ctx context.Context, inputs []any) ([]any, error) {
	return RunBatch(ctx, p, inputs)
}

// ---------------------------------------------------------------------------
// FuncRunnable — wraps a plain function as a Runnable
// ---------------------------------------------------------------------------

// FuncRunnable adapts a function with signature func(context.Context, any) (any, error)
// into a Runnable. Use it to insert arbitrary transformation steps into a pipeline.
type FuncRunnable struct {
	fn   func(ctx context.Context, input any) (any, error)
	name string
}

// NewFuncRunnable creates a Runnable from a function.
func NewFuncRunnable(name string, fn func(ctx context.Context, input any) (any, error)) *FuncRunnable {
	return &FuncRunnable{fn: fn, name: name}
}

func (f *FuncRunnable) Invoke(ctx context.Context, input any) (any, error) {
	return f.fn(ctx, input)
}

func (f *FuncRunnable) Stream(ctx context.Context, input any) (<-chan schema.StreamChunk, error) {
	ch := make(chan schema.StreamChunk, 1)
	go func() {
		defer close(ch)
		out, err := f.fn(ctx, input)
		if err != nil {
			ch <- schema.StreamChunk{Err: err}
			return
		}
		ch <- schema.StreamChunk{Value: out, Done: true}
	}()
	return ch, nil
}

func (f *FuncRunnable) Pipe(next Runnable) Runnable {
	return &pipeRunnable{first: f, second: next}
}

func (f *FuncRunnable) Batch(ctx context.Context, inputs []any) ([]any, error) {
	return RunBatch(ctx, f, inputs)
}

// ---------------------------------------------------------------------------
// LLMRunnable — wraps an llm.LLM as a Runnable
// ---------------------------------------------------------------------------

// LLMRunnable adapts an llm.LLM so it can participate in a Pipe chain.
// Input must be []schema.Message; output is *schema.Generation.
type LLMRunnable struct {
	LLM       llm.LLM
	Opts      []llm.Option
	Callbacks *callbacks.CallbackManager
}

// NewLLMRunnable wraps an LLM with optional call options and callbacks.
func NewLLMRunnable(model llm.LLM, opts ...llm.Option) *LLMRunnable {
	return &LLMRunnable{LLM: model, Opts: opts}
}

// WithCallbacks attaches a CallbackManager (returns self for chaining).
func (r *LLMRunnable) WithCallbacks(cm *callbacks.CallbackManager) *LLMRunnable {
	r.Callbacks = cm
	return r
}

func (r *LLMRunnable) Invoke(ctx context.Context, input any) (any, error) {
	msgs, err := toMessages(input)
	if err != nil {
		return nil, fmt.Errorf("LLMRunnable: %w", err)
	}
	llmCtx := ctx
	if r.Callbacks != nil {
		llmCtx = callbacks.WithRunID(ctx, callbacks.NewRunID())
		r.Callbacks.OnLLMStart(llmCtx, r.LLM.ModelName(), msgs)
	}
	gen, err := r.LLM.Generate(llmCtx, msgs, r.Opts...)
	if err != nil {
		if r.Callbacks != nil {
			r.Callbacks.OnError(llmCtx, "LLMRunnable", err)
		}
		return nil, fmt.Errorf("LLMRunnable: %w", err)
	}
	if r.Callbacks != nil {
		r.Callbacks.OnLLMEnd(llmCtx, r.LLM.ModelName(), gen)
	}
	return gen, nil
}

func (r *LLMRunnable) Stream(ctx context.Context, input any) (<-chan schema.StreamChunk, error) {
	msgs, err := toMessages(input)
	if err != nil {
		return nil, fmt.Errorf("LLMRunnable: %w", err)
	}
	llmCtx := ctx
	if r.Callbacks != nil {
		llmCtx = callbacks.WithRunID(ctx, callbacks.NewRunID())
		r.Callbacks.OnLLMStart(llmCtx, r.LLM.ModelName(), msgs)
	}
	llmCh, err := r.LLM.Stream(llmCtx, msgs, r.Opts...)
	if err != nil {
		if r.Callbacks != nil {
			r.Callbacks.OnError(llmCtx, "LLMRunnable", err)
		}
		return nil, fmt.Errorf("LLMRunnable: stream: %w", err)
	}

	out := make(chan schema.StreamChunk, 32)
	go func() {
		defer close(out)
		for chunk := range llmCh {
			if r.Callbacks != nil && chunk.Text != "" {
				r.Callbacks.OnLLMStream(llmCtx, r.LLM.ModelName(), chunk)
			}
			if chunk.Err != nil {
				if r.Callbacks != nil {
					r.Callbacks.OnError(llmCtx, "LLMRunnable", chunk.Err)
				}
				out <- chunk
				return
			}
			if chunk.Done {
				if r.Callbacks != nil {
					r.Callbacks.OnLLMEnd(llmCtx, r.LLM.ModelName(), &schema.Generation{})
				}
			}
			// Propagate LLM chunks directly; set Value from Text for
			// downstream runnables that consume Value.
			chunk.Value = chunk.Text
			out <- chunk
		}
	}()
	return out, nil
}

func (r *LLMRunnable) Pipe(next Runnable) Runnable {
	return &pipeRunnable{first: r, second: next}
}

func (r *LLMRunnable) Batch(ctx context.Context, inputs []any) ([]any, error) {
	return RunBatch(ctx, r, inputs)
}

// ---------------------------------------------------------------------------
// LLMChain — prompt → LLM → output parser
// ---------------------------------------------------------------------------

// Formatter can format a map of variables into []schema.Message.
// Both *prompt.PromptTemplate (via adapter) and *prompt.ChatPromptTemplate
// implement this when wrapped with the adapters below.
type Formatter interface {
	FormatMessages(vars map[string]any) ([]schema.Message, error)
}

// LLMChain is the canonical chain: format a prompt, call an LLM, parse output.
// It implements Runnable, so it can be composed with Pipe.
//
//	c := chain.NewLLMChain(chatPrompt, myLLM, output.StrOutputParser{})
//	result, err := c.Invoke(ctx, map[string]any{"question": "What is Go?"})
type LLMChain struct {
	name      string
	formatter Formatter
	llm       llm.LLM
	llmOpts   []llm.Option
	parser    interface{ Parse(string) (any, error) }
	callbacks *callbacks.CallbackManager
}

// LLMChainOption configures an LLMChain.
type LLMChainOption func(*LLMChain)

// WithChainName sets the chain name (used in callbacks).
func WithChainName(name string) LLMChainOption {
	return func(c *LLMChain) { c.name = name }
}

// WithLLMOptions adds per-call LLM options.
func WithLLMOptions(opts ...llm.Option) LLMChainOption {
	return func(c *LLMChain) { c.llmOpts = append(c.llmOpts, opts...) }
}

// WithChainCallbacks attaches a CallbackManager.
func WithChainCallbacks(cm *callbacks.CallbackManager) LLMChainOption {
	return func(c *LLMChain) { c.callbacks = cm }
}

// NewLLMChain constructs an LLMChain.
// parser must implement Parse(string) (any, error) — use output.StrOutputParser{},
// output.JSONOutputParser{}, or output.NewStructOutputParser[T]() etc.
func NewLLMChain(formatter Formatter, model llm.LLM, parser interface{ Parse(string) (any, error) }, opts ...LLMChainOption) *LLMChain {
	c := &LLMChain{
		name:      "LLMChain",
		formatter: formatter,
		llm:       model,
		parser:    parser,
	}
	for _, o := range opts {
		o(c)
	}
	return c
}

// Invoke executes prompt → LLM → parser and returns the parsed output.
// input must be map[string]any containing the template variables.
func (c *LLMChain) Invoke(ctx context.Context, input any) (any, error) {
	vars, err := toVars(input)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", c.name, err)
	}

	// Inject a chain-level run ID so child spans can reference it as parent.
	chainCtx := ctx
	if c.callbacks != nil {
		chainCtx = callbacks.WithRunID(ctx, callbacks.NewRunID())
		c.callbacks.OnChainStart(chainCtx, c.name, vars)
	}

	msgs, err := c.formatter.FormatMessages(vars)
	if err != nil {
		return nil, fmt.Errorf("%s: format: %w", c.name, err)
	}

	// Inject an LLM-level run ID nested under the chain run.
	llmCtx := chainCtx
	if c.callbacks != nil {
		llmCtx = callbacks.WithRunID(chainCtx, callbacks.NewRunID())
		c.callbacks.OnLLMStart(llmCtx, c.llm.ModelName(), msgs)
	}
	gen, err := c.llm.Generate(llmCtx, msgs, c.llmOpts...)
	if err != nil {
		if c.callbacks != nil {
			c.callbacks.OnError(llmCtx, c.name, err)
		}
		return nil, fmt.Errorf("%s: llm: %w", c.name, err)
	}
	if c.callbacks != nil {
		c.callbacks.OnLLMEnd(llmCtx, c.llm.ModelName(), gen)
	}

	parsed, err := c.parser.Parse(gen.Text)
	if err != nil {
		return nil, fmt.Errorf("%s: parse: %w", c.name, err)
	}

	if c.callbacks != nil {
		c.callbacks.OnChainEnd(chainCtx, c.name, map[string]any{"output": parsed})
	}
	return parsed, nil
}

// Stream runs the prompt and LLM in streaming mode; the parser is applied
// to the accumulated full text at the end.
func (c *LLMChain) Stream(ctx context.Context, input any) (<-chan schema.StreamChunk, error) {
	vars, err := toVars(input)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", c.name, err)
	}

	chainCtx := ctx
	if c.callbacks != nil {
		chainCtx = callbacks.WithRunID(ctx, callbacks.NewRunID())
		c.callbacks.OnChainStart(chainCtx, c.name, vars)
	}

	msgs, err := c.formatter.FormatMessages(vars)
	if err != nil {
		return nil, fmt.Errorf("%s: format: %w", c.name, err)
	}

	llmCtx := chainCtx
	if c.callbacks != nil {
		llmCtx = callbacks.WithRunID(chainCtx, callbacks.NewRunID())
		c.callbacks.OnLLMStart(llmCtx, c.llm.ModelName(), msgs)
	}

	llmCh, err := c.llm.Stream(llmCtx, msgs, c.llmOpts...)
	if err != nil {
		if c.callbacks != nil {
			c.callbacks.OnError(llmCtx, c.name, err)
		}
		return nil, fmt.Errorf("%s: stream: %w", c.name, err)
	}

	out := make(chan schema.StreamChunk, 32)
	go func() {
		defer close(out)
		var full string
		for chunk := range llmCh {
			if chunk.Err != nil {
				if c.callbacks != nil {
					c.callbacks.OnError(llmCtx, c.name, chunk.Err)
				}
				out <- chunk
				return
			}
			if c.callbacks != nil && chunk.Text != "" {
				c.callbacks.OnLLMStream(llmCtx, c.llm.ModelName(), chunk)
			}
			full += chunk.Text
			// Forward text; set Value for downstream consumers.
			chunk.Value = chunk.Text
			out <- chunk
			if chunk.Done {
				if c.callbacks != nil {
					c.callbacks.OnLLMEnd(llmCtx, c.llm.ModelName(), &schema.Generation{Text: full})
				}
				parsed, err := c.parser.Parse(full)
				if err != nil {
					out <- schema.StreamChunk{Err: err}
					return
				}
				if c.callbacks != nil {
					c.callbacks.OnChainEnd(chainCtx, c.name, map[string]any{"output": parsed})
				}
				out <- schema.StreamChunk{Value: parsed, Done: true}
				return
			}
		}
	}()
	return out, nil
}

func (c *LLMChain) Pipe(next Runnable) Runnable {
	return &pipeRunnable{first: c, second: next}
}

func (c *LLMChain) Batch(ctx context.Context, inputs []any) ([]any, error) {
	return RunBatch(ctx, c, inputs)
}

// ---------------------------------------------------------------------------
// SequentialChain — runs Runnables in order, threading output→input
// ---------------------------------------------------------------------------

// SequentialChain runs a list of Runnables in sequence. The output of each
// step is passed as the input to the next.
type SequentialChain struct {
	name      string
	steps     []Runnable
	callbacks *callbacks.CallbackManager
}

// NewSequentialChain constructs a SequentialChain from an ordered step list.
func NewSequentialChain(name string, steps ...Runnable) *SequentialChain {
	return &SequentialChain{name: name, steps: steps}
}

func (s *SequentialChain) Invoke(ctx context.Context, input any) (any, error) {
	if s.callbacks != nil {
		if vars, _ := input.(map[string]any); vars != nil {
			s.callbacks.OnChainStart(ctx, s.name, vars)
		}
	}
	current := input
	for i, step := range s.steps {
		out, err := step.Invoke(ctx, current)
		if err != nil {
			return nil, fmt.Errorf("%s: step %d: %w", s.name, i, err)
		}
		current = out
	}
	if s.callbacks != nil {
		s.callbacks.OnChainEnd(ctx, s.name, map[string]any{"output": current})
	}
	return current, nil
}

func (s *SequentialChain) Stream(ctx context.Context, input any) (<-chan schema.StreamChunk, error) {
	// Only the final step streams; all prior steps run to completion.
	current := input
	for i, step := range s.steps[:len(s.steps)-1] {
		out, err := step.Invoke(ctx, current)
		if err != nil {
			return nil, fmt.Errorf("%s: step %d: %w", s.name, i, err)
		}
		current = out
	}
	return s.steps[len(s.steps)-1].Stream(ctx, current)
}

func (s *SequentialChain) Pipe(next Runnable) Runnable {
	return &pipeRunnable{first: s, second: next}
}

func (s *SequentialChain) Batch(ctx context.Context, inputs []any) ([]any, error) {
	return RunBatch(ctx, s, inputs)
}

// ---------------------------------------------------------------------------
// MapChain — runs multiple Runnables on the same input concurrently
// ---------------------------------------------------------------------------

// MapChain fans input out to multiple Runnables in parallel and collects
// their outputs into a map[string]any keyed by the supplied names.
type MapChain struct {
	name     string
	branches map[string]Runnable
}

// NewMapChain constructs a MapChain with named branches.
//
//	m := chain.NewMapChain("parallel", map[string]chain.Runnable{
//	    "summary": summaryChain,
//	    "bullets": bulletChain,
//	})
func NewMapChain(name string, branches map[string]Runnable) *MapChain {
	return &MapChain{name: name, branches: branches}
}

func (m *MapChain) Invoke(ctx context.Context, input any) (any, error) {
	type result struct {
		key string
		val any
		err error
	}

	ch := make(chan result, len(m.branches))
	var wg sync.WaitGroup

	for key, r := range m.branches {
		wg.Add(1)
		go func(k string, run Runnable) {
			defer wg.Done()
			out, err := run.Invoke(ctx, input)
			ch <- result{key: k, val: out, err: err}
		}(key, r)
	}

	go func() { wg.Wait(); close(ch) }()

	out := make(map[string]any, len(m.branches))
	for res := range ch {
		if res.err != nil {
			return nil, fmt.Errorf("%s: branch %q: %w", m.name, res.key, res.err)
		}
		out[res.key] = res.val
	}
	return out, nil
}

func (m *MapChain) Stream(ctx context.Context, input any) (<-chan schema.StreamChunk, error) {
	return nil, fmt.Errorf("%s: streaming not supported for parallel MapChain; use Invoke instead", m.name)
}

func (m *MapChain) Pipe(next Runnable) Runnable {
	return &pipeRunnable{first: m, second: next}
}

func (m *MapChain) Batch(ctx context.Context, inputs []any) ([]any, error) {
	return RunBatch(ctx, m, inputs)
}

// ---------------------------------------------------------------------------
// RouterChain — routes to one of several sub-chains based on a selector
// ---------------------------------------------------------------------------

// RouterFunc is called with the Invoke input to select which sub-chain to run.
type RouterFunc func(ctx context.Context, input any) (string, error)

// RouterChain selects one Runnable from a registry based on RouterFunc output.
type RouterChain struct {
	name     string
	router   RouterFunc
	chains   map[string]Runnable
	fallback Runnable // used when router returns an unknown key
}

// NewRouterChain constructs a RouterChain.
func NewRouterChain(name string, router RouterFunc, chains map[string]Runnable, fallback Runnable) *RouterChain {
	return &RouterChain{name: name, router: router, chains: chains, fallback: fallback}
}

func (r *RouterChain) Invoke(ctx context.Context, input any) (any, error) {
	key, err := r.router(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("%s: router: %w", r.name, err)
	}
	target, ok := r.chains[key]
	if !ok {
		if r.fallback != nil {
			return r.fallback.Invoke(ctx, input)
		}
		return nil, fmt.Errorf("%s: unknown route %q and no fallback set", r.name, key)
	}
	return target.Invoke(ctx, input)
}

func (r *RouterChain) Stream(ctx context.Context, input any) (<-chan schema.StreamChunk, error) {
	key, err := r.router(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("%s: router: %w", r.name, err)
	}
	target, ok := r.chains[key]
	if !ok {
		if r.fallback != nil {
			return r.fallback.Stream(ctx, input)
		}
		return nil, fmt.Errorf("%s: unknown route %q", r.name, key)
	}
	return target.Stream(ctx, input)
}

func (r *RouterChain) Pipe(next Runnable) Runnable {
	return &pipeRunnable{first: r, second: next}
}

func (r *RouterChain) Batch(ctx context.Context, inputs []any) ([]any, error) {
	return RunBatch(ctx, r, inputs)
}

// ---------------------------------------------------------------------------
// Adapters — make prompt types satisfy Formatter
// ---------------------------------------------------------------------------

// PromptTemplateFormatter adapts *prompt.PromptTemplate to Formatter by
// wrapping the plain string output in a single human message.
// Use chain.NewPromptTemplateFormatter(myPromptTemplate).
type PromptTemplateFormatter struct {
	template interface {
		Format(map[string]any) (string, error)
	}
}

// NewPromptTemplateFormatter wraps a *prompt.PromptTemplate as a Formatter.
func NewPromptTemplateFormatter(t interface {
	Format(map[string]any) (string, error)
}) *PromptTemplateFormatter {
	return &PromptTemplateFormatter{template: t}
}

func (p *PromptTemplateFormatter) FormatMessages(vars map[string]any) ([]schema.Message, error) {
	text, err := p.template.Format(vars)
	if err != nil {
		return nil, err
	}
	return []schema.Message{schema.NewHumanMessage(text)}, nil
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

func toMessages(input any) ([]schema.Message, error) {
	switch v := input.(type) {
	case []schema.Message:
		return v, nil
	case *schema.Generation:
		return []schema.Message{v.Message}, nil
	case schema.Message:
		return []schema.Message{v}, nil
	case string:
		return []schema.Message{schema.NewHumanMessage(v)}, nil
	default:
		return nil, fmt.Errorf("expected []schema.Message, got %T", input)
	}
}

func toVars(input any) (map[string]any, error) {
	switch v := input.(type) {
	case map[string]any:
		return v, nil
	case string:
		return map[string]any{"input": v}, nil
	default:
		return nil, fmt.Errorf("expected map[string]any input, got %T", input)
	}
}
