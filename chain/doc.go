// Package chain implements the Runnable interface — golangchain's LCEL
// (LangChain Expression Language) equivalent — and the concrete chain types.
//
// # Runnable
//
// Every chain, LLM wrapper, prompt formatter, and output parser implements [Runnable]:
//
//	type Runnable interface {
//	    Invoke(ctx context.Context, input any) (any, error)
//	    Stream(ctx context.Context, input any) (<-chan StreamChunk, error)
//	    Pipe(next Runnable) Runnable
//	}
//
// Compose chains with [Runnable.Pipe]:
//
//	pipeline := promptRunnable.Pipe(llmRunnable).Pipe(parserRunnable)
//	result, err := pipeline.Invoke(ctx, map[string]any{"question": "..."})
//
// # Chain types
//
//   - [FuncRunnable]     — wraps any func(ctx, any) (any, error)
//   - [LLMRunnable]      — wraps an llm.LLM for use inside a Pipe
//   - [LLMChain]         — prompt → LLM → output parser (the canonical chain)
//   - [SequentialChain]  — threads output of step N as input to step N+1
//   - [MapChain]         — fans input to multiple branches in parallel
//   - [RouterChain]      — picks one sub-chain based on a routing function
//
// # LLMChain example
//
//	c := chain.NewLLMChain(
//	    chatPrompt,
//	    model,
//	    output.AsAny(output.StrOutputParser{}),
//	    chain.WithChainName("MyChain"),
//	)
//	answer, err := c.Invoke(ctx, map[string]any{"question": "What is Go?"})
package chain
