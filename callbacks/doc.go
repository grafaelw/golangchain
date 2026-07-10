// Package callbacks defines the Handler interface and CallbackManager used by
// every component in golangchain to emit lifecycle events.
//
// # Handler interface
//
// [Handler] covers 13 lifecycle events across all subsystems:
//
//	// LLM events
//	OnLLMStart(ctx, modelName string, prompts []schema.Message)
//	OnLLMEnd(ctx, modelName string, gen *schema.Generation)
//	OnLLMStream(ctx, modelName string, chunk schema.StreamChunk)
//	// Chain events
//	OnChainStart(ctx, chainName string, inputs map[string]any)
//	OnChainEnd(ctx, chainName string, outputs map[string]any)
//	// Tool events
//	OnToolStart(ctx, toolName, input string)
//	OnToolEnd(ctx, toolName, output string)
//	// Agent events
//	OnAgentAction(ctx context.Context, action schema.AgentAction)
//	OnAgentFinish(ctx context.Context, finish schema.AgentFinish)
//	// Graph events
//	OnGraphNodeStart(ctx, graphName, nodeName string)
//	OnGraphNodeEnd(ctx, graphName, nodeName string)
//	OnGraphCheckpoint(ctx, graphName, threadID string)
//	// Error
//	OnError(ctx context.Context, source string, err error)
//
// # Usage
//
// Embed [NoOpHandler] in your struct and override only the events you care about:
//
//	type MyTracer struct {
//	    callbacks.NoOpHandler
//	}
//	func (t *MyTracer) OnLLMEnd(_ context.Context, model string, gen *schema.Generation) {
//	    fmt.Printf("model=%s tokens=%d\n", model, gen.Usage.TotalTokens)
//	}
//
//	cm := callbacks.NewCallbackManager(&MyTracer{})
//	cm.Add(callbacks.NewLoggingHandler(log.Printf)) // add more at runtime
//
// # Callback propagation
//
// Components receive the CallbackManager in two ways:
//
//  1. Explicit injection — the manager is passed as a constructor option or
//     set via a dedicated method on the component struct.
//
//     This is the primary pattern. Use it when constructing chains, agents,
//     and graphs:
//
//     chain := chain.NewLLMChain(
//     chatPrompt, model, parser,
//     chain.WithChainCallbacks(cm), // explicit
//     )
//
//     executor := agent.NewAgentExecutor(
//     ag, tools,
//     agent.WithCallbackManager(cm), // explicit
//     )
//
//     compiled, _ := g.Compile(
//     graph.WithGraphCallbacks(cm), // explicit
//     )
//
//  2. Context-based retrieval — a component reads the manager from the
//     context.Context via CallbackManagerFromContext(ctx).
//
//     This fallback pattern is used by nested components that are created
//     inside an already-running component (e.g., ToolCallingAgent.Plan and
//     ReActAgent.Plan). The parent injects the manager into context via
//     WithCallbackManager(ctx, cm), and the child retrieves it:
//
//     cm := callbacks.CallbackManagerFromContext(ctx)
//     if cm != nil {
//     cm.OnLLMStart(llmCtx, model, msgs)
//     }
//
//     New components should prefer explicit injection (pattern 1). Use
//     context-based retrieval (pattern 2) only when the component does not
//     have a natural place to store the manager as a field.
package callbacks
