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
package callbacks
