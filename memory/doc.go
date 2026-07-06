// Package memory provides conversation memory implementations for golangchain.
//
// All memory types implement the [Memory] interface:
//
//	type Memory interface {
//	    LoadMemoryVariables(ctx context.Context) (map[string]any, error)
//	    SaveContext(ctx context.Context, humanInput, aiOutput string) error
//	    Messages() []schema.Message
//	    Clear(ctx context.Context) error
//	}
//
// # Memory types
//
//   - [ConversationBufferMemory] — stores the complete history, unbounded
//   - [ConversationWindowMemory] — keeps only the last k turns; older messages are dropped
//   - [ConversationSummaryMemory] — compresses old turns into a running summary via an LLM call
//
// # Usage with LLMChain
//
//	mem := memory.NewConversationWindowMemory(5)
//
//	for _, question := range questions {
//	    vars, _ := mem.LoadMemoryVariables(ctx)
//	    vars["question"] = question
//	    answer, _ := chain.Invoke(ctx, vars)
//	    _ = mem.SaveContext(ctx, question, answer.(string))
//	}
package memory
