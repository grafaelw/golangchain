// Package agent implements the golangchain agent framework.
//
// An agent uses an LLM to decide which tools to call and in what order,
// running in a loop until it produces a final answer.
//
// # Agent strategies
//
//   - [ReActAgent] — text-based Thought/Action/Observation loop; works with any LLM
//   - [ToolCallingAgent] — uses the model's native function/tool-calling API;
//     requires a model that supports tool_calls (GPT-4o, Claude 3+, Gemini 1.5+)
//
// # AgentExecutor
//
// [AgentExecutor] owns the run loop, tool dispatch, memory injection, and
// streaming of [AgentEvent] values:
//
//	executor := agent.NewAgentExecutor(
//	    agent.NewToolCallingAgent(model, myTools, systemPrompt),
//	    myTools,
//	    agent.WithMaxIter(10),
//	    agent.WithVerbose(true),
//	)
//
//	// Blocking
//	answer, err := executor.Run(ctx, "What is the population of Amsterdam?")
//
//	// Streaming — yields AgentEvent values in real time
//	for event := range executor.Stream(ctx, "What is 1337 * 42?") {
//	    switch event.Type {
//	    case agent.EventThought:     fmt.Println("Thought:", event.Thought)
//	    case agent.EventToolCall:    fmt.Println("Calling:", event.Action.Tool)
//	    case agent.EventToolResult:  fmt.Println("Result:", event.Observation)
//	    case agent.EventFinalAnswer: fmt.Println("Answer:", event.Answer)
//	    }
//	}
//
// # Custom agents
//
// Implement the [Agent] interface to define your own planning strategy:
//
//	type Agent interface {
//	    Plan(ctx, messages []schema.Message, steps []schema.AgentStep) ([]schema.AgentAction, *schema.AgentFinish, error)
//	    Name() string
//	}
package agent
