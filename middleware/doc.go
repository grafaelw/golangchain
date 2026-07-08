// Package middleware provides composable middleware for the agent loop.
//
// Middleware intercepts and transforms the agent's lifecycle at key points:
// before and after each LLM call, and before and after each tool execution.
// Individual middleware implementations handle retry logic, summarization,
// human-in-the-loop approval, content filtering, PII detection, todo-list
// injection, and more.
//
// # Middleware interface
//
// Implement [Middleware] to hook into the agent loop:
//
//	type Middleware interface {
//	    BeforeModel(ctx context.Context, messages []schema.Message, steps []schema.AgentStep) ([]schema.Message, error)
//	    AfterModel(ctx context.Context, gen *schema.Generation) (*schema.Generation, error)
//	    BeforeTool(ctx context.Context, toolName, input string) (string, error)
//	    AfterTool(ctx context.Context, toolName, output string) (string, error)
//	    Name() string
//	}
//
// Embed [NoOpMiddleware] in your struct and override only the methods you need:
//
//	type MyMiddleware struct {
//	    middleware.NoOpMiddleware
//	}
//	func (m *MyMiddleware) Name() string { return "MyMiddleware" }
//	func (m *MyMiddleware) BeforeModel(ctx context.Context, msgs []schema.Message, steps []schema.AgentStep) ([]schema.Message, error) {
//	    // transform msgs...
//	    return msgs, nil
//	}
//
// # Built-in middleware
//
//   - [ModelRetryMiddleware]   — retries failed LLM calls with exponential backoff
//   - [ToolRetryMiddleware]    — retries failed tool calls with exponential backoff
//   - [SummarizationMiddleware] — compresses conversation history when it exceeds a token threshold
//   - [TodoListMiddleware]     — injects a todo list into the system prompt
//   - [HumanInTheLoopMiddleware] — requires human approval for tool calls matching configured patterns
//   - [ContentFilterMiddleware]  — filters inputs/outputs through an allow function
//   - [PIIMiddleware]            — detects and masks PII in tool inputs and outputs
//
// # Composing middleware
//
// Use [Chain] to compose multiple middleware into a single [Middleware]:
//
//	mw := middleware.Chain(
//	    middleware.NewModelRetryMiddleware(middleware.WithModelMaxRetries(5)),
//	    middleware.NewSummarizationMiddleware(model),
//	    middleware.NewTodoListMiddleware(),
//	    middleware.NewHumanInTheLoopMiddleware(approveFn),
//	)
//	executor := agent.NewAgentExecutor(agent, tools, agent.WithMiddleware(mw))
package middleware
