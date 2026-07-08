package agent

import (
	"fmt"

	"github.com/grafaelw/golangchain/callbacks"
	"github.com/grafaelw/golangchain/graph"
	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/memory"
	"github.com/grafaelw/golangchain/middleware"
	"github.com/grafaelw/golangchain/schema"
	"github.com/grafaelw/golangchain/tools"
)

// AgentOption configures CreateAgent.
type AgentOption func(*agentConfig)

type agentConfig struct {
	model          llm.LLM
	tools          []tools.Tool
	systemPrompt   string
	middleware     []middleware.Middleware
	checkpointer   any
	responseFormat any
	maxIter        int
	verbose        bool
	name           string
	memory         memory.Memory
	callbacks      *callbacks.CallbackManager
}

// WithModel sets the LLM provider for the agent.
func WithModel(m llm.LLM) AgentOption {
	return func(c *agentConfig) { c.model = m }
}

// WithTools registers tools available to the agent.
func WithTools(t ...tools.Tool) AgentOption {
	return func(c *agentConfig) { c.tools = t }
}

// WithSystemPrompt sets the system prompt injected before every LLM call.
func WithSystemPrompt(p string) AgentOption {
	return func(c *agentConfig) { c.systemPrompt = p }
}

// WithAgentMiddleware attaches middleware to the agent loop.
func WithAgentMiddleware(mw ...middleware.Middleware) AgentOption {
	return func(c *agentConfig) { c.middleware = append(c.middleware, mw...) }
}

// WithAgentCheckpointer sets a checkpointer for persisting agent state across
// runs. Accepted values: *graph.MemoryCheckpointer[[]schema.Message],
// *graph.FileCheckpointer[[]schema.Message], *graph.JSONLCheckpointer[[]schema.Message].
// The checkpointer is converted to an AgentCheckpointer internally.
func WithAgentCheckpointer(cp any) AgentOption {
	return func(c *agentConfig) { c.checkpointer = cp }
}

// WithAgentResponseFormat requests structured JSON output. The value should be
// the zero value of the desired response struct type (e.g. new(MyReply)).
// The final answer is validated as JSON and returned as a canonical JSON string.
func WithAgentResponseFormat(v any) AgentOption {
	return func(c *agentConfig) { c.responseFormat = v }
}

// WithAgentMaxIter sets the maximum number of plan/execute iterations.
func WithAgentMaxIter(n int) AgentOption {
	return func(c *agentConfig) { c.maxIter = n }
}

// WithAgentVerbose enables verbose logging.
func WithAgentVerbose(v bool) AgentOption {
	return func(c *agentConfig) { c.verbose = v }
}

// WithAgentName sets a human-readable name for the agent.
func WithAgentName(n string) AgentOption {
	return func(c *agentConfig) { c.name = n }
}

// WithAgentMemory attaches a memory provider for conversation history.
func WithAgentMemory(m memory.Memory) AgentOption {
	return func(c *agentConfig) { c.memory = m }
}

// WithAgentCallbacks attaches a CallbackManager.
func WithAgentCallbacks(cm *callbacks.CallbackManager) AgentOption {
	return func(c *agentConfig) { c.callbacks = cm }
}

// CreateAgent builds a ToolCallingAgent wrapped in an AgentExecutor with
// middleware, checkpointing, and structured-output support applied.
func CreateAgent(opts ...AgentOption) (*AgentExecutor, error) {
	cfg := &agentConfig{
		maxIter: 10,
		name:    "ToolCallingAgent",
	}
	for _, o := range opts {
		o(cfg)
	}

	if cfg.model == nil {
		return nil, fmt.Errorf("agent: CreateAgent: model is required")
	}

	ag := NewToolCallingAgent(cfg.model, cfg.tools, cfg.systemPrompt)

	var execOpts []ExecutorOption
	if cfg.memory != nil {
		execOpts = append(execOpts, WithMemory(cfg.memory))
	}
	if cfg.callbacks != nil {
		execOpts = append(execOpts, WithCallbackManager(cfg.callbacks))
	}
	if cfg.maxIter > 0 {
		execOpts = append(execOpts, WithMaxIter(cfg.maxIter))
	}
	execOpts = append(execOpts, WithVerbose(cfg.verbose))

	var agentCP AgentCheckpointer
	if cfg.checkpointer != nil {
		switch cp := cfg.checkpointer.(type) {
		case *graph.MemoryCheckpointer[[]schema.Message]:
			agentCP = AsAgentCheckpointer(cp)
		case *graph.FileCheckpointer[[]schema.Message]:
			agentCP = AsAgentCheckpointer(cp)
		case *graph.JSONLCheckpointer[[]schema.Message]:
			agentCP = AsAgentCheckpointer(cp)
		case graph.Checkpointer[[]schema.Message]:
			agentCP = AsAgentCheckpointer(cp)
		default:
			return nil, fmt.Errorf("agent: CreateAgent: unsupported checkpointer type %T; use graph.MemoryCheckpointer, FileCheckpointer, or JSONLCheckpointer with []schema.Message", cfg.checkpointer)
		}
	}

	if len(cfg.middleware) > 0 {
		execOpts = append(execOpts, WithMiddleware(cfg.middleware...))
	}
	if cfg.responseFormat != nil {
		execOpts = append(execOpts, WithResponseFormat(cfg.responseFormat))
	}

	executor := NewAgentExecutor(ag, cfg.tools, execOpts...)
	executor.Checkpointer = agentCP
	executor.ResponseFormat = cfg.responseFormat

	return executor, nil
}
