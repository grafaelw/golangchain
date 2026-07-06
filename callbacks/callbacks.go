// Package callbacks defines the Handler interface and CallbackManager used by
// every component in golangchain to emit lifecycle events.
// Users plug in custom handlers for logging, tracing, or OpenTelemetry export.
package callbacks

import (
	"context"
	"sync"

	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// Handler interface
// ---------------------------------------------------------------------------

// Handler receives lifecycle events from chains, LLMs, tools, agents, and
// the graph engine. Implement this interface to add logging, tracing, or
// metrics to any golangchain component.
//
// All methods receive a context so handlers can respect cancellation and carry
// OpenTelemetry spans or other context values.
type Handler interface {
	// LLM events
	OnLLMStart(ctx context.Context, modelName string, prompts []schema.Message)
	OnLLMEnd(ctx context.Context, modelName string, gen *schema.Generation)
	OnLLMStream(ctx context.Context, modelName string, chunk schema.StreamChunk)

	// Chain events
	OnChainStart(ctx context.Context, chainName string, inputs map[string]any)
	OnChainEnd(ctx context.Context, chainName string, outputs map[string]any)

	// Tool events
	OnToolStart(ctx context.Context, toolName string, input string)
	OnToolEnd(ctx context.Context, toolName string, output string)

	// Agent events
	OnAgentAction(ctx context.Context, action schema.AgentAction)
	OnAgentFinish(ctx context.Context, finish schema.AgentFinish)

	// Graph events
	OnGraphNodeStart(ctx context.Context, graphName, nodeName string)
	OnGraphNodeEnd(ctx context.Context, graphName, nodeName string)
	OnGraphCheckpoint(ctx context.Context, graphName, threadID string)

	// Error — called when any component encounters an unrecoverable error.
	OnError(ctx context.Context, source string, err error)
}

// ---------------------------------------------------------------------------
// NoOpHandler — embed to satisfy Handler without implementing every method
// ---------------------------------------------------------------------------

// NoOpHandler provides empty implementations of every Handler method.
// Embed it in your own struct and override only the methods you need.
type NoOpHandler struct{}

func (NoOpHandler) OnLLMStart(ctx context.Context, modelName string, prompts []schema.Message) {}
func (NoOpHandler) OnLLMEnd(ctx context.Context, modelName string, gen *schema.Generation)     {}
func (NoOpHandler) OnLLMStream(ctx context.Context, modelName string, chunk schema.StreamChunk) {}
func (NoOpHandler) OnChainStart(ctx context.Context, chainName string, inputs map[string]any)   {}
func (NoOpHandler) OnChainEnd(ctx context.Context, chainName string, outputs map[string]any)    {}
func (NoOpHandler) OnToolStart(ctx context.Context, toolName string, input string)              {}
func (NoOpHandler) OnToolEnd(ctx context.Context, toolName string, output string)               {}
func (NoOpHandler) OnAgentAction(ctx context.Context, action schema.AgentAction)                {}
func (NoOpHandler) OnAgentFinish(ctx context.Context, finish schema.AgentFinish)                {}
func (NoOpHandler) OnGraphNodeStart(ctx context.Context, graphName, nodeName string)            {}
func (NoOpHandler) OnGraphNodeEnd(ctx context.Context, graphName, nodeName string)              {}
func (NoOpHandler) OnGraphCheckpoint(ctx context.Context, graphName, threadID string)           {}
func (NoOpHandler) OnError(ctx context.Context, source string, err error)                       {}

// ---------------------------------------------------------------------------
// CallbackManager — fan-out to multiple handlers
// ---------------------------------------------------------------------------

// CallbackManager multiplexes events to a list of Handler implementations.
// It is safe for concurrent use.
type CallbackManager struct {
	mu       sync.RWMutex
	handlers []Handler
}

// NewCallbackManager constructs a CallbackManager pre-loaded with handlers.
func NewCallbackManager(handlers ...Handler) *CallbackManager {
	return &CallbackManager{handlers: handlers}
}

// Add appends a handler at runtime (goroutine-safe).
func (m *CallbackManager) Add(h Handler) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.handlers = append(m.handlers, h)
}

func (m *CallbackManager) each(fn func(Handler)) {
	m.mu.RLock()
	hs := make([]Handler, len(m.handlers))
	copy(hs, m.handlers)
	m.mu.RUnlock()
	for _, h := range hs {
		fn(h)
	}
}

func (m *CallbackManager) OnLLMStart(ctx context.Context, modelName string, prompts []schema.Message) {
	m.each(func(h Handler) { h.OnLLMStart(ctx, modelName, prompts) })
}
func (m *CallbackManager) OnLLMEnd(ctx context.Context, modelName string, gen *schema.Generation) {
	m.each(func(h Handler) { h.OnLLMEnd(ctx, modelName, gen) })
}
func (m *CallbackManager) OnLLMStream(ctx context.Context, modelName string, chunk schema.StreamChunk) {
	m.each(func(h Handler) { h.OnLLMStream(ctx, modelName, chunk) })
}
func (m *CallbackManager) OnChainStart(ctx context.Context, chainName string, inputs map[string]any) {
	m.each(func(h Handler) { h.OnChainStart(ctx, chainName, inputs) })
}
func (m *CallbackManager) OnChainEnd(ctx context.Context, chainName string, outputs map[string]any) {
	m.each(func(h Handler) { h.OnChainEnd(ctx, chainName, outputs) })
}
func (m *CallbackManager) OnToolStart(ctx context.Context, toolName string, input string) {
	m.each(func(h Handler) { h.OnToolStart(ctx, toolName, input) })
}
func (m *CallbackManager) OnToolEnd(ctx context.Context, toolName string, output string) {
	m.each(func(h Handler) { h.OnToolEnd(ctx, toolName, output) })
}
func (m *CallbackManager) OnAgentAction(ctx context.Context, action schema.AgentAction) {
	m.each(func(h Handler) { h.OnAgentAction(ctx, action) })
}
func (m *CallbackManager) OnAgentFinish(ctx context.Context, finish schema.AgentFinish) {
	m.each(func(h Handler) { h.OnAgentFinish(ctx, finish) })
}
func (m *CallbackManager) OnGraphNodeStart(ctx context.Context, graphName, nodeName string) {
	m.each(func(h Handler) { h.OnGraphNodeStart(ctx, graphName, nodeName) })
}
func (m *CallbackManager) OnGraphNodeEnd(ctx context.Context, graphName, nodeName string) {
	m.each(func(h Handler) { h.OnGraphNodeEnd(ctx, graphName, nodeName) })
}
func (m *CallbackManager) OnGraphCheckpoint(ctx context.Context, graphName, threadID string) {
	m.each(func(h Handler) { h.OnGraphCheckpoint(ctx, graphName, threadID) })
}
func (m *CallbackManager) OnError(ctx context.Context, source string, err error) {
	m.each(func(h Handler) { h.OnError(ctx, source, err) })
}

// ---------------------------------------------------------------------------
// LoggingHandler — ready-made handler that prints to any io.Writer
// ---------------------------------------------------------------------------

// LoggingHandler is a built-in Handler that logs every event to an io.Writer
// (default: os.Stderr). It is useful during development.
type LoggingHandler struct {
	NoOpHandler
	logFn func(format string, args ...any)
}

// NewLoggingHandler returns a LoggingHandler that writes to the supplied log
// function (e.g. log.Printf or slog.Info-style wrapper).
func NewLoggingHandler(logFn func(format string, args ...any)) *LoggingHandler {
	return &LoggingHandler{logFn: logFn}
}

func (l *LoggingHandler) OnLLMStart(_ context.Context, model string, msgs []schema.Message) {
	l.logFn("[LLM] start model=%s messages=%d", model, len(msgs))
}
func (l *LoggingHandler) OnLLMEnd(_ context.Context, model string, gen *schema.Generation) {
	l.logFn("[LLM] end model=%s stop=%s prompt_tokens=%d completion_tokens=%d",
		model, gen.StopReason, gen.Usage.PromptTokens, gen.Usage.CompletionTokens)
}
func (l *LoggingHandler) OnChainStart(_ context.Context, name string, _ map[string]any) {
	l.logFn("[Chain] start name=%s", name)
}
func (l *LoggingHandler) OnChainEnd(_ context.Context, name string, _ map[string]any) {
	l.logFn("[Chain] end name=%s", name)
}
func (l *LoggingHandler) OnToolStart(_ context.Context, name, input string) {
	l.logFn("[Tool] start name=%s input=%q", name, input)
}
func (l *LoggingHandler) OnToolEnd(_ context.Context, name, output string) {
	l.logFn("[Tool] end name=%s output=%q", name, output)
}
func (l *LoggingHandler) OnAgentAction(_ context.Context, a schema.AgentAction) {
	l.logFn("[Agent] action tool=%s input=%q", a.Tool, a.ToolInput)
}
func (l *LoggingHandler) OnAgentFinish(_ context.Context, f schema.AgentFinish) {
	l.logFn("[Agent] finish output=%q", f.Output)
}
func (l *LoggingHandler) OnGraphNodeStart(_ context.Context, graph, node string) {
	l.logFn("[Graph] node start graph=%s node=%s", graph, node)
}
func (l *LoggingHandler) OnGraphNodeEnd(_ context.Context, graph, node string) {
	l.logFn("[Graph] node end graph=%s node=%s", graph, node)
}
func (l *LoggingHandler) OnError(_ context.Context, source string, err error) {
	l.logFn("[ERROR] source=%s err=%v", source, err)
}
