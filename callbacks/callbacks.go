// Package callbacks defines the Handler interface and CallbackManager used by
// every component in golangchain to emit lifecycle events.
// Users plug in custom handlers for logging, tracing, or OpenTelemetry export.
package callbacks

import (
	"context"
	"crypto/rand"
	"encoding/hex"
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

func (NoOpHandler) OnLLMStart(_ context.Context, _ string, _ []schema.Message)    {}
func (NoOpHandler) OnLLMEnd(_ context.Context, _ string, _ *schema.Generation)    {}
func (NoOpHandler) OnLLMStream(_ context.Context, _ string, _ schema.StreamChunk) {}
func (NoOpHandler) OnChainStart(_ context.Context, _ string, _ map[string]any)    {}
func (NoOpHandler) OnChainEnd(_ context.Context, _ string, _ map[string]any)      {}
func (NoOpHandler) OnToolStart(_ context.Context, _, _ string)                    {}
func (NoOpHandler) OnToolEnd(_ context.Context, _, _ string)                      {}
func (NoOpHandler) OnAgentAction(_ context.Context, _ schema.AgentAction)         {}
func (NoOpHandler) OnAgentFinish(_ context.Context, _ schema.AgentFinish)         {}
func (NoOpHandler) OnGraphNodeStart(_ context.Context, _, _ string)               {}
func (NoOpHandler) OnGraphNodeEnd(_ context.Context, _, _ string)                 {}
func (NoOpHandler) OnGraphCheckpoint(_ context.Context, _, _ string)              {}
func (NoOpHandler) OnError(_ context.Context, _ string, _ error)                  {}

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
// Context helpers — run ID propagation for hierarchical tracing
// ---------------------------------------------------------------------------

type runKey struct{}
type parentRunKey struct{}
type callbacksContextKey struct{}

// NewRunID generates a short, unique identifier for a trace run.
func NewRunID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// WithRunID returns a new context that marks id as the current run.
// The previous run ID (if any) is automatically promoted to the parent slot,
// so handlers can reconstruct the full parent→child hierarchy.
func WithRunID(ctx context.Context, id string) context.Context {
	prev := RunIDFromContext(ctx)
	ctx = context.WithValue(ctx, parentRunKey{}, prev)
	return context.WithValue(ctx, runKey{}, id)
}

// RunIDFromContext returns the current run ID stored in ctx.
// Returns "" if none has been set.
func RunIDFromContext(ctx context.Context) string {
	id, _ := ctx.Value(runKey{}).(string)
	return id
}

// ParentRunIDFromContext returns the parent run ID stored in ctx.
// Returns "" if the current run is a root (has no parent).
func ParentRunIDFromContext(ctx context.Context) string {
	id, _ := ctx.Value(parentRunKey{}).(string)
	return id
}

// WithCallbackManager stores cm in ctx so functions that do not receive the
// manager directly (e.g., agent Plan methods) can retrieve it.
func WithCallbackManager(ctx context.Context, cm *CallbackManager) context.Context {
	return context.WithValue(ctx, callbacksContextKey{}, cm)
}

// CallbackManagerFromContext retrieves a CallbackManager from ctx.
// Returns nil if none has been stored.
func CallbackManagerFromContext(ctx context.Context) *CallbackManager {
	cm, _ := ctx.Value(callbacksContextKey{}).(*CallbackManager)
	return cm
}

// ---------------------------------------------------------------------------
// LoggingHandler — ready-made handler that prints to any printf-style func
// ---------------------------------------------------------------------------

// LoggingHandler is a built-in Handler that logs every event using the
// supplied printf-style function (e.g. log.Printf or a slog wrapper).
type LoggingHandler struct {
	NoOpHandler
	logFn func(format string, args ...any)
}

// NewLoggingHandler returns a LoggingHandler that writes via logFn.
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
func (l *LoggingHandler) OnLLMStream(_ context.Context, model string, chunk schema.StreamChunk) {
	if chunk.Done {
		l.logFn("[LLM] stream done model=%s", model)
	}
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
func (l *LoggingHandler) OnGraphCheckpoint(_ context.Context, graph, threadID string) {
	l.logFn("[Graph] checkpoint graph=%s thread=%s", graph, threadID)
}
func (l *LoggingHandler) OnError(_ context.Context, source string, err error) {
	l.logFn("[ERROR] source=%s err=%v", source, err)
}
