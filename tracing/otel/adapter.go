// Package otel provides an OpenTelemetry adapter that implements
// callbacks.Handler, mapping golangchain lifecycle events to OTel spans.
//
// Usage:
//
//	import (
//	    "go.opentelemetry.io/otel"
//	    gotel "github.com/grafaelw/golangchain/tracing/otel"
//	)
//
//	cb := callbacks.NewCallbackManager(
//	    gotel.NewHandler("my-app", otel.GetTracerProvider().Tracer("golangchain")),
//	)
//	// All chains, agents, and graphs using this callback manager will
//	// emit OTel spans.
//
// Dependencies (add to your go.mod):
//
//	go get go.opentelemetry.io/otel go.opentelemetry.io/otel/trace
package otel

import (
	"context"
	"sync"

	"github.com/grafaelw/golangchain/callbacks"
	"github.com/grafaelw/golangchain/schema"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

// Handler bridges golangchain callbacks to OpenTelemetry spans.
// Each lifecycle event starts or ends a span keyed by the
// callbacks run ID stored in the context.
type Handler struct {
	callbacks.NoOpHandler
	serviceName string
	tracer      trace.Tracer

	mu    sync.Mutex
	spans map[string]trace.Span
}

// NewHandler creates an OTel handler. Pass your application's TracerProvider
// and tracer name.
func NewHandler(serviceName string, tracer trace.Tracer) *Handler {
	return &Handler{
		serviceName: serviceName,
		tracer:      tracer,
		spans:       make(map[string]trace.Span),
	}
}

func (h *Handler) startSpan(ctx context.Context, name string, kind trace.SpanKind, attrs ...attribute.KeyValue) (context.Context, trace.Span) {
	runID := callbacks.RunIDFromContext(ctx)
	if runID == "" {
		return ctx, trace.SpanFromContext(ctx)
	}

	parentID := callbacks.ParentRunIDFromContext(ctx)
	var parentCtx context.Context
	if parentID != "" {
		if parent, ok := h.getSpan(parentID); ok {
			parentCtx = trace.ContextWithSpan(ctx, parent)
		}
	}
	if parentCtx == nil {
		parentCtx = ctx
	}

	spanCtx, span := h.tracer.Start(parentCtx, name,
		trace.WithSpanKind(kind),
		trace.WithAttributes(attrs...),
	)
	h.setSpan(runID, span)
	return spanCtx, span
}

func (h *Handler) endSpan(ctx context.Context) trace.Span {
	runID := callbacks.RunIDFromContext(ctx)
	if runID == "" {
		return nil
	}
	span, ok := h.getSpan(runID)
	if !ok {
		return nil
	}
	span.End()
	h.deleteSpan(runID)
	return span
}

func (h *Handler) getSpan(id string) (trace.Span, bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	s, ok := h.spans[id]
	return s, ok
}

func (h *Handler) setSpan(id string, s trace.Span) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.spans[id] = s
}

func (h *Handler) deleteSpan(id string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.spans, id)
}

// OnLLMStart starts a "llm/{model}" span of kind INTERNAL.
func (h *Handler) OnLLMStart(ctx context.Context, model string, msgs []schema.Message) {
	attrs := []attribute.KeyValue{
		attribute.String("llm.model", model),
		attribute.Int("llm.messages", len(msgs)),
	}
	if name := h.serviceName; name != "" {
		attrs = append(attrs, attribute.String("service.name", name))
	}
	h.startSpan(ctx, "llm/"+model, trace.SpanKindInternal, attrs...)
}

// OnLLMEnd ends the LLM span and records token usage.
func (h *Handler) OnLLMEnd(ctx context.Context, _ string, gen *schema.Generation) {
	span := h.endSpan(ctx)
	if span != nil && gen != nil {
		if gen.StopReason != "" {
			span.SetAttributes(attribute.String("llm.stop_reason", gen.StopReason))
		}
		if gen.Usage.TotalTokens > 0 {
			span.SetAttributes(
				attribute.Int("llm.usage.prompt_tokens", gen.Usage.PromptTokens),
				attribute.Int("llm.usage.completion_tokens", gen.Usage.CompletionTokens),
				attribute.Int("llm.usage.total_tokens", gen.Usage.TotalTokens),
			)
		}
	}
}

// OnLLMStream is a no-op for OTel spans (streaming tokens are not
// individually traced).
func (h *Handler) OnLLMStream(_ context.Context, _ string, _ schema.StreamChunk) {}

// OnChainStart starts a "chain/{name}" span.
func (h *Handler) OnChainStart(ctx context.Context, name string, _ map[string]any) {
	h.startSpan(ctx, "chain/"+name, trace.SpanKindInternal)
}

// OnChainEnd ends the chain span.
func (h *Handler) OnChainEnd(ctx context.Context, _ string, _ map[string]any) {
	h.endSpan(ctx)
}

// OnToolStart starts a "tool/{name}" span.
func (h *Handler) OnToolStart(ctx context.Context, name, _ string) {
	h.startSpan(ctx, "tool/"+name, trace.SpanKindInternal)
}

// OnToolEnd ends the tool span.
func (h *Handler) OnToolEnd(ctx context.Context, _ string, _ string) {
	h.endSpan(ctx)
}

// OnAgentAction starts an agent action span.
func (h *Handler) OnAgentAction(ctx context.Context, action schema.AgentAction) {
	h.startSpan(ctx, "agent/"+action.Tool, trace.SpanKindInternal,
		attribute.String("agent.tool", action.Tool),
		attribute.String("agent.input", action.ToolInput),
	)
}

// OnAgentFinish ends the agent span.
func (h *Handler) OnAgentFinish(ctx context.Context, _ schema.AgentFinish) {
	h.endSpan(ctx)
}

// OnGraphNodeStart starts a "graph/{graph}/{node}" span.
func (h *Handler) OnGraphNodeStart(ctx context.Context, graph, node string) {
	h.startSpan(ctx, "graph/"+graph+"/"+node, trace.SpanKindInternal,
		attribute.String("graph.name", graph),
		attribute.String("graph.node", node),
	)
}

// OnGraphNodeEnd ends the graph node span.
func (h *Handler) OnGraphNodeEnd(ctx context.Context, _, _ string) {
	h.endSpan(ctx)
}

// OnGraphCheckpoint records a checkpoint event as a span attribute.
func (h *Handler) OnGraphCheckpoint(ctx context.Context, _, threadID string) {
	span := trace.SpanFromContext(ctx)
	if span.IsRecording() {
		span.AddEvent("graph.checkpoint", trace.WithAttributes(
			attribute.String("graph.thread_id", threadID),
		))
	}
}

// OnError marks the current span as errored and ends it.
func (h *Handler) OnError(ctx context.Context, source string, err error) {
	runID := callbacks.RunIDFromContext(ctx)
	if runID == "" {
		return
	}
	span, ok := h.getSpan(runID)
	if !ok {
		return
	}
	span.SetAttributes(attribute.String("error.source", source))
	span.SetStatus(codes.Error, err.Error())
	span.End()
	h.deleteSpan(runID)
}
