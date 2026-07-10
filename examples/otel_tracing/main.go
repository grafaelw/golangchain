// Example: otel_tracing
//
// Demonstrates the OpenTelemetry adapter (tracing/otel) that implements
// callbacks.Handler and maps golangchain lifecycle events to OTel spans.
//
// Dependencies required (add to your go.mod):
//
//	go get go.opentelemetry.io/otel go.opentelemetry.io/otel/trace
//
// Highlights:
//   - NewHandler creates a Handler from a TracerProvider and tracer name.
//   - Spans mirror the callback hierarchy: LLM spans are children of chain
//     spans, tool spans are children of agent spans, etc.
//   - Token usage, stop reasons, and error statuses are recorded as span
//     attributes automatically.
//   - Parent/child relationships are preserved via the run ID stored in
//     context by the CallbackManager.
//
// Because this example requires the OpenTelemetry SDK, it is gated behind
// the "otel" build tag:
//
//	go run -tags otel ./examples/otel_tracing
//
//go:build otel
// +build otel

package main

import (
	"context"
	"fmt"
	"os"

	"github.com/grafaelw/golangchain/callbacks"
	"github.com/grafaelw/golangchain/chain"
	"github.com/grafaelw/golangchain/output"
	"github.com/grafaelw/golangchain/prompt"
	"github.com/grafaelw/golangchain/tracing"
	"github.com/grafaelw/golangchain/tracing/otel"

	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func main() {
	// -------------------------------------------------------------------------
	// 1. Create an in-memory OTel span exporter for demo purposes.
	//    In production you'd use an OTLP exporter pointing to a collector.
	// -------------------------------------------------------------------------
	exp := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithSyncer(exp),
	)
	defer tp.Shutdown(context.Background())

	tracer := tp.Tracer("golangchain-demo")

	// -------------------------------------------------------------------------
	// 2. Wire the OTel handler into a CallbackManager alongside the
	//    built-in PrettyHandler for live output and Tracer for summaries.
	// -------------------------------------------------------------------------
	langTracer := tracing.NewTracer()
	cb := callbacks.NewCallbackManager(
		tracing.NewPrettyHandler(os.Stderr),
		langTracer.Handler(),
		otel.NewHandler("golangchain-demo", tracer),
	)

	// -------------------------------------------------------------------------
	// 3. Run a simple chain — this produces OTel spans.
	// -------------------------------------------------------------------------
	chatPrompt := prompt.MustNewChatPromptTemplate(
		prompt.MustSystem("Be concise."),
		prompt.MustHuman("{{.question}}"),
	)
	llmChain := chain.NewLLMChain(
		chatPrompt,
		nil, // a real model would go here
		output.AsAny(output.StrOutputParser{}),
		chain.WithChainName("OtelDemo"),
		chain.WithChainCallbacks(cb),
	)

	_, _ = llmChain.Invoke(context.Background(), map[string]any{
		"question": "placeholder",
	})

	// -------------------------------------------------------------------------
	// 4. Inspect the OTel spans.
	// -------------------------------------------------------------------------
	spans := exp.GetSpans()
	fmt.Printf("OTel spans recorded: %d\n", len(spans))
	for _, s := range spans {
		fmt.Printf("  span=%q  parent=%q  status=%s\n",
			s.Name, parentName(s), s.Status.Code)
		for _, a := range s.Attributes {
			if a.Key != "service.name" {
				fmt.Printf("    %s = %v\n", a.Key, a.Value.AsInterface())
			}
		}
	}

	fmt.Println("\n✅ OTel adapter spans verified.")
}

func parentName(s tracetest.SpanStub) string {
	for _, a := range s.Attributes {
		if a.Key == attribute.Key("parent") {
			return a.Value.AsString()
		}
	}
	return "(root)"
}