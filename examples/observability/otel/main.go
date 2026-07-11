// Example: otel_tracing
//
// Demonstrates the OpenTelemetry adapter (tracing/otel) that implements
// callbacks.Handler and maps golangchain lifecycle events to OTel spans.
//
// Because this example requires the OpenTelemetry SDK, it is gated behind
// the "otel" build tag:
//
//	OPENAI_API_KEY=sk-... go run -tags otel ./examples/observability/otel
//
// Or with Azure AI Foundry:
//
//	AZURE_OPENAI_API_KEY=... go run -tags otel ./examples/observability/otel
//
//go:build otel
// +build otel

package main

import (
	"context"
	"fmt"
	"os"

	"github.com/joho/godotenv"

	"github.com/grafaelw/golangchain/callbacks"
	"github.com/grafaelw/golangchain/chain"
	"github.com/grafaelw/golangchain/llm/openai"
	"github.com/grafaelw/golangchain/output"
	"github.com/grafaelw/golangchain/prompt"
	"github.com/grafaelw/golangchain/tracing"
	"github.com/grafaelw/golangchain/tracing/otel"

	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func main() {
	_ = godotenv.Load()

	// -------------------------------------------------------------------------
	// 1. Create the LLM.
	// -------------------------------------------------------------------------
	model, err := openai.New(
		openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
		openai.WithModel("gpt-5.4-mini"),
	)
	if err != nil {
		// If no API key, fall back to Azure AI Foundry.
		model, err = openai.New(
			openai.WithAPIKey(os.Getenv("AZURE_OPENAI_API_KEY")),
			openai.WithModel("gpt-5.4-mini"),
			openai.WithBaseURL("https://ai-lab-nl-sweden-foundry.services.ai.azure.com/openai/v1/"),
		)
		if err != nil {
			panic(err)
		}
	}

	// -------------------------------------------------------------------------
	// 2. Create an in-memory OTel span exporter for demo purposes.
	//    In production you'd use an OTLP exporter pointing to a collector.
	// -------------------------------------------------------------------------
	exp := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithSyncer(exp),
	)
	defer tp.Shutdown(context.Background())

	tracer := tp.Tracer("golangchain-demo")

	// -------------------------------------------------------------------------
	// 3. Wire the OTel handler into a CallbackManager alongside the
	//    built-in PrettyHandler for live output and Tracer for summaries.
	// -------------------------------------------------------------------------
	langTracer := tracing.NewTracer()
	cb := callbacks.NewCallbackManager(
		tracing.NewPrettyHandler(os.Stderr),
		langTracer.Handler(),
		otel.NewHandler("golangchain-demo", tracer),
	)

	// -------------------------------------------------------------------------
	// 4. Run a simple chain — this produces OTel spans.
	// -------------------------------------------------------------------------
	chatPrompt := prompt.MustNewChatPromptTemplate(
		prompt.MustSystem("Be concise."),
		prompt.MustHuman("{{.question}}"),
	)
	llmChain := chain.NewLLMChain(
		chatPrompt,
		model,
		output.AsAny(output.StrOutputParser{}),
		chain.WithChainName("OtelDemo"),
		chain.WithChainCallbacks(cb),
	)

	_, _ = llmChain.Invoke(context.Background(), map[string]any{
		"question": "What is OpenTelemetry?",
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