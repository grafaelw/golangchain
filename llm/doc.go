// Package llm defines the LLM interface and the functional-option system used
// by all provider implementations.
//
// # LLM interface
//
// Every provider implements [LLM]:
//
//	type LLM interface {
//	    ModelName() string
//	    Generate(ctx context.Context, msgs []schema.Message, opts ...Option) (*schema.Generation, error)
//	    Stream(ctx context.Context, msgs []schema.Message, opts ...Option) (<-chan schema.StreamChunk, error)
//	}
//
// # Call options
//
// Per-call behaviour is controlled through functional options:
//
//	gen, err := model.Generate(ctx, messages,
//	    llm.WithTemperature(0.7),
//	    llm.WithMaxTokens(512),
//	    llm.WithTools(myToolDef),
//	)
//
// Available options: [WithModel], [WithTemperature], [WithMaxTokens], [WithTopP],
// [WithStopSequences], [WithTools], [WithToolChoice], [WithFrequencyPenalty],
// [WithPresencePenalty], [WithSeed], [WithResponseFormat], [WithUser].
//
// # Provider packages
//
//   - llm/openai      — OpenAI Chat Completions API
//   - llm/azure       — Azure OpenAI Service
//   - llm/anthropic   — Anthropic Claude
//   - llm/gemini      — Google Gemini
//   - llm/ollama      — Local Ollama server
//   - llm/openaicompat — any OpenAI-schema compatible server
package llm
