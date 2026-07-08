// Package llm defines the LLM interface and option types shared by all
// provider implementations in sub-packages.
package llm

import (
	"context"

	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// Core interface
// ---------------------------------------------------------------------------

// LLM is the primary interface every provider must satisfy.
// Generate performs a single, blocking call; Stream returns a channel of
// incremental chunks closed when generation is complete or an error occurs.
type LLM interface {
	// Generate sends messages to the model and returns the full generation.
	Generate(ctx context.Context, messages []schema.Message, opts ...Option) (*schema.Generation, error)

	// Stream sends messages and returns a channel of incremental chunks.
	// The channel is closed after the final chunk (chunk.Done == true) or on
	// error (chunk.Err != nil). Callers must drain or stop reading the channel
	// to avoid goroutine leaks; passing a cancellable ctx is recommended.
	Stream(ctx context.Context, messages []schema.Message, opts ...Option) (<-chan schema.StreamChunk, error)

	// ModelName returns the resolved model identifier for this instance.
	ModelName() string
}

// ---------------------------------------------------------------------------
// Call options
// ---------------------------------------------------------------------------

// Options holds the full set of per-call parameters. All fields are pointers
// so callers can distinguish "not set" from a zero value.
type Options struct {
	// Model overrides the default model for this call only.
	Model *string
	// Temperature controls randomness [0, 2].
	Temperature *float64
	// MaxTokens limits the number of generated tokens.
	MaxTokens *int
	// TopP is nucleus sampling probability.
	TopP *float64
	// StopSequences halts generation when any sequence is produced.
	StopSequences []string
	// Tools is the list of tool definitions made available to the model.
	Tools []schema.ToolDef
	// ToolChoice forces tool selection: "" | "auto" | "none" | "required" | specific tool name.
	ToolChoice string
	// FrequencyPenalty reduces repetition [-2, 2].
	FrequencyPenalty *float64
	// PresencePenalty encourages new topics [-2, 2].
	PresencePenalty *float64
	// Seed makes outputs deterministic when supported.
	Seed *int
	// ResponseFormat requests structured output ("json_object" or "json_schema").
	ResponseFormat string
	// User is an end-user identifier passed to the API for abuse detection.
	User string
}

// Option is a functional option that mutates an Options struct.
type Option func(*Options)

// Apply runs all options against a zero Options and returns the result.
func Apply(opts []Option) Options {
	o := Options{}
	for _, fn := range opts {
		fn(&o)
	}
	return o
}

// WithModel overrides the model for a single call.
func WithModel(model string) Option {
	return func(o *Options) { o.Model = &model }
}

// WithTemperature sets the sampling temperature.
func WithTemperature(t float64) Option {
	return func(o *Options) { o.Temperature = &t }
}

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(n int) Option {
	return func(o *Options) { o.MaxTokens = &n }
}

// WithTopP sets nucleus sampling probability.
func WithTopP(p float64) Option {
	return func(o *Options) { o.TopP = &p }
}

// WithStopSequences sets halt sequences.
func WithStopSequences(seqs ...string) Option {
	return func(o *Options) { o.StopSequences = seqs }
}

// WithTools provides tool definitions for the model to call.
func WithTools(tools ...schema.ToolDef) Option {
	return func(o *Options) { o.Tools = tools }
}

// WithToolChoice sets the tool selection mode.
func WithToolChoice(choice string) Option {
	return func(o *Options) { o.ToolChoice = choice }
}

// WithFrequencyPenalty sets the frequency penalty.
func WithFrequencyPenalty(p float64) Option {
	return func(o *Options) { o.FrequencyPenalty = &p }
}

// WithPresencePenalty sets the presence penalty.
func WithPresencePenalty(p float64) Option {
	return func(o *Options) { o.PresencePenalty = &p }
}

// WithSeed makes generation deterministic (provider-dependent).
func WithSeed(seed int) Option {
	return func(o *Options) { o.Seed = &seed }
}

// WithResponseFormat requests a specific output format ("json_object", etc.).
func WithResponseFormat(format string) Option {
	return func(o *Options) { o.ResponseFormat = format }
}

// WithUser attaches an end-user identifier.
func WithUser(user string) Option {
	return func(o *Options) { o.User = user }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// MessagesToText concatenates message contents into a single string, useful
// for providers that only accept plain-text input.
func MessagesToText(messages []schema.Message) string {
	var out string
	for i, m := range messages {
		if i > 0 {
			out += "\n"
		}
		out += string(m.Role) + ": " + m.Content
	}
	return out
}
