// Package ollama provides a golangchain LLM backed by a local Ollama server.
// Ollama exposes an OpenAI-compatible /v1/chat/completions endpoint, so this
// package is a thin wrapper around the openaicompat provider.
package ollama

import (
	"context"
	"errors"
	"fmt"
	"io"

	goopenai "github.com/sashabaranov/go-openai"

	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

type config struct {
	baseURL string // default: http://localhost:11434/v1
	model   string // e.g. "llama3", "mistral", "phi3"
}

// ProviderOption configures the Ollama provider.
type ProviderOption func(*config)

// WithBaseURL overrides the Ollama server URL.
func WithBaseURL(url string) ProviderOption { return func(c *config) { c.baseURL = url } }

// WithModel sets the Ollama model tag.
func WithModel(model string) ProviderOption { return func(c *config) { c.model = model } }

// ---------------------------------------------------------------------------
// LLM
// ---------------------------------------------------------------------------

// LLM is the Ollama provider.
type LLM struct {
	client *goopenai.Client
	cfg    config
}

// New constructs an Ollama LLM provider.
//
//	llm, err := ollama.New(
//	    ollama.WithModel("llama3"),
//	    // ollama.WithBaseURL("http://192.168.1.5:11434/v1"),
//	)
func New(opts ...ProviderOption) (*LLM, error) {
	cfg := config{
		baseURL: "http://localhost:11434/v1",
		model:   "llama3",
	}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.model == "" {
		return nil, errors.New("ollama: model name is required (use WithModel)")
	}

	// Ollama's OpenAI-compatible API requires the API key field to be non-empty
	// but doesn't validate it.
	clientCfg := goopenai.DefaultConfig("ollama")
	clientCfg.BaseURL = cfg.baseURL

	return &LLM{client: goopenai.NewClientWithConfig(clientCfg), cfg: cfg}, nil
}

// ModelName returns the configured Ollama model tag.
func (l *LLM) ModelName() string { return l.cfg.model }

// Generate performs a blocking call to the local Ollama server.
func (l *LLM) Generate(ctx context.Context, messages []schema.Message, opts ...llm.Option) (*schema.Generation, error) {
	o := llm.Apply(opts)
	req := l.buildRequest(messages, o, false)

	resp, err := l.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("ollama: generate: %w", err)
	}
	if len(resp.Choices) == 0 {
		return nil, errors.New("ollama: generate: no choices returned")
	}

	choice := resp.Choices[0]
	return &schema.Generation{
		Text:       choice.Message.Content,
		Message:    schema.Message{Role: schema.RoleAI, Content: choice.Message.Content},
		StopReason: string(choice.FinishReason),
		Usage: schema.TokenUsage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}, nil
}

// Stream performs a streaming call to the local Ollama server.
func (l *LLM) Stream(ctx context.Context, messages []schema.Message, opts ...llm.Option) (<-chan schema.StreamChunk, error) {
	o := llm.Apply(opts)
	req := l.buildRequest(messages, o, true)

	stream, err := l.client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("ollama: stream: %w", err)
	}

	ch := make(chan schema.StreamChunk, 32)
	go func() {
		defer close(ch)
		defer stream.Close()
		for {
			resp, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				ch <- schema.StreamChunk{Done: true}
				return
			}
			if err != nil {
				ch <- schema.StreamChunk{Err: fmt.Errorf("ollama: stream recv: %w", err)}
				return
			}
			if len(resp.Choices) == 0 {
				continue
			}
			ch <- schema.StreamChunk{Text: resp.Choices[0].Delta.Content}
		}
	}()

	return ch, nil
}

func (l *LLM) buildRequest(messages []schema.Message, o llm.Options, stream bool) goopenai.ChatCompletionRequest {
	model := l.cfg.model
	if o.Model != nil {
		model = *o.Model
	}
	req := goopenai.ChatCompletionRequest{
		Model:    model,
		Messages: toOpenAIMessages(messages),
		Stream:   stream,
	}
	if o.Temperature != nil {
		req.Temperature = float32(*o.Temperature)
	}
	if o.MaxTokens != nil {
		req.MaxTokens = *o.MaxTokens
	}
	return req
}

func toOpenAIMessages(msgs []schema.Message) []goopenai.ChatCompletionMessage {
	out := make([]goopenai.ChatCompletionMessage, 0, len(msgs))
	for _, m := range msgs {
		cm := goopenai.ChatCompletionMessage{Content: m.Content}
		switch m.Role {
		case schema.RoleSystem:
			cm.Role = goopenai.ChatMessageRoleSystem
		case schema.RoleHuman:
			cm.Role = goopenai.ChatMessageRoleUser
		case schema.RoleAI:
			cm.Role = goopenai.ChatMessageRoleAssistant
		case schema.RoleTool:
			cm.Role = goopenai.ChatMessageRoleTool
		}
		out = append(out, cm)
	}
	return out
}
