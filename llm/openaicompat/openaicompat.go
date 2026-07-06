// Package openaicompat provides a golangchain LLM for any server that speaks
// the OpenAI Chat Completions API schema (LM Studio, vLLM, LocalAI, etc.).
// It is identical in structure to the openai package but accepts an arbitrary
// base URL and does not require a real API key.
package openaicompat

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
	baseURL string
	apiKey  string // many servers accept any non-empty value
	model   string
}

// ProviderOption configures the OpenAI-compatible provider.
type ProviderOption func(*config)

func WithBaseURL(url string) ProviderOption  { return func(c *config) { c.baseURL = url } }
func WithAPIKey(key string) ProviderOption   { return func(c *config) { c.apiKey = key } }
func WithModel(model string) ProviderOption  { return func(c *config) { c.model = model } }

// ---------------------------------------------------------------------------
// LLM
// ---------------------------------------------------------------------------

// LLM is the generic OpenAI-compatible provider.
type LLM struct {
	client *goopenai.Client
	cfg    config
}

// New constructs an OpenAI-compatible LLM provider.
//
//	// LM Studio running locally:
//	llm, err := openaicompat.New(
//	    openaicompat.WithBaseURL("http://localhost:1234/v1"),
//	    openaicompat.WithModel("local-model"),
//	)
func New(opts ...ProviderOption) (*LLM, error) {
	cfg := config{apiKey: "not-required", model: "default"}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.baseURL == "" {
		return nil, errors.New("openaicompat: base URL is required (use WithBaseURL)")
	}

	clientCfg := goopenai.DefaultConfig(cfg.apiKey)
	clientCfg.BaseURL = cfg.baseURL

	return &LLM{client: goopenai.NewClientWithConfig(clientCfg), cfg: cfg}, nil
}

// ModelName returns the configured model name.
func (l *LLM) ModelName() string { return l.cfg.model }

// Generate performs a blocking chat-completions call.
func (l *LLM) Generate(ctx context.Context, messages []schema.Message, opts ...llm.Option) (*schema.Generation, error) {
	o := llm.Apply(opts)
	req := l.buildRequest(messages, o, false)

	resp, err := l.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("openaicompat: generate: %w", err)
	}
	if len(resp.Choices) == 0 {
		return nil, errors.New("openaicompat: generate: no choices returned")
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

// Stream performs a streaming chat-completions call.
func (l *LLM) Stream(ctx context.Context, messages []schema.Message, opts ...llm.Option) (<-chan schema.StreamChunk, error) {
	o := llm.Apply(opts)
	req := l.buildRequest(messages, o, true)

	stream, err := l.client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("openaicompat: stream: %w", err)
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
				ch <- schema.StreamChunk{Err: fmt.Errorf("openaicompat: stream recv: %w", err)}
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
