// Package anthropic provides a golangchain LLM backed by Anthropic's Claude API.
package anthropic

import (
	"context"
	"errors"
	"fmt"
	"strings"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

type config struct {
	apiKey  string
	model   string
	baseURL string
}

// ProviderOption configures the Anthropic provider.
type ProviderOption func(*config)

func WithAPIKey(key string) ProviderOption  { return func(c *config) { c.apiKey = key } }
func WithModel(model string) ProviderOption { return func(c *config) { c.model = model } }
func WithBaseURL(url string) ProviderOption { return func(c *config) { c.baseURL = url } }

// ---------------------------------------------------------------------------
// LLM
// ---------------------------------------------------------------------------

// LLM is the Anthropic Claude provider.
type LLM struct {
	client *anthropicsdk.Client
	cfg    config
}

// New constructs an Anthropic LLM provider.
func New(opts ...ProviderOption) (*LLM, error) {
	cfg := config{model: "claude-3-5-sonnet-20241022"}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.apiKey == "" {
		return nil, errors.New("anthropic: API key is required (use WithAPIKey)")
	}

	clientOpts := []option.RequestOption{option.WithAPIKey(cfg.apiKey)}
	if cfg.baseURL != "" {
		clientOpts = append(clientOpts, option.WithBaseURL(cfg.baseURL))
	}
	client := anthropicsdk.NewClient(clientOpts...)
	return &LLM{client: &client, cfg: cfg}, nil
}

// ModelName returns the configured Claude model.
func (l *LLM) ModelName() string { return l.cfg.model }

// Generate performs a blocking Messages API call.
func (l *LLM) Generate(ctx context.Context, messages []schema.Message, opts ...llm.Option) (*schema.Generation, error) {
	o := llm.Apply(opts)
	model := l.cfg.model
	if o.Model != nil {
		model = *o.Model
	}

	system, anthropicMsgs := splitMessages(messages)

	maxTokens := int64(4096)
	if o.MaxTokens != nil {
		maxTokens = int64(*o.MaxTokens)
	}

	params := anthropicsdk.MessageNewParams{
		Model:     anthropicsdk.Model(model),
		Messages:  anthropicMsgs,
		MaxTokens: maxTokens,
	}
	if system != "" {
		params.System = []anthropicsdk.TextBlockParam{{Text: system}}
	}
	if o.Temperature != nil {
		params.Temperature = anthropicsdk.Float(*o.Temperature)
	}
	if o.TopP != nil {
		params.TopP = anthropicsdk.Float(*o.TopP)
	}

	resp, err := l.client.Messages.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("anthropic: generate: %w", err)
	}

	text := extractText(resp.Content)
	return &schema.Generation{
		Text:       text,
		Message:    schema.Message{Role: schema.RoleAI, Content: text},
		StopReason: string(resp.StopReason),
		Usage: schema.TokenUsage{
			PromptTokens:     int(resp.Usage.InputTokens),
			CompletionTokens: int(resp.Usage.OutputTokens),
			TotalTokens:      int(resp.Usage.InputTokens + resp.Usage.OutputTokens),
		},
	}, nil
}

// Stream performs a streaming Messages API call.
func (l *LLM) Stream(ctx context.Context, messages []schema.Message, opts ...llm.Option) (<-chan schema.StreamChunk, error) {
	o := llm.Apply(opts)
	model := l.cfg.model
	if o.Model != nil {
		model = *o.Model
	}

	system, anthropicMsgs := splitMessages(messages)
	maxTokens := int64(4096)
	if o.MaxTokens != nil {
		maxTokens = int64(*o.MaxTokens)
	}

	params := anthropicsdk.MessageNewParams{
		Model:     anthropicsdk.Model(model),
		Messages:  anthropicMsgs,
		MaxTokens: maxTokens,
	}
	if system != "" {
		params.System = []anthropicsdk.TextBlockParam{{Text: system}}
	}
	if o.Temperature != nil {
		params.Temperature = anthropicsdk.Float(*o.Temperature)
	}

	stream := l.client.Messages.NewStreaming(ctx, params)

	ch := make(chan schema.StreamChunk, 32)
	go func() {
		defer close(ch)
		for stream.Next() {
			event := stream.Current()
			// event.Type is a string: "content_block_delta", "message_stop", etc.
			switch event.Type {
			case "content_block_delta":
				if event.Delta.Type == "text_delta" {
					ch <- schema.StreamChunk{Text: event.Delta.Text}
				}
			case "message_stop":
				ch <- schema.StreamChunk{Done: true}
				return
			}
		}
		if err := stream.Err(); err != nil {
			ch <- schema.StreamChunk{Err: fmt.Errorf("anthropic: stream: %w", err)}
		}
	}()

	return ch, nil
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func splitMessages(msgs []schema.Message) (system string, out []anthropicsdk.MessageParam) {
	var sb strings.Builder
	for _, m := range msgs {
		switch m.Role {
		case schema.RoleSystem:
			if sb.Len() > 0 {
				sb.WriteString("\n")
			}
			sb.WriteString(m.Content)
		case schema.RoleHuman:
			out = append(out, anthropicsdk.NewUserMessage(
				anthropicsdk.NewTextBlock(m.Content),
			))
		case schema.RoleAI:
			out = append(out, anthropicsdk.NewAssistantMessage(
				anthropicsdk.NewTextBlock(m.Content),
			))
		}
	}
	system = sb.String()
	return
}

func extractText(blocks []anthropicsdk.ContentBlockUnion) string {
	var sb strings.Builder
	for _, b := range blocks {
		if b.Type == "text" {
			sb.WriteString(b.Text)
		}
	}
	return sb.String()
}
