// Package openai provides a golangchain LLM implementation backed by the
// OpenAI Chat Completions API (github.com/sashabaranov/go-openai).
package openai

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
// Provider config
// ---------------------------------------------------------------------------

type config struct {
	apiKey    string
	model     string
	baseURL   string // optional: override for OpenAI-compatible servers
	orgID     string
	projectID string
}

// ProviderOption configures the OpenAI provider at construction time.
type ProviderOption func(*config)

// WithAPIKey sets the OpenAI API key.
func WithAPIKey(key string) ProviderOption { return func(c *config) { c.apiKey = key } }

// WithModel sets the default model (e.g. "gpt-4o", "gpt-4-turbo").
func WithModel(model string) ProviderOption { return func(c *config) { c.model = model } }

// WithBaseURL overrides the API base URL (useful for proxies or compatible servers).
func WithBaseURL(url string) ProviderOption { return func(c *config) { c.baseURL = url } }

// WithOrganization sets the OpenAI organization ID.
func WithOrganization(org string) ProviderOption { return func(c *config) { c.orgID = org } }

// WithProject sets the OpenAI project ID.
func WithProject(proj string) ProviderOption { return func(c *config) { c.projectID = proj } }

// ---------------------------------------------------------------------------
// LLM struct
// ---------------------------------------------------------------------------

// LLM is the OpenAI provider. Construct with New().
type LLM struct {
	client *goopenai.Client
	cfg    config
}

// New constructs an OpenAI LLM provider.
//
//	llm, err := openai.New(
//	    openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
//	    openai.WithModel("gpt-4o"),
//	)
func New(opts ...ProviderOption) (*LLM, error) {
	cfg := config{model: "gpt-4o"}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.apiKey == "" {
		return nil, errors.New("openai: API key is required (use WithAPIKey or set OPENAI_API_KEY)")
	}

	clientCfg := goopenai.DefaultConfig(cfg.apiKey)
	if cfg.baseURL != "" {
		clientCfg.BaseURL = cfg.baseURL
	}
	if cfg.orgID != "" {
		clientCfg.OrgID = cfg.orgID
	}

	return &LLM{client: goopenai.NewClientWithConfig(clientCfg), cfg: cfg}, nil
}

// ModelName returns the configured default model name.
func (l *LLM) ModelName() string { return l.cfg.model }

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------

// Generate performs a blocking chat-completions call.
func (l *LLM) Generate(ctx context.Context, messages []schema.Message, opts ...llm.Option) (*schema.Generation, error) {
	o := llm.Apply(opts)
	req := l.buildRequest(messages, o, false)

	resp, err := l.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("openai: generate: %w", err)
	}
	if len(resp.Choices) == 0 {
		return nil, errors.New("openai: generate: no choices returned")
	}

	choice := resp.Choices[0]
	gen := &schema.Generation{
		Text: choice.Message.Content,
		Message: schema.Message{
			Role:    schema.RoleAI,
			Content: choice.Message.Content,
		},
		StopReason: string(choice.FinishReason),
		Usage: schema.TokenUsage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}

	// Translate tool calls
	for _, tc := range choice.Message.ToolCalls {
		gen.Message.ToolCalls = append(gen.Message.ToolCalls, schema.ToolCall{
			ID:        tc.ID,
			Type:      string(tc.Type),
			Name:      tc.Function.Name,
			Arguments: []byte(tc.Function.Arguments),
		})
	}

	return gen, nil
}

// ---------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------

// Stream performs a streaming chat-completions call.
// The returned channel is closed after the final chunk or on error.
func (l *LLM) Stream(ctx context.Context, messages []schema.Message, opts ...llm.Option) (<-chan schema.StreamChunk, error) {
	o := llm.Apply(opts)
	req := l.buildRequest(messages, o, true)

	stream, err := l.client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("openai: stream: %w", err)
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
				ch <- schema.StreamChunk{Err: fmt.Errorf("openai: stream recv: %w", err)}
				return
			}
			if len(resp.Choices) == 0 {
				continue
			}
			delta := resp.Choices[0].Delta
			ch <- schema.StreamChunk{Text: delta.Content}
		}
	}()

	return ch, nil
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

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
	if o.TopP != nil {
		req.TopP = float32(*o.TopP)
	}
	if len(o.StopSequences) > 0 {
		req.Stop = o.StopSequences
	}
	if o.FrequencyPenalty != nil {
		req.FrequencyPenalty = float32(*o.FrequencyPenalty)
	}
	if o.PresencePenalty != nil {
		req.PresencePenalty = float32(*o.PresencePenalty)
	}
	if o.Seed != nil {
		req.Seed = o.Seed
	}
	if o.User != "" {
		req.User = o.User
	}
	for _, td := range o.Tools {
		req.Tools = append(req.Tools, goopenai.Tool{
			Type: goopenai.ToolTypeFunction,
			Function: &goopenai.FunctionDefinition{
				Name:        td.Name,
				Description: td.Description,
				Parameters:  td.Parameters,
			},
		})
	}
	if o.ToolChoice != "" {
		switch o.ToolChoice {
		case "none":
			req.ToolChoice = "none"
		case "auto":
			req.ToolChoice = "auto"
		case "required":
			req.ToolChoice = "required"
		default:
			req.ToolChoice = goopenai.ToolChoice{
				Type:     goopenai.ToolTypeFunction,
				Function: goopenai.ToolFunction{Name: o.ToolChoice},
			}
		}
	}

	return req
}

func toOpenAIMessages(msgs []schema.Message) []goopenai.ChatCompletionMessage {
	out := make([]goopenai.ChatCompletionMessage, 0, len(msgs))
	for _, m := range msgs {
		cm := goopenai.ChatCompletionMessage{
			Content: m.Content,
			Name:    m.Name,
		}
		switch m.Role {
		case schema.RoleSystem:
			cm.Role = goopenai.ChatMessageRoleSystem
		case schema.RoleHuman:
			cm.Role = goopenai.ChatMessageRoleUser
		case schema.RoleAI:
			cm.Role = goopenai.ChatMessageRoleAssistant
		case schema.RoleTool:
			cm.Role = goopenai.ChatMessageRoleTool
			cm.ToolCallID = m.ToolCallID
		}
		for _, tc := range m.ToolCalls {
			cm.ToolCalls = append(cm.ToolCalls, goopenai.ToolCall{
				ID:   tc.ID,
				Type: goopenai.ToolType(tc.Type),
				Function: goopenai.FunctionCall{
					Name:      tc.Name,
					Arguments: string(tc.Arguments),
				},
			})
		}
		out = append(out, cm)
	}
	return out
}
