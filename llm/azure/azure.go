// Package azure provides a golangchain LLM backed by Azure OpenAI Service.
// It uses the official openai-go client with azure-specific configuration
// (endpoint, deployment name, API version).
package azure

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/openai/openai-go"
	azureoption "github.com/openai/openai-go/azure"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"

	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

type config struct {
	apiKey     string
	endpoint   string // e.g. https://<resource>.openai.azure.com/
	deployment string // deployment name (= model alias in Azure portal)
	apiVersion string // e.g. "2024-02-01"
	entraToken string // alternative to apiKey: Azure AD bearer token
}

// ProviderOption configures the Azure OpenAI provider.
type ProviderOption func(*config)

func WithAPIKey(key string) ProviderOption     { return func(c *config) { c.apiKey = key } }
func WithEndpoint(ep string) ProviderOption    { return func(c *config) { c.endpoint = ep } }
func WithDeployment(d string) ProviderOption   { return func(c *config) { c.deployment = d } }
func WithAPIVersion(v string) ProviderOption   { return func(c *config) { c.apiVersion = v } }
func WithEntraToken(t string) ProviderOption   { return func(c *config) { c.entraToken = t } }

// ---------------------------------------------------------------------------
// LLM
// ---------------------------------------------------------------------------

// LLM is the Azure OpenAI provider.
type LLM struct {
	client openai.Client
	cfg    config
}

// New constructs an Azure OpenAI LLM.
//
//	llm, err := azure.New(
//	    azure.WithAPIKey(os.Getenv("AZURE_OPENAI_API_KEY")),
//	    azure.WithEndpoint(os.Getenv("AZURE_OPENAI_ENDPOINT")),
//	    azure.WithDeployment("gpt-4o"),
//	    azure.WithAPIVersion("2024-02-01"),
//	)
func New(opts ...ProviderOption) (*LLM, error) {
	cfg := config{apiVersion: "2024-02-01"}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.endpoint == "" {
		return nil, errors.New("azure: endpoint is required (use WithEndpoint)")
	}
	if cfg.deployment == "" {
		return nil, errors.New("azure: deployment name is required (use WithDeployment)")
	}
	if cfg.apiKey == "" && cfg.entraToken == "" {
		return nil, errors.New("azure: either API key (WithAPIKey) or Entra token (WithEntraToken) is required")
	}

	reqOpts := []option.RequestOption{
		azureoption.WithEndpoint(cfg.endpoint, cfg.apiVersion),
	}
	if cfg.entraToken != "" {
		reqOpts = append(reqOpts, option.WithHeader("Authorization", "Bearer "+cfg.entraToken))
	} else {
		reqOpts = append(reqOpts, azureoption.WithAPIKey(cfg.apiKey))
	}

	return &LLM{client: openai.NewClient(reqOpts...), cfg: cfg}, nil
}

// ModelName returns the deployment name (used as the model identifier in Azure).
func (l *LLM) ModelName() string { return l.cfg.deployment }

// Generate performs a blocking chat-completions call via Azure OpenAI.
func (l *LLM) Generate(ctx context.Context, messages []schema.Message, opts ...llm.Option) (*schema.Generation, error) {
	o := llm.Apply(opts)
	params := l.buildParams(messages, o)

	resp, err := l.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("azure: generate: %w", err)
	}
	if len(resp.Choices) == 0 {
		return nil, errors.New("azure: generate: no choices returned")
	}

	choice := resp.Choices[0]
	gen := &schema.Generation{
		Text:       choice.Message.Content,
		Message:    schema.Message{Role: schema.RoleAI, Content: choice.Message.Content},
		StopReason: choice.FinishReason,
		Usage: schema.TokenUsage{
			PromptTokens:     int(resp.Usage.PromptTokens),
			CompletionTokens: int(resp.Usage.CompletionTokens),
			TotalTokens:      int(resp.Usage.TotalTokens),
		},
	}
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

// Stream performs a streaming chat-completions call via Azure OpenAI.
func (l *LLM) Stream(ctx context.Context, messages []schema.Message, opts ...llm.Option) (<-chan schema.StreamChunk, error) {
	o := llm.Apply(opts)
	params := l.buildParams(messages, o)

	stream := l.client.Chat.Completions.NewStreaming(ctx, params)

	ch := make(chan schema.StreamChunk, 32)
	go func() {
		defer close(ch)
		defer stream.Close()

		for stream.Next() {
			chunk := stream.Current()
			if len(chunk.Choices) == 0 {
				continue
			}
			ch <- schema.StreamChunk{Text: chunk.Choices[0].Delta.Content}
		}
		if err := stream.Err(); err != nil {
			ch <- schema.StreamChunk{Err: fmt.Errorf("azure: stream recv: %w", err)}
			return
		}
		ch <- schema.StreamChunk{Done: true}
	}()

	return ch, nil
}

func (l *LLM) buildParams(messages []schema.Message, o llm.Options) openai.ChatCompletionNewParams {
	// Azure uses deployment name as the model field.
	model := l.cfg.deployment
	if o.Model != nil {
		model = *o.Model
	}
	params := openai.ChatCompletionNewParams{
		Model:    shared.ChatModel(model),
		Messages: toOpenAIMessages(messages),
	}
	if o.Temperature != nil {
		params.Temperature = openai.Float(*o.Temperature)
	}
	if o.MaxTokens != nil {
		params.MaxTokens = openai.Int(int64(*o.MaxTokens))
	}
	if o.TopP != nil {
		params.TopP = openai.Float(*o.TopP)
	}
	if len(o.StopSequences) > 0 {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: o.StopSequences}
	}
	for _, td := range o.Tools {
		var fnParams shared.FunctionParameters
		if len(td.Parameters) > 0 {
			_ = json.Unmarshal(td.Parameters, &fnParams)
		}
		params.Tools = append(params.Tools, openai.ChatCompletionToolParam{
			Function: shared.FunctionDefinitionParam{
				Name:        td.Name,
				Description: openai.String(td.Description),
				Parameters:  fnParams,
			},
		})
	}
	return params
}

func toOpenAIMessages(msgs []schema.Message) []openai.ChatCompletionMessageParamUnion {
	out := make([]openai.ChatCompletionMessageParamUnion, 0, len(msgs))
	for _, m := range msgs {
		switch m.Role {
		case schema.RoleSystem:
			out = append(out, openai.SystemMessage(m.Content))
		case schema.RoleHuman:
			out = append(out, openai.UserMessage(m.Content))
		case schema.RoleAI:
			out = append(out, openai.AssistantMessage(m.Content))
		case schema.RoleTool:
			out = append(out, openai.ToolMessage(m.Content, m.ToolCallID))
		}
	}
	return out
}
