// Package azure provides a golangchain LLM backed by Azure OpenAI Service.
// It reuses the go-openai client with an Azure-specific configuration
// (endpoint, deployment name, API version).
package azure

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
	apiKey         string
	endpoint       string // e.g. https://<resource>.openai.azure.com/
	deployment     string // deployment name (= model alias in Azure portal)
	apiVersion     string // e.g. "2024-02-01"
	entraToken     string // alternative to apiKey: Azure AD token
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
	client *goopenai.Client
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

	var clientCfg goopenai.ClientConfig
	if cfg.entraToken != "" {
		clientCfg = goopenai.DefaultAzureConfig(cfg.entraToken, cfg.endpoint)
	} else {
		clientCfg = goopenai.DefaultAzureConfig(cfg.apiKey, cfg.endpoint)
	}
	clientCfg.APIVersion = cfg.apiVersion

	return &LLM{client: goopenai.NewClientWithConfig(clientCfg), cfg: cfg}, nil
}

// ModelName returns the deployment name (used as the model identifier in Azure).
func (l *LLM) ModelName() string { return l.cfg.deployment }

// Generate performs a blocking chat-completions call via Azure OpenAI.
func (l *LLM) Generate(ctx context.Context, messages []schema.Message, opts ...llm.Option) (*schema.Generation, error) {
	o := llm.Apply(opts)
	req := l.buildRequest(messages, o, false)

	resp, err := l.client.CreateChatCompletion(ctx, req)
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
		StopReason: string(choice.FinishReason),
		Usage: schema.TokenUsage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}
	for _, tc := range choice.Message.ToolCalls {
		gen.Message.ToolCalls = append(gen.Message.ToolCalls, schema.ToolCall{
			ID: tc.ID, Type: string(tc.Type),
			Name: tc.Function.Name, Arguments: []byte(tc.Function.Arguments),
		})
	}
	return gen, nil
}

// Stream performs a streaming chat-completions call via Azure OpenAI.
func (l *LLM) Stream(ctx context.Context, messages []schema.Message, opts ...llm.Option) (<-chan schema.StreamChunk, error) {
	o := llm.Apply(opts)
	req := l.buildRequest(messages, o, true)

	stream, err := l.client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("azure: stream: %w", err)
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
				ch <- schema.StreamChunk{Err: fmt.Errorf("azure: stream recv: %w", err)}
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
	// Azure uses deployment name as the model field.
	model := l.cfg.deployment
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
	return req
}

func toOpenAIMessages(msgs []schema.Message) []goopenai.ChatCompletionMessage {
	out := make([]goopenai.ChatCompletionMessage, 0, len(msgs))
	for _, m := range msgs {
		cm := goopenai.ChatCompletionMessage{Content: m.Content, Name: m.Name}
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
		out = append(out, cm)
	}
	return out
}
