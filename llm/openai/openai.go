// Package openai provides a golangchain LLM implementation backed by the
// OpenAI Chat Completions API (github.com/openai/openai-go).
package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"

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
	client openai.Client
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

	reqOpts := []option.RequestOption{option.WithAPIKey(cfg.apiKey)}
	if cfg.baseURL != "" {
		reqOpts = append(reqOpts, option.WithBaseURL(cfg.baseURL))
	}
	if cfg.orgID != "" {
		reqOpts = append(reqOpts, option.WithOrganization(cfg.orgID))
	}
	if cfg.projectID != "" {
		reqOpts = append(reqOpts, option.WithProject(cfg.projectID))
	}

	return &LLM{client: openai.NewClient(reqOpts...), cfg: cfg}, nil
}

// ModelName returns the configured default model name.
func (l *LLM) ModelName() string { return l.cfg.model }

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------

// Generate performs a blocking chat-completions call.
func (l *LLM) Generate(ctx context.Context, messages []schema.Message, opts ...llm.Option) (*schema.Generation, error) {
	o := llm.Apply(opts)
	params := l.buildParams(messages, o)

	resp, err := l.client.Chat.Completions.New(ctx, params)
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
		StopReason: choice.FinishReason,
		Usage: schema.TokenUsage{
			PromptTokens:     int(resp.Usage.PromptTokens),
			CompletionTokens: int(resp.Usage.CompletionTokens),
			TotalTokens:      int(resp.Usage.TotalTokens),
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

	modelName := l.cfg.model
	if o.Model != nil {
		modelName = *o.Model
	}
	gen.EstimatedCost = gen.Usage.EstimateCost(modelName)

	return gen, nil
}

// ---------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------

// Stream performs a streaming chat-completions call.
// Text deltas and tool-call deltas are emitted as separate StreamChunks.
// The final chunk carries Done=true and accumulated usage/tool calls.
func (l *LLM) Stream(ctx context.Context, messages []schema.Message, opts ...llm.Option) (<-chan schema.StreamChunk, error) {
	o := llm.Apply(opts)
	params := l.buildParams(messages, o)

	stream := l.client.Chat.Completions.NewStreaming(ctx, params)

	ch := make(chan schema.StreamChunk, 32)
	go func() {
		defer close(ch)
		defer func() { _ = stream.Close() }()

		type accumulating struct {
			id        string
			name      string
			arguments string
		}
		var acc []accumulating

		var totalUsage schema.TokenUsage

		for stream.Next() {
			chunk := stream.Current()
			if len(chunk.Choices) == 0 {
				continue
			}
			delta := chunk.Choices[0].Delta

			text := delta.Content

			for _, tcd := range delta.ToolCalls {
				idx := int(tcd.Index)
				for len(acc) <= idx {
					acc = append(acc, accumulating{})
				}
				a := &acc[idx]
				if tcd.ID != "" {
					a.id = tcd.ID
				}
				if tcd.Function.Name != "" {
					a.name = tcd.Function.Name
				}
				a.arguments += tcd.Function.Arguments
			}

			if len(delta.ToolCalls) > 0 {
				tcd := delta.ToolCalls[0]
				ch <- schema.StreamChunk{
					Text: text,
					ToolCallDelta: &schema.ToolCallDelta{
						Index:     int(tcd.Index),
						ID:        tcd.ID,
						Name:      tcd.Function.Name,
						Arguments: tcd.Function.Arguments,
					},
				}
			} else {
				ch <- schema.StreamChunk{Text: text}
			}

			if chunk.Usage.TotalTokens > 0 {
				totalUsage = schema.TokenUsage{
					PromptTokens:     int(chunk.Usage.PromptTokens),
					CompletionTokens: int(chunk.Usage.CompletionTokens),
					TotalTokens:      int(chunk.Usage.TotalTokens),
				}
			}
		}

		if err := stream.Err(); err != nil {
			ch <- schema.StreamChunk{Err: fmt.Errorf("openai: stream recv: %w", err)}
			return
		}

		done := schema.StreamChunk{Done: true, Usage: totalUsage}

		for _, a := range acc {
			if a.id != "" {
				done.ToolCalls = append(done.ToolCalls, schema.ToolCall{
					ID:        a.id,
					Type:      "function",
					Name:      a.name,
					Arguments: json.RawMessage(a.arguments),
				})
			}
		}

		ch <- done
	}()

	return ch, nil
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

func (l *LLM) buildParams(messages []schema.Message, o llm.Options) openai.ChatCompletionNewParams {
	model := l.cfg.model
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
	if o.FrequencyPenalty != nil {
		params.FrequencyPenalty = openai.Float(*o.FrequencyPenalty)
	}
	if o.PresencePenalty != nil {
		params.PresencePenalty = openai.Float(*o.PresencePenalty)
	}
	if o.Seed != nil {
		params.Seed = openai.Int(int64(*o.Seed))
	}
	if o.User != "" {
		params.User = openai.String(o.User)
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
	if o.ToolChoice != "" {
		switch o.ToolChoice {
		case "none":
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
				OfAuto: openai.String("none"),
			}
		case "auto":
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
				OfAuto: openai.String("auto"),
			}
		case "required":
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
				OfAuto: openai.String("required"),
			}
		default:
			params.ToolChoice = openai.ChatCompletionToolChoiceOptionParamOfChatCompletionNamedToolChoice(
				openai.ChatCompletionNamedToolChoiceFunctionParam{Name: o.ToolChoice},
			)
		}
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
			msg := openai.AssistantMessage(m.Content)
			// Attach tool calls if present
			if len(m.ToolCalls) > 0 {
				asst := openai.ChatCompletionAssistantMessageParam{}
				asst.Content.OfString = openai.String(m.Content)
				for _, tc := range m.ToolCalls {
					asst.ToolCalls = append(asst.ToolCalls, openai.ChatCompletionMessageToolCallParam{
						ID: tc.ID,
						Function: openai.ChatCompletionMessageToolCallFunctionParam{
							Name:      tc.Name,
							Arguments: string(tc.Arguments),
						},
					})
				}
				msg = openai.ChatCompletionMessageParamUnion{OfAssistant: &asst}
			}
			out = append(out, msg)
		case schema.RoleTool:
			out = append(out, openai.ToolMessage(m.Content, m.ToolCallID))
		}
	}
	return out
}
