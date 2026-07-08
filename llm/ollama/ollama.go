// Package ollama provides a golangchain LLM backed by a local Ollama server.
// Ollama exposes an OpenAI-compatible /v1/chat/completions endpoint, so this
// package is a thin wrapper around the openaicompat provider.
package ollama

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
	client openai.Client
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
	return &LLM{
		client: openai.NewClient(
			option.WithBaseURL(cfg.baseURL),
			option.WithAPIKey("ollama"),
		),
		cfg: cfg,
	}, nil
}

// ModelName returns the configured Ollama model tag.
func (l *LLM) ModelName() string { return l.cfg.model }

// Generate performs a blocking call to the local Ollama server.
func (l *LLM) Generate(ctx context.Context, messages []schema.Message, opts ...llm.Option) (*schema.Generation, error) {
	o := llm.Apply(opts)
	params := l.buildParams(messages, o)

	resp, err := l.client.Chat.Completions.New(ctx, params)
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
		StopReason: choice.FinishReason,
		Usage: schema.TokenUsage{
			PromptTokens:     int(resp.Usage.PromptTokens),
			CompletionTokens: int(resp.Usage.CompletionTokens),
			TotalTokens:      int(resp.Usage.TotalTokens),
		},
	}, nil
}

// Stream performs a streaming call to the local Ollama server.
func (l *LLM) Stream(ctx context.Context, messages []schema.Message, opts ...llm.Option) (<-chan schema.StreamChunk, error) {
	o := llm.Apply(opts)
	params := l.buildParams(messages, o)

	stream := l.client.Chat.Completions.NewStreaming(ctx, params)

	ch := make(chan schema.StreamChunk, 32)
	go func() {
		defer close(ch)
		defer func() { _ = stream.Close() }()

		for stream.Next() {
			chunk := stream.Current()
			if len(chunk.Choices) == 0 {
				continue
			}
			ch <- schema.StreamChunk{Text: chunk.Choices[0].Delta.Content}
		}
		if err := stream.Err(); err != nil {
			ch <- schema.StreamChunk{Err: fmt.Errorf("ollama: stream recv: %w", err)}
			return
		}
		ch <- schema.StreamChunk{Done: true}
	}()

	return ch, nil
}

func (l *LLM) buildParams(messages []schema.Message, o llm.Options) openai.ChatCompletionNewParams {
	model := l.cfg.model
	if o.Model != nil {
		model = *o.Model
	}
	params := openai.ChatCompletionNewParams{
		Model:    shared.ChatModel(model),
		Messages: toOllamaMessages(messages),
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

func toOllamaMessages(msgs []schema.Message) []openai.ChatCompletionMessageParamUnion {
	out := make([]openai.ChatCompletionMessageParamUnion, 0, len(msgs))
	for _, m := range msgs {
		switch m.Role {
		case schema.RoleSystem:
			out = append(out, openai.SystemMessage(m.Content))
		case schema.RoleHuman:
			out = append(out, openai.UserMessage(m.Content))
		case schema.RoleAI:
			out = append(out, buildAIMessage(m))
		case schema.RoleTool:
			out = append(out, openai.ToolMessage(m.Content, m.ToolCallID))
		}
	}
	return out
}

func buildAIMessage(m schema.Message) openai.ChatCompletionMessageParamUnion {
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
	if len(asst.ToolCalls) > 0 {
		return openai.ChatCompletionMessageParamUnion{OfAssistant: &asst}
	}
	return openai.AssistantMessage(m.Content)
}
