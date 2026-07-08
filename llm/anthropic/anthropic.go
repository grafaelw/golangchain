// Package anthropic provides a golangchain LLM backed by Anthropic's Claude API.
package anthropic

import (
	"context"
	"encoding/json"
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
	for _, td := range o.Tools {
		params.Tools = append(params.Tools, toAnthropicTool(td))
	}
	if o.ToolChoice != "" {
		params.ToolChoice = toAnthropicToolChoice(o.ToolChoice)
	}

	resp, err := l.client.Messages.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("anthropic: generate: %w", err)
	}

	text, toolCalls := extractContent(resp.Content)
	msg := schema.Message{Role: schema.RoleAI, Content: text, ToolCalls: toolCalls}
	return &schema.Generation{
		Text:       text,
		Message:    msg,
		StopReason: string(resp.StopReason),
		Usage: schema.TokenUsage{
			PromptTokens:     int(resp.Usage.InputTokens),
			CompletionTokens: int(resp.Usage.OutputTokens),
			TotalTokens:      int(resp.Usage.InputTokens + resp.Usage.OutputTokens),
		},
	}, nil
}

// Stream performs a streaming Messages API call with support for tool calls.
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
	for _, td := range o.Tools {
		params.Tools = append(params.Tools, toAnthropicTool(td))
	}
	if o.ToolChoice != "" {
		params.ToolChoice = toAnthropicToolChoice(o.ToolChoice)
	}

	stream := l.client.Messages.NewStreaming(ctx, params)

	ch := make(chan schema.StreamChunk, 32)
	go func() {
		defer close(ch)
		type accum struct {
			id   string
			name string
			buf  strings.Builder
		}
		toolAccum := map[int]*accum{}
		blockIdx := 0
		for stream.Next() {
			event := stream.Current()
			switch event.Type {
			case "content_block_start":
				if event.ContentBlock.Type == "tool_use" {
					toolAccum[blockIdx] = &accum{
						id:   event.ContentBlock.ID,
						name: event.ContentBlock.Name,
					}
				}
			case "content_block_delta":
				if event.Delta.Type == "text_delta" {
					ch <- schema.StreamChunk{Text: event.Delta.Text}
				} else if event.Delta.Type == "input_json_delta" && event.Delta.PartialJSON != "" {
					if a, ok := toolAccum[blockIdx]; ok {
						a.buf.WriteString(event.Delta.PartialJSON)
					}
				}
			case "content_block_stop":
				blockIdx++
			case "message_stop":
				if len(toolAccum) > 0 {
					delta := schema.ToolCallDelta{}
					for i := 0; i < len(toolAccum); i++ {
						if tc, ok := toolAccum[i]; ok {
							delta.Index = i
							delta.ID = tc.id
							delta.Name = tc.name
							delta.Arguments = tc.buf.String()
						}
					}
					ch <- schema.StreamChunk{ToolCallDelta: &delta}
				}
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
			blocks := buildAssistantBlocks(m)
			out = append(out, anthropicsdk.NewAssistantMessage(blocks...))
		case schema.RoleTool:
			out = append(out, anthropicsdk.NewUserMessage(
				anthropicsdk.NewToolResultBlock(m.ToolCallID, m.Content, false),
			))
		}
	}
	system = sb.String()
	return
}

func buildAssistantBlocks(m schema.Message) []anthropicsdk.ContentBlockParamUnion {
	var blocks []anthropicsdk.ContentBlockParamUnion
	if m.Content != "" {
		blocks = append(blocks, anthropicsdk.NewTextBlock(m.Content))
	}
	for _, tc := range m.ToolCalls {
		var input any
		if len(tc.Arguments) > 0 {
			_ = json.Unmarshal(tc.Arguments, &input)
		}
		if input == nil {
			input = map[string]any{}
		}
		blocks = append(blocks, anthropicsdk.NewToolUseBlock(tc.ID, input, tc.Name))
	}
	return blocks
}

func extractContent(blocks []anthropicsdk.ContentBlockUnion) (string, []schema.ToolCall) {
	var text strings.Builder
	var toolCalls []schema.ToolCall
	for _, b := range blocks {
		switch b.Type {
		case "text":
			text.WriteString(b.Text)
		case "tool_use":
			tb := b.AsToolUse()
			toolCalls = append(toolCalls, schema.ToolCall{
				ID:        tb.ID,
				Name:      tb.Name,
				Arguments: tb.Input,
			})
		}
	}
	return text.String(), toolCalls
}

func toAnthropicTool(td schema.ToolDef) anthropicsdk.ToolUnionParam {
	var props any
	if len(td.Parameters) > 0 {
		var m map[string]any
		if err := json.Unmarshal(td.Parameters, &m); err == nil {
			if p, ok := m["properties"]; ok {
				props = p
			}
		}
	}
	var required []string
	if len(td.Parameters) > 0 {
		var m map[string]any
		if err := json.Unmarshal(td.Parameters, &m); err == nil {
			if req, ok := m["required"].([]any); ok {
				for _, r := range req {
					if s, ok := r.(string); ok {
						required = append(required, s)
					}
				}
			}
		}
	}
	return anthropicsdk.ToolUnionParam{
		OfTool: &anthropicsdk.ToolParam{
			Name:        td.Name,
			Description: anthropicsdk.String(td.Description),
			InputSchema: anthropicsdk.ToolInputSchemaParam{
				Type:       "object",
				Properties: props,
				Required:   required,
			},
		},
	}
}

func toAnthropicToolChoice(choice string) anthropicsdk.ToolChoiceUnionParam {
	switch choice {
	case "auto":
		return anthropicsdk.ToolChoiceUnionParam{OfAuto: &anthropicsdk.ToolChoiceAutoParam{}}
	case "any", "required":
		return anthropicsdk.ToolChoiceUnionParam{OfAny: &anthropicsdk.ToolChoiceAnyParam{}}
	case "none":
		return anthropicsdk.ToolChoiceUnionParam{OfNone: &anthropicsdk.ToolChoiceNoneParam{}}
	default:
		return anthropicsdk.ToolChoiceUnionParam{
			OfTool: &anthropicsdk.ToolChoiceToolParam{Name: choice},
		}
	}
}
