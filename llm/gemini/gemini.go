// Package gemini provides a golangchain LLM backed by Google's Gemini API.
package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"google.golang.org/genai"

	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

type config struct {
	apiKey string
	model  string
}

// ProviderOption configures the Gemini provider.
type ProviderOption func(*config)

func WithAPIKey(key string) ProviderOption  { return func(c *config) { c.apiKey = key } }
func WithModel(model string) ProviderOption { return func(c *config) { c.model = model } }

// ---------------------------------------------------------------------------
// LLM
// ---------------------------------------------------------------------------

// LLM is the Google Gemini provider.
type LLM struct {
	client *genai.Client
	cfg    config
}

// New constructs a Gemini LLM provider.
func New(ctx context.Context, opts ...ProviderOption) (*LLM, error) {
	cfg := config{model: "gemini-1.5-flash"}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.apiKey == "" {
		return nil, errors.New("gemini: API key is required (use WithAPIKey)")
	}

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  cfg.apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, fmt.Errorf("gemini: create client: %w", err)
	}

	return &LLM{client: client, cfg: cfg}, nil
}

// ModelName returns the configured Gemini model.
func (l *LLM) ModelName() string { return l.cfg.model }

// Generate performs a blocking GenerateContent call.
func (l *LLM) Generate(ctx context.Context, messages []schema.Message, opts ...llm.Option) (*schema.Generation, error) {
	o := llm.Apply(opts)
	model := l.cfg.model
	if o.Model != nil {
		model = *o.Model
	}

	contents, cfg := toGeminiRequest(messages, o)
	resp, err := l.client.Models.GenerateContent(ctx, model, contents, cfg)
	if err != nil {
		return nil, fmt.Errorf("gemini: generate: %w", err)
	}

	text, toolCalls := extractGeminiContent(resp)
	msg := schema.Message{Role: schema.RoleAI, Content: text, ToolCalls: toolCalls}
	usage := schema.TokenUsage{}
	if resp.UsageMetadata != nil {
		usage.PromptTokens = int(resp.UsageMetadata.PromptTokenCount)
		usage.CompletionTokens = int(resp.UsageMetadata.CandidatesTokenCount)
		usage.TotalTokens = int(resp.UsageMetadata.TotalTokenCount)
	}
	return &schema.Generation{
		Text:    text,
		Message: msg,
		Usage:   usage,
	}, nil
}

// Stream performs a streaming GenerateContent call.
func (l *LLM) Stream(ctx context.Context, messages []schema.Message, opts ...llm.Option) (<-chan schema.StreamChunk, error) {
	o := llm.Apply(opts)
	model := l.cfg.model
	if o.Model != nil {
		model = *o.Model
	}

	contents, cfg := toGeminiRequest(messages, o)

	ch := make(chan schema.StreamChunk, 32)
	go func() {
		defer close(ch)
		iter := l.client.Models.GenerateContentStream(ctx, model, contents, cfg)
		for resp, err := range iter {
			if err != nil {
				ch <- schema.StreamChunk{Err: fmt.Errorf("gemini: stream: %w", err)}
				return
			}
			for _, cand := range resp.Candidates {
				if cand.Content == nil {
					continue
				}
				for _, part := range cand.Content.Parts {
					if part.Text != "" {
						ch <- schema.StreamChunk{Text: part.Text}
					}
					if part.FunctionCall != nil {
						args, _ := json.Marshal(part.FunctionCall.Args)
						ch <- schema.StreamChunk{
							ToolCallDelta: &schema.ToolCallDelta{
								ID:        part.FunctionCall.ID,
								Name:      part.FunctionCall.Name,
								Arguments: string(args),
							},
						}
					}
				}
			}
		}
		ch <- schema.StreamChunk{Done: true}
	}()

	return ch, nil
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func toGeminiRequest(msgs []schema.Message, o llm.Options) ([]*genai.Content, *genai.GenerateContentConfig) {
	var contents []*genai.Content
	var systemParts []*genai.Part

	for _, m := range msgs {
		switch m.Role {
		case schema.RoleSystem:
			systemParts = append(systemParts, genai.NewPartFromText(m.Content))
		case schema.RoleHuman:
			contents = append(contents, genai.NewContentFromParts(
				[]*genai.Part{genai.NewPartFromText(m.Content)}, genai.RoleUser,
			))
		case schema.RoleAI:
			parts := buildAIParts(m)
			contents = append(contents, genai.NewContentFromParts(parts, genai.RoleModel))
		case schema.RoleTool:
			if m.ToolCallID != "" {
				name := extractToolName(msgs, m.ToolCallID)
				resp := map[string]any{}
				if strings.TrimSpace(m.Content) != "" {
					if err := json.Unmarshal([]byte(m.Content), &resp); err != nil {
						resp = map[string]any{"output": m.Content}
					}
				}
				contents = append(contents, genai.NewContentFromFunctionResponse(
					name, resp, genai.RoleUser,
				))
			}
		}
	}

	cfg := &genai.GenerateContentConfig{}

	if len(systemParts) > 0 {
		cfg.SystemInstruction = genai.NewContentFromParts(systemParts, "system")
	}
	if o.Temperature != nil {
		t := float32(*o.Temperature)
		cfg.Temperature = &t
	}
	if o.MaxTokens != nil {
		cfg.MaxOutputTokens = int32(*o.MaxTokens)
	}
	if o.TopP != nil {
		p := float32(*o.TopP)
		cfg.TopP = &p
	}
	if len(o.StopSequences) > 0 {
		cfg.StopSequences = o.StopSequences
	}
	if len(o.Tools) > 0 {
		cfg.Tools = append(cfg.Tools, toGeminiTool(o.Tools))
	}
	if o.ToolChoice != "" {
		cfg.ToolConfig = toGeminiToolConfig(o.ToolChoice)
	}

	return contents, cfg
}

func buildAIParts(m schema.Message) []*genai.Part {
	var parts []*genai.Part
	if m.Content != "" {
		parts = append(parts, genai.NewPartFromText(m.Content))
	}
	for _, tc := range m.ToolCalls {
		parts = append(parts, genai.NewPartFromFunctionCall(tc.Name, parseToolArgs(tc.Arguments)))
	}
	return parts
}

func extractToolName(msgs []schema.Message, toolCallID string) string {
	for _, m := range msgs {
		for _, tc := range m.ToolCalls {
			if tc.ID == toolCallID {
				return tc.Name
			}
		}
	}
	return ""
}

func parseToolArgs(raw json.RawMessage) map[string]any {
	if len(raw) == 0 {
		return map[string]any{}
	}
	var args map[string]any
	if err := json.Unmarshal(raw, &args); err != nil {
		return map[string]any{}
	}
	return args
}

func extractGeminiContent(resp *genai.GenerateContentResponse) (string, []schema.ToolCall) {
	if resp == nil {
		return "", nil
	}
	var text strings.Builder
	var toolCalls []schema.ToolCall
	for _, cand := range resp.Candidates {
		if cand.Content == nil {
			continue
		}
		for _, part := range cand.Content.Parts {
			if part.Text != "" {
				text.WriteString(part.Text)
			}
			if part.FunctionCall != nil {
				args, _ := json.Marshal(part.FunctionCall.Args)
				toolCalls = append(toolCalls, schema.ToolCall{
					ID:        part.FunctionCall.ID,
					Name:      part.FunctionCall.Name,
					Arguments: args,
				})
			}
		}
	}
	return text.String(), toolCalls
}

func toGeminiTool(toolDefs []schema.ToolDef) *genai.Tool {
	var decls []*genai.FunctionDeclaration
	for _, td := range toolDefs {
		decl := &genai.FunctionDeclaration{
			Name:        td.Name,
			Description: td.Description,
		}
		if len(td.Parameters) > 0 {
			var params map[string]any
			if err := json.Unmarshal(td.Parameters, &params); err == nil {
				decl.ParametersJsonSchema = params
			}
		}
		decls = append(decls, decl)
	}
	return &genai.Tool{FunctionDeclarations: decls}
}

func toGeminiToolConfig(choice string) *genai.ToolConfig {
	config := &genai.ToolConfig{}
	switch choice {
	case "auto":
		config.FunctionCallingConfig = &genai.FunctionCallingConfig{
			Mode: genai.FunctionCallingConfigModeAuto,
		}
	case "any", "required":
		config.FunctionCallingConfig = &genai.FunctionCallingConfig{
			Mode: genai.FunctionCallingConfigModeAny,
		}
	case "none":
		config.FunctionCallingConfig = &genai.FunctionCallingConfig{
			Mode: genai.FunctionCallingConfigModeNone,
		}
	}
	return config
}
