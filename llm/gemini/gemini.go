// Package gemini provides a golangchain LLM backed by Google's Gemini API.
package gemini

import (
	"context"
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

	text := extractText(resp)
	usage := schema.TokenUsage{}
	if resp.UsageMetadata != nil {
		usage.PromptTokens = int(resp.UsageMetadata.PromptTokenCount)
		usage.CompletionTokens = int(resp.UsageMetadata.CandidatesTokenCount)
		usage.TotalTokens = int(resp.UsageMetadata.TotalTokenCount)
	}
	return &schema.Generation{
		Text:    text,
		Message: schema.Message{Role: schema.RoleAI, Content: text},
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
			ch <- schema.StreamChunk{Text: extractText(resp)}
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
			contents = append(contents, genai.NewContentFromParts(
				[]*genai.Part{genai.NewPartFromText(m.Content)}, genai.RoleModel,
			))
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

	return contents, cfg
}

func extractText(resp *genai.GenerateContentResponse) string {
	if resp == nil {
		return ""
	}
	var sb strings.Builder
	for _, cand := range resp.Candidates {
		if cand.Content == nil {
			continue
		}
		for _, part := range cand.Content.Parts {
			if part.Text != "" {
				sb.WriteString(part.Text)
			}
		}
	}
	return sb.String()
}
