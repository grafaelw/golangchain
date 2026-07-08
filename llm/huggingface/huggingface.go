// Package huggingface provides embeddings.Embedder and llm.LLM
// implementations backed by the HuggingFace Inference API. It communicates
// via REST using only the standard library.
package huggingface

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/grafaelw/golangchain/embeddings"
	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// Embedder
// ---------------------------------------------------------------------------

// EmbedderOption configures the HuggingFace embedder.
type EmbedderOption func(*EmbedderConfig)

// EmbedderConfig holds immutable configuration for the embedder.
type EmbedderConfig struct {
	apiKey  string
	model   string
	baseURL string
	client  *http.Client
}

// WithEmbeddingModel sets the HuggingFace embedding model
// (default "sentence-transformers/all-MiniLM-L6-v2").
func WithEmbeddingModel(model string) EmbedderOption {
	return func(c *EmbedderConfig) { c.model = model }
}

// WithEmbeddingBaseURL overrides the API base URL.
func WithEmbeddingBaseURL(url string) EmbedderOption {
	return func(c *EmbedderConfig) { c.baseURL = url }
}

// WithEmbeddingHTTPClient overrides the HTTP client.
func WithEmbeddingHTTPClient(client *http.Client) EmbedderOption {
	return func(c *EmbedderConfig) { c.client = client }
}

// Embedder implements embeddings.Embedder via the HuggingFace Inference API.
type Embedder struct {
	cfg EmbedderConfig
}

// NewEmbedder creates a HuggingFace embedding client.
//
//	emb, err := huggingface.NewEmbedder("hf_...",
//	    huggingface.WithEmbeddingModel("sentence-transformers/all-MiniLM-L6-v2"),
//	)
func NewEmbedder(apiKey string, opts ...EmbedderOption) (*Embedder, error) {
	if apiKey == "" {
		return nil, errors.New("huggingface: API key is required")
	}
	cfg := EmbedderConfig{
		apiKey:  apiKey,
		model:   "sentence-transformers/all-MiniLM-L6-v2",
		baseURL: "https://api-inference.huggingface.co",
		client:  &http.Client{Timeout: 60 * time.Second},
	}
	for _, o := range opts {
		o(&cfg)
	}
	return &Embedder{cfg: cfg}, nil
}

var _ embeddings.Embedder = (*Embedder)(nil)

// EmbedDocuments embeds a batch of document texts.
func (e *Embedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	return e.embedMultiple(ctx, texts)
}

// EmbedQuery embeds a single query text.
func (e *Embedder) EmbedQuery(ctx context.Context, text string) ([]float64, error) {
	results, err := e.embedMultiple(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, errors.New("huggingface: empty embedding result")
	}
	return results[0], nil
}

func (e *Embedder) embedMultiple(ctx context.Context, texts []string) ([][]float64, error) {
	url := e.cfg.baseURL + "/models/" + e.cfg.model
	body, err := json.Marshal(map[string]any{"inputs": texts})
	if err != nil {
		return nil, fmt.Errorf("huggingface: marshal: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("huggingface: request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+e.cfg.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.cfg.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("huggingface: embed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("huggingface: embed: status %d: %s", resp.StatusCode, string(b))
	}

	var raw []json.RawMessage
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, fmt.Errorf("huggingface: embed: decode: %w", err)
	}
	out := make([][]float64, 0, len(raw))
	for _, r := range raw {
		var vec []float64
		if err := json.Unmarshal(r, &vec); err != nil {
			return nil, fmt.Errorf("huggingface: embed: unmarshal vector: %w", err)
		}
		out = append(out, vec)
	}
	if len(out) != len(texts) {
		return nil, fmt.Errorf("huggingface: expected %d embeddings, got %d", len(texts), len(out))
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// LLM
// ---------------------------------------------------------------------------

// LLMOption configures the HuggingFace LLM.
type LLMOption func(*LLMConfig)

// LLMConfig holds immutable configuration for the LLM.
type LLMConfig struct {
	apiKey    string
	model     string
	baseURL   string
	client    *http.Client
	maxTokens int
}

// WithLLMModel sets the HuggingFace text-generation model
// (default "mistralai/Mistral-7B-Instruct-v0.3").
func WithLLMModel(model string) LLMOption {
	return func(c *LLMConfig) { c.model = model }
}

// WithLLMBaseURL overrides the API base URL.
func WithLLMBaseURL(url string) LLMOption {
	return func(c *LLMConfig) { c.baseURL = url }
}

// WithLLMHTTPClient overrides the HTTP client.
func WithLLMHTTPClient(client *http.Client) LLMOption {
	return func(c *LLMConfig) { c.client = client }
}

// WithLLMMaxTokens sets the maximum tokens to generate (default 512).
func WithLLMMaxTokens(n int) LLMOption {
	return func(c *LLMConfig) { c.maxTokens = n }
}

// LLM implements llm.LLM via the HuggingFace Inference API.
type LLM struct {
	cfg LLMConfig
}

// NewLLM creates a HuggingFace LLM client.
//
//	model, err := huggingface.NewLLM("hf_...",
//	    huggingface.WithLLMModel("mistralai/Mistral-7B-Instruct-v0.3"),
//	)
func NewLLM(apiKey string, opts ...LLMOption) (*LLM, error) {
	if apiKey == "" {
		return nil, errors.New("huggingface: API key is required")
	}
	cfg := LLMConfig{
		apiKey:    apiKey,
		model:     "mistralai/Mistral-7B-Instruct-v0.3",
		baseURL:   "https://api-inference.huggingface.co",
		client:    &http.Client{Timeout: 120 * time.Second},
		maxTokens: 512,
	}
	for _, o := range opts {
		o(&cfg)
	}
	return &LLM{cfg: cfg}, nil
}

var _ llm.LLM = (*LLM)(nil)

// ModelName returns the configured HuggingFace model ID.
func (l *LLM) ModelName() string { return l.cfg.model }

// Generate sends a prompt and returns the completed text.
func (l *LLM) Generate(ctx context.Context, messages []schema.Message, opts ...llm.Option) (*schema.Generation, error) {
	o := llm.Apply(opts)
	prompt := llm.MessagesToText(messages)

	maxTokens := l.cfg.maxTokens
	if o.MaxTokens != nil {
		maxTokens = *o.MaxTokens
	}
	temperature := 0.7
	if o.Temperature != nil {
		temperature = *o.Temperature
	}

	body, err := json.Marshal(map[string]any{
		"inputs": prompt,
		"parameters": map[string]any{
			"max_new_tokens":   maxTokens,
			"temperature":      temperature,
			"return_full_text": false,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("huggingface: marshal: %w", err)
	}

	url := l.cfg.baseURL + "/models/" + l.cfg.model
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("huggingface: request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+l.cfg.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := l.cfg.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("huggingface: generate: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("huggingface: generate: status %d: %s", resp.StatusCode, string(b))
	}

	var result []struct {
		GeneratedText string `json:"generated_text"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("huggingface: generate: decode: %w", err)
	}
	if len(result) == 0 {
		return nil, errors.New("huggingface: generate: empty result")
	}

	text := result[0].GeneratedText
	return &schema.Generation{
		Text:    text,
		Message: schema.Message{Role: schema.RoleAI, Content: text},
	}, nil
}

// Stream sends a prompt and returns a channel of incremental chunks.
// HuggingFace Inference API does not natively support true streaming; this
// falls back to a single chunk containing the full response.
func (l *LLM) Stream(ctx context.Context, messages []schema.Message, opts ...llm.Option) (<-chan schema.StreamChunk, error) {
	gen, err := l.Generate(ctx, messages, opts...)
	ch := make(chan schema.StreamChunk, 2)
	if err != nil {
		close(ch)
		return nil, err
	}
	go func() {
		defer close(ch)
		ch <- schema.StreamChunk{Text: gen.Text}
		ch <- schema.StreamChunk{Done: true, Usage: gen.Usage}
	}()
	return ch, nil
}
