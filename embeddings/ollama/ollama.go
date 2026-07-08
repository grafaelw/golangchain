// Package ollamaembeddings provides an Embedder implementation using a local
// Ollama server's /api/embed endpoint.
package ollamaembeddings

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
)

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

type config struct {
	baseURL   string // default: http://localhost:11434
	model     string // default: nomic-embed-text
	batchSize int
}

// ProviderOption configures the Ollama embedder.
type ProviderOption func(*config)

// WithBaseURL overrides the Ollama server URL.
func WithBaseURL(url string) ProviderOption { return func(c *config) { c.baseURL = url } }

// WithModel sets the Ollama embedding model.
func WithModel(model string) ProviderOption { return func(c *config) { c.model = model } }

// WithBatchSize sets the maximum documents per API call.
func WithBatchSize(n int) ProviderOption { return func(c *config) { c.batchSize = n } }

// ---------------------------------------------------------------------------
// Embedder
// ---------------------------------------------------------------------------

// Embedder implements embeddings.Embedder against a local Ollama server.
type Embedder struct {
	cfg    config
	client *http.Client
}

// New constructs an Ollama embedder.
func New(opts ...ProviderOption) (*Embedder, error) {
	cfg := config{
		baseURL:   "http://localhost:11434",
		model:     "nomic-embed-text",
		batchSize: 512,
	}
	for _, o := range opts {
		o(&cfg)
	}
	return &Embedder{
		cfg: cfg,
		client: &http.Client{
			Timeout: 60 * time.Second,
		},
	}, nil
}

// EmbedDocuments embeds a batch of document texts.
func (e *Embedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	all := make([][]float64, 0, len(texts))
	for i := 0; i < len(texts); i += e.cfg.batchSize {
		end := i + e.cfg.batchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch, err := e.embed(ctx, texts[i:end])
		if err != nil {
			return nil, err
		}
		all = append(all, batch...)
	}
	return all, nil
}

// EmbedQuery embeds a single query string.
func (e *Embedder) EmbedQuery(ctx context.Context, text string) ([]float64, error) {
	results, err := e.embed(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, errors.New("ollamaembeddings: empty result")
	}
	return results[0], nil
}

// Ensure interface compliance.
var _ embeddings.Embedder = (*Embedder)(nil)

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

type embedRequest struct {
	Model    string   `json:"model"`
	Input    []string `json:"input"`
	Truncate *bool    `json:"truncate,omitempty"`
}

type embedResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
}

func (e *Embedder) embed(ctx context.Context, texts []string) ([][]float64, error) {
	truncate := true
	body, err := json.Marshal(embedRequest{
		Model:    e.cfg.model,
		Input:    texts,
		Truncate: &truncate,
	})
	if err != nil {
		return nil, fmt.Errorf("ollamaembeddings: marshal: %w", err)
	}
	url := e.cfg.baseURL + "/api/embed"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ollamaembeddings: request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollamaembeddings: call: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollamaembeddings: status %d: %s", resp.StatusCode, string(b))
	}
	var result embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("ollamaembeddings: decode: %w", err)
	}
	if len(result.Embeddings) != len(texts) {
		return nil, fmt.Errorf("ollamaembeddings: expected %d embeddings, got %d", len(texts), len(result.Embeddings))
	}
	return result.Embeddings, nil
}
