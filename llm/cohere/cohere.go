// Package cohere provides an embeddings.Embedder backed by Cohere's Embed
// API. It communicates via Cohere's REST API using only the standard library.
package cohere

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/grafaelw/golangchain/embeddings"
)

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

// Option configures the Cohere embedder.
type Option func(*Embedder)

// WithModel sets the Cohere embedding model (default "embed-english-v3.0").
func WithModel(model string) Option {
	return func(e *Embedder) { e.model = model }
}

// WithBaseURL overrides the API base URL (default "https://api.cohere.com/v1").
func WithBaseURL(url string) Option {
	return func(e *Embedder) { e.baseURL = strings.TrimSuffix(url, "/") }
}

// WithHTTPClient overrides the HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(e *Embedder) { e.client = client }
}

// WithBatchSize sets the maximum texts per API call (default 96).
func WithBatchSize(n int) Option {
	return func(e *Embedder) { e.batchSize = n }
}

// ---------------------------------------------------------------------------
// Embedder
// ---------------------------------------------------------------------------

// Embedder implements embeddings.Embedder backed by Cohere's embed endpoint.
type Embedder struct {
	apiKey    string
	model     string
	baseURL   string
	batchSize int
	client    *http.Client
}

// New creates a Cohere embedding client.
//
//	emb, err := cohere.New(os.Getenv("COHERE_API_KEY"),
//	    cohere.WithModel("embed-english-v3.0"),
//	)
func New(apiKey string, opts ...Option) (*Embedder, error) {
	if apiKey == "" {
		return nil, errors.New("cohere: API key is required")
	}
	e := &Embedder{
		apiKey:    apiKey,
		model:     "embed-english-v3.0",
		baseURL:   "https://api.cohere.com/v1",
		batchSize: 96,
		client:    &http.Client{Timeout: 60 * time.Second},
	}
	for _, o := range opts {
		o(e)
	}
	return e, nil
}

var _ embeddings.Embedder = (*Embedder)(nil)

// ---------------------------------------------------------------------------
// Embedder implementation
// ---------------------------------------------------------------------------

// EmbedDocuments generates embeddings for a batch of document texts.
func (e *Embedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	all := make([][]float64, 0, len(texts))
	for i := 0; i < len(texts); i += e.batchSize {
		end := i + e.batchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch, err := e.embed(ctx, texts[i:end], "search_document")
		if err != nil {
			return nil, err
		}
		all = append(all, batch...)
	}
	return all, nil
}

// EmbedQuery generates an embedding for a single query string.
func (e *Embedder) EmbedQuery(ctx context.Context, text string) ([]float64, error) {
	results, err := e.embed(ctx, []string{text}, "search_query")
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, errors.New("cohere: empty embedding result")
	}
	return results[0], nil
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

type embedRequest struct {
	Model     string   `json:"model"`
	Texts     []string `json:"texts"`
	InputType string   `json:"input_type"`
}

type embedResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
}

func (e *Embedder) embed(ctx context.Context, texts []string, inputType string) ([][]float64, error) {
	body, err := json.Marshal(embedRequest{
		Model:     e.model,
		Texts:     texts,
		InputType: inputType,
	})
	if err != nil {
		return nil, fmt.Errorf("cohere: marshal: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+"/embed", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("cohere: request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+e.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("cohere: call: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("cohere: status %d: %s", resp.StatusCode, string(b))
	}
	var result embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("cohere: decode: %w", err)
	}
	if len(result.Embeddings) != len(texts) {
		return nil, fmt.Errorf("cohere: expected %d embeddings, got %d", len(texts), len(result.Embeddings))
	}
	return result.Embeddings, nil
}
