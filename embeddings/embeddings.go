// Package embeddings defines the Embedder interface and provides OpenAI and
// Azure OpenAI implementations for generating vector embeddings from text.
package embeddings

import (
	"context"
	"errors"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/azure"
	"github.com/openai/openai-go/option"
)

// ---------------------------------------------------------------------------
// Embedder interface
// ---------------------------------------------------------------------------

// Embedder generates dense vector embeddings from text. Implementations are
// used by VectorStore to index documents and encode queries.
type Embedder interface {
	// EmbedDocuments generates embeddings for a batch of documents.
	// Returns one []float64 per document, in the same order.
	EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error)

	// EmbedQuery generates an embedding for a single query string.
	// May use a different model instruction than EmbedDocuments.
	EmbedQuery(ctx context.Context, text string) ([]float64, error)
}

// ---------------------------------------------------------------------------
// OpenAI Embedder
// ---------------------------------------------------------------------------

// OpenAIEmbedder uses OpenAI's text-embedding models.
type OpenAIEmbedder struct {
	client    openai.Client
	model     string
	batchSize int
	baseURL   string // optional: override for OpenAI-compatible servers
}

// OpenAIEmbedderOption configures an OpenAIEmbedder.
type OpenAIEmbedderOption func(*OpenAIEmbedder)

func WithModel(model string) OpenAIEmbedderOption {
	return func(e *OpenAIEmbedder) { e.model = model }
}
func WithBatchSize(n int) OpenAIEmbedderOption {
	return func(e *OpenAIEmbedder) { e.batchSize = n }
}

// WithBaseURL overrides the API base URL (useful for proxies or
// OpenAI-compatible servers).
func WithBaseURL(url string) OpenAIEmbedderOption {
	return func(e *OpenAIEmbedder) { e.baseURL = url }
}

// NewOpenAIEmbedder creates an OpenAI embedding client.
//
//	emb, err := embeddings.NewOpenAIEmbedder(
//	    os.Getenv("OPENAI_API_KEY"),
//	    embeddings.WithModel("text-embedding-3-small"),
//	)
func NewOpenAIEmbedder(apiKey string, opts ...OpenAIEmbedderOption) (*OpenAIEmbedder, error) {
	if apiKey == "" {
		return nil, errors.New("embeddings: OpenAI API key is required")
	}
	e := &OpenAIEmbedder{
		model:     "text-embedding-3-small",
		batchSize: 512,
	}
	for _, o := range opts {
		o(e)
	}

	reqOpts := []option.RequestOption{option.WithAPIKey(apiKey)}
	if e.baseURL != "" {
		reqOpts = append(reqOpts, option.WithBaseURL(e.baseURL))
	}
	e.client = openai.NewClient(reqOpts...)

	return e, nil
}

// EmbedDocuments embeds a batch of texts, batching requests to stay within
// API limits.
func (e *OpenAIEmbedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	all := make([][]float64, 0, len(texts))
	for i := 0; i < len(texts); i += e.batchSize {
		end := i + e.batchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[i:end]
		embeddings, err := e.embed(ctx, batch)
		if err != nil {
			return nil, err
		}
		all = append(all, embeddings...)
	}
	return all, nil
}

// Close releases idle connections held by the underlying HTTP client.
func (e *OpenAIEmbedder) Close() error { return nil }

// EmbedQuery embeds a single query string.
func (e *OpenAIEmbedder) EmbedQuery(ctx context.Context, text string) ([]float64, error) {
	results, err := e.embed(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, errors.New("embeddings: empty result from OpenAI")
	}
	return results[0], nil
}

func (e *OpenAIEmbedder) embed(ctx context.Context, texts []string) ([][]float64, error) {
	resp, err := e.client.Embeddings.New(ctx, openai.EmbeddingNewParams{
		Input: openai.EmbeddingNewParamsInputUnion{OfArrayOfStrings: texts},
		Model: e.model,
	})
	if err != nil {
		return nil, fmt.Errorf("embeddings: openai: %w", err)
	}
	out := make([][]float64, len(resp.Data))
	for i, d := range resp.Data {
		out[i] = d.Embedding
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// Azure OpenAI Embedder
// ---------------------------------------------------------------------------

// AzureEmbedder uses Azure OpenAI's embedding deployment.
type AzureEmbedder struct {
	client     openai.Client
	deployment string
	batchSize  int
}

// AzureEmbedderOption configures an AzureEmbedder.
type AzureEmbedderOption func(*AzureEmbedder)

func WithAzureBatchSize(n int) AzureEmbedderOption {
	return func(e *AzureEmbedder) { e.batchSize = n }
}

// NewAzureEmbedder creates an Azure OpenAI embedding client.
//
//	emb, err := embeddings.NewAzureEmbedder(
//	    os.Getenv("AZURE_OPENAI_API_KEY"),
//	    os.Getenv("AZURE_OPENAI_ENDPOINT"),
//	    "text-embedding-3-small",  // deployment name
//	    "2024-02-01",              // API version
//	)
func NewAzureEmbedder(apiKey, endpoint, deployment, apiVersion string, opts ...AzureEmbedderOption) (*AzureEmbedder, error) {
	if apiKey == "" {
		return nil, errors.New("embeddings: Azure API key is required")
	}
	if endpoint == "" {
		return nil, errors.New("embeddings: Azure endpoint is required")
	}
	if deployment == "" {
		return nil, errors.New("embeddings: Azure deployment is required")
	}

	e := &AzureEmbedder{
		deployment: deployment,
		batchSize:  512,
	}
	for _, o := range opts {
		o(e)
	}

	e.client = openai.NewClient(
		azure.WithEndpoint(endpoint, apiVersion),
		azure.WithAPIKey(apiKey),
	)

	return e, nil
}

// EmbedDocuments embeds a batch of texts, batching requests.
func (e *AzureEmbedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	all := make([][]float64, 0, len(texts))
	for i := 0; i < len(texts); i += e.batchSize {
		end := i + e.batchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[i:end]
		embeddings, err := e.embed(ctx, batch)
		if err != nil {
			return nil, err
		}
		all = append(all, embeddings...)
	}
	return all, nil
}

// Close releases idle connections held by the underlying HTTP client.
func (e *AzureEmbedder) Close() error { return nil }

// EmbedQuery embeds a single query string.
func (e *AzureEmbedder) EmbedQuery(ctx context.Context, text string) ([]float64, error) {
	results, err := e.embed(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, errors.New("embeddings: empty result from Azure")
	}
	return results[0], nil
}

func (e *AzureEmbedder) embed(ctx context.Context, texts []string) ([][]float64, error) {
	resp, err := e.client.Embeddings.New(ctx, openai.EmbeddingNewParams{
		Input: openai.EmbeddingNewParamsInputUnion{OfArrayOfStrings: texts},
		Model: e.deployment,
	})
	if err != nil {
		return nil, fmt.Errorf("embeddings: azure: %w", err)
	}
	out := make([][]float64, len(resp.Data))
	for i, d := range resp.Data {
		out[i] = d.Embedding
	}
	return out, nil
}
