package azureaisearch

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
	"github.com/grafaelw/golangchain/schema"
	"github.com/grafaelw/golangchain/vectorstore"
)

const defaultAPIVersion = "2024-07-01"

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

// Option configures the Azure AI Search store.
type Option func(*Store)

// WithIndexName sets the search index name (default "documents").
func WithIndexName(name string) Option {
	return func(s *Store) { s.indexName = name }
}

// WithDimensions sets the expected vector dimensions (default 1536).
func WithDimensions(dim int) Option {
	return func(s *Store) { s.dimensions = dim }
}

// WithAPIVersion sets the Azure Search REST API version.
func WithAPIVersion(v string) Option {
	return func(s *Store) { s.apiVersion = v }
}

// WithHTTPClient overrides the HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(s *Store) { s.client = client }
}

// WithSemanticConfig sets the semantic configuration name for hybrid search.
func WithSemanticConfig(name string) Option {
	return func(s *Store) { s.semanticConfig = name }
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

// Store implements vectorstore.VectorStore backed by Azure AI Search.
type Store struct {
	endpoint       string
	apiKey         string
	indexName      string
	apiVersion     string
	dimensions     int
	semanticConfig string
	client         *http.Client
	embedder       embeddings.Embedder
	created        bool
}

// New creates an Azure AI Search-backed vector store.
//
//	store, err := azureaisearch.New(
//	    "https://myservice.search.windows.net",
//	    "ADMIN_API_KEY",
//	    embedder,
//	    azureaisearch.WithIndexName("myindex"),
//	)
func New(endpoint, apiKey string, embedder embeddings.Embedder, opts ...Option) (*Store, error) {
	if endpoint == "" {
		return nil, errors.New("azureaisearch: endpoint is required")
	}
	if apiKey == "" {
		return nil, errors.New("azureaisearch: API key is required")
	}
	if embedder == nil {
		return nil, errors.New("azureaisearch: embedder is required")
	}
	s := &Store{
		endpoint:   strings.TrimRight(endpoint, "/"),
		apiKey:     apiKey,
		indexName:  "documents",
		apiVersion: defaultAPIVersion,
		dimensions: 1536,
		embedder:   embedder,
		client:     &http.Client{Timeout: 60 * time.Second},
	}
	for _, o := range opts {
		o(s)
	}
	return s, nil
}

var _ vectorstore.VectorStore = (*Store)(nil)

// ---------------------------------------------------------------------------
// VectorStore implementation
// ---------------------------------------------------------------------------

// AddDocuments embeds and indexes documents into Azure AI Search.
func (s *Store) AddDocuments(ctx context.Context, docs []schema.Document) error {
	if len(docs) == 0 {
		return nil
	}
	if err := s.ensureIndex(ctx); err != nil {
		return err
	}

	texts := make([]string, len(docs))
	for i, d := range docs {
		texts[i] = d.PageContent
	}
	vectors, err := s.embedder.EmbedDocuments(ctx, texts)
	if err != nil {
		return fmt.Errorf("azureaisearch: embed: %w", err)
	}
	if len(vectors) != len(docs) {
		return fmt.Errorf("azureaisearch: expected %d vectors, got %d", len(docs), len(vectors))
	}

	type docAction struct {
		ID            string         `json:"id"`
		Content       string         `json:"content"`
		ContentVector []float64      `json:"contentVector"`
		Metadata      map[string]any `json:"metadata,omitempty"`
	}
	type indexAction struct {
		Action string    `json:"action"`
		Doc    docAction `json:"doc"`
	}

	actions := make([]indexAction, len(docs))
	for i, doc := range docs {
		id := fmt.Sprintf("doc_%d", i)
		if v, ok := doc.Metadata["id"]; ok {
			id = fmt.Sprint(v)
		}
		md := make(map[string]any)
		for k, v := range doc.Metadata {
			md[k] = v
		}
		actions[i] = indexAction{
			Action: "upload",
			Doc: docAction{
				ID:            id,
				Content:       doc.PageContent,
				ContentVector: vectors[i],
				Metadata:      md,
			},
		}
	}

	body, _ := json.Marshal(map[string]any{"value": actions})
	resp, err := s.request(ctx, http.MethodPost, "/indexes/"+s.indexName+"/docs/index", body)
	if err != nil {
		return fmt.Errorf("azureaisearch: index: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("azureaisearch: index: status %d: %s", resp.StatusCode, string(b))
	}
	return nil
}

// SimilaritySearch embeds the query and returns the k nearest documents.
func (s *Store) SimilaritySearch(ctx context.Context, query string, k int) ([]schema.Document, error) {
	vec, err := s.embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("azureaisearch: embed query: %w", err)
	}
	return s.SimilaritySearchByVector(ctx, vec, k)
}

// SimilaritySearchByVector returns the k nearest documents to a query vector.
func (s *Store) SimilaritySearchByVector(ctx context.Context, vector []float64, k int) ([]schema.Document, error) {
	if err := s.ensureIndex(ctx); err != nil {
		return nil, err
	}

	body, _ := json.Marshal(map[string]any{
		"search": "*",
		"vectorQueries": []map[string]any{
			{
				"kind":   "vector",
				"vector": vector,
				"k":      k,
				"fields": "contentVector",
			},
		},
		"select": "id,content,metadata",
	})
	resp, err := s.request(ctx, http.MethodPost, "/indexes/"+s.indexName+"/docs/search", body)
	if err != nil {
		return nil, fmt.Errorf("azureaisearch: search: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("azureaisearch: search: status %d: %s", resp.StatusCode, string(b))
	}

	var result struct {
		Docs []struct {
			ID       string         `json:"id"`
			Content  string         `json:"content"`
			Metadata map[string]any `json:"metadata"`
			Score    float64        `json:"@search.score"`
		} `json:"value"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("azureaisearch: search: decode: %w", err)
	}

	docs := make([]schema.Document, 0, len(result.Docs))
	for _, d := range result.Docs {
		docs = append(docs, schema.Document{
			PageContent: d.Content,
			Score:       d.Score,
			Metadata:    d.Metadata,
		})
	}
	return docs, nil
}

// Delete removes documents by ID from Azure AI Search.
func (s *Store) Delete(ctx context.Context, ids []string) error {
	if err := s.ensureIndex(ctx); err != nil {
		return err
	}

	type delAction struct {
		Action string `json:"action"`
		ID     string `json:"id"`
	}
	actions := make([]delAction, len(ids))
	for i, id := range ids {
		actions[i] = delAction{Action: "delete", ID: id}
	}

	body, _ := json.Marshal(map[string]any{"value": actions})
	resp, err := s.request(ctx, http.MethodPost, "/indexes/"+s.indexName+"/docs/index", body)
	if err != nil {
		return fmt.Errorf("azureaisearch: delete: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("azureaisearch: delete: status %d: %s", resp.StatusCode, string(b))
	}
	return nil
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

func (s *Store) ensureIndex(ctx context.Context) error {
	if s.created {
		return nil
	}

	// Check if index already exists.
	checkURL := fmt.Sprintf("/indexes/%s", s.indexName)
	resp, err := s.request(ctx, http.MethodGet, checkURL, nil)
	if err != nil {
		return fmt.Errorf("azureaisearch: check index: %w", err)
	}
	if resp.StatusCode == http.StatusOK {
		_ = resp.Body.Close()
		s.created = true
		return nil
	}
	_ = resp.Body.Close()

	indexSchema := map[string]any{
		"name": s.indexName,
		"fields": []map[string]any{
			{"name": "id", "type": "Edm.String", "key": true, "filterable": true},
			{"name": "content", "type": "Edm.String", "searchable": true, "filterable": true},
			{"name": "contentVector", "type": "Collection(Edm.Single)", "dimensions": s.dimensions, "vectorSearchProfile": "default", "searchable": true},
			{"name": "metadata", "type": "Edm.String", "searchable": false, "filterable": true},
		},
		"vectorSearch": map[string]any{
			"algorithms": []map[string]any{
				{
					"name": "default",
					"kind": "hnsw",
					"hnswParameters": map[string]any{
						"metric": "cosine",
					},
				},
			},
			"profiles": []map[string]any{
				{
					"name":      "default",
					"algorithm": "default",
				},
			},
		},
	}

	if s.semanticConfig != "" {
		indexSchema["semantic"] = map[string]any{
			"configurations": []map[string]any{
				{
					"name": s.semanticConfig,
					"prioritizedFields": map[string]any{
						"titleField": nil,
						"prioritizedContentFields": []map[string]any{
							{"fieldName": "content"},
						},
						"prioritizedKeywordsFields": []map[string]any{},
					},
				},
			},
		}
	}

	body, _ := json.Marshal(indexSchema)
	resp, err = s.request(ctx, http.MethodPut, "/indexes/"+s.indexName, body)
	if err != nil {
		return fmt.Errorf("azureaisearch: create index: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 400 && resp.StatusCode != http.StatusConflict {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("azureaisearch: create index %q: status %d: %s", s.indexName, resp.StatusCode, string(b))
	}
	s.created = true
	return nil
}

func (s *Store) request(ctx context.Context, method, path string, body []byte) (*http.Response, error) {
	url := s.endpoint + path + "?api-version=" + s.apiVersion
	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("api-key", s.apiKey)
	req.Header.Set("Content-Type", "application/json")
	if body != nil {
		req.Header.Set("Content-Length", fmt.Sprintf("%d", len(body)))
	}
	return s.client.Do(req)
}

// Close closes the HTTP transport's idle connections.
func (s *Store) Close() error {
	s.client.CloseIdleConnections()
	return nil
}
