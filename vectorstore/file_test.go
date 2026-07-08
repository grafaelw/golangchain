package vectorstore

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/grafaelw/golangchain/schema"
)

// fakeEmbedder is a deterministic bag-of-chars embedder for tests.
type fakeEmbedder struct{}

func (fakeEmbedder) EmbedDocuments(_ context.Context, texts []string) ([][]float64, error) {
	out := make([][]float64, len(texts))
	for i, t := range texts {
		out[i] = embed(t)
	}
	return out, nil
}
func (fakeEmbedder) EmbedQuery(_ context.Context, text string) ([]float64, error) {
	return embed(text), nil
}
func embed(s string) []float64 {
	v := make([]float64, 26)
	for _, r := range s {
		if r >= 'a' && r <= 'z' {
			v[r-'a']++
		}
	}
	return v
}

func TestFileVectorStoreRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "index.json")

	store, err := NewFileVectorStore(path, fakeEmbedder{})
	if err != nil {
		t.Fatal(err)
	}
	docs := []schema.Document{
		{PageContent: "apple banana", Metadata: map[string]any{"id": "1"}},
		{PageContent: "cherry date", Metadata: map[string]any{"id": "2"}},
	}
	if err := store.AddDocuments(context.Background(), docs); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("file not written: %v", err)
	}

	// Re-open — no re-embed required.
	store2, err := NewFileVectorStore(path, fakeEmbedder{})
	if err != nil {
		t.Fatal(err)
	}
	if store2.Len() != 2 {
		t.Fatalf("expected 2 entries after reload, got %d", store2.Len())
	}
	hits, err := store2.SimilaritySearch(context.Background(), "apple", 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].PageContent != "apple banana" {
		t.Fatalf("unexpected hits: %#v", hits)
	}
}
