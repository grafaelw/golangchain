package vectorstore_test

import (
	"context"
	"testing"

	"github.com/grafaelw/golangchain/schema"
	"github.com/grafaelw/golangchain/vectorstore"
)

// ---------------------------------------------------------------------------
// Fake embedder — deterministic, no API calls
// ---------------------------------------------------------------------------

// fakeEmbedder assigns each unique text a fixed vector based on a lookup table.
// If the text isn't in the table it falls back to an all-zeros vector.
type fakeEmbedder struct {
	table map[string][]float64
	dim   int
}

func newFakeEmbedder(dim int) *fakeEmbedder {
	return &fakeEmbedder{table: make(map[string][]float64), dim: dim}
}

func (f *fakeEmbedder) set(text string, vec []float64) {
	f.table[text] = vec
}

func (f *fakeEmbedder) EmbedDocuments(_ context.Context, texts []string) ([][]float64, error) {
	out := make([][]float64, len(texts))
	for i, t := range texts {
		if v, ok := f.table[t]; ok {
			out[i] = v
		} else {
			out[i] = make([]float64, f.dim)
		}
	}
	return out, nil
}

func (f *fakeEmbedder) EmbedQuery(_ context.Context, text string) ([]float64, error) {
	if v, ok := f.table[text]; ok {
		return v, nil
	}
	return make([]float64, f.dim), nil
}

// ---------------------------------------------------------------------------
// Helper: unit vector in direction of index i
// ---------------------------------------------------------------------------

func unitVec(dim, i int) []float64 {
	v := make([]float64, dim)
	v[i] = 1.0
	return v
}

// ---------------------------------------------------------------------------
// AddDocuments
// ---------------------------------------------------------------------------

func TestAddDocuments_Empty(t *testing.T) {
	e := newFakeEmbedder(4)
	store := vectorstore.NewInMemoryVectorStore(e)
	if err := store.AddDocuments(context.Background(), nil); err != nil {
		t.Fatalf("AddDocuments(nil): %v", err)
	}
	if store.Len() != 0 {
		t.Errorf("Len: want 0, got %d", store.Len())
	}
}

func TestAddDocuments_Single(t *testing.T) {
	e := newFakeEmbedder(4)
	e.set("doc1", unitVec(4, 0))
	store := vectorstore.NewInMemoryVectorStore(e)

	err := store.AddDocuments(context.Background(), []schema.Document{
		{PageContent: "doc1", Metadata: map[string]any{"id": "d1"}},
	})
	if err != nil {
		t.Fatalf("AddDocuments: %v", err)
	}
	if store.Len() != 1 {
		t.Errorf("Len: want 1, got %d", store.Len())
	}
}

func TestAddDocuments_Multiple(t *testing.T) {
	dim := 4
	e := newFakeEmbedder(dim)
	docs := []schema.Document{
		{PageContent: "golang programming"},
		{PageContent: "python scripting"},
		{PageContent: "rust systems"},
	}
	for _, d := range docs {
		e.set(d.PageContent, unitVec(dim, 0)) // all same direction for simplicity
	}
	store := vectorstore.NewInMemoryVectorStore(e)
	if err := store.AddDocuments(context.Background(), docs); err != nil {
		t.Fatalf("AddDocuments: %v", err)
	}
	if store.Len() != 3 {
		t.Errorf("Len: want 3, got %d", store.Len())
	}
}

// ---------------------------------------------------------------------------
// SimilaritySearch
// ---------------------------------------------------------------------------

func TestSimilaritySearch_TopK(t *testing.T) {
	dim := 4
	e := newFakeEmbedder(dim)

	// Three orthogonal documents; query matches doc0
	e.set("doc0", unitVec(dim, 0)) // [1,0,0,0]
	e.set("doc1", unitVec(dim, 1)) // [0,1,0,0]
	e.set("doc2", unitVec(dim, 2)) // [0,0,1,0]
	e.set("query", unitVec(dim, 0)) // identical to doc0

	store := vectorstore.NewInMemoryVectorStore(e)
	for _, txt := range []string{"doc0", "doc1", "doc2"} {
		store.AddDocuments(context.Background(), []schema.Document{{PageContent: txt}})
	}

	results, err := store.SimilaritySearch(context.Background(), "query", 2)
	if err != nil {
		t.Fatalf("SimilaritySearch: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("want 2 results, got %d", len(results))
	}
	// doc0 should be ranked first (cosine = 1.0)
	if results[0].PageContent != "doc0" {
		t.Errorf("top result should be doc0, got %q", results[0].PageContent)
	}
	if results[0].Score < 0.99 {
		t.Errorf("score should be ~1.0, got %f", results[0].Score)
	}
}

func TestSimilaritySearch_EmptyStore(t *testing.T) {
	e := newFakeEmbedder(4)
	e.set("q", unitVec(4, 0))
	store := vectorstore.NewInMemoryVectorStore(e)

	results, err := store.SimilaritySearch(context.Background(), "q", 5)
	if err != nil {
		t.Fatalf("SimilaritySearch on empty store: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("want 0 results from empty store, got %d", len(results))
	}
}

func TestSimilaritySearch_KGreaterThanDocs(t *testing.T) {
	dim := 4
	e := newFakeEmbedder(dim)
	e.set("a", unitVec(dim, 0))
	e.set("q", unitVec(dim, 0))
	store := vectorstore.NewInMemoryVectorStore(e)
	store.AddDocuments(context.Background(), []schema.Document{{PageContent: "a"}})

	// Request 10 but only 1 doc exists
	results, err := store.SimilaritySearch(context.Background(), "q", 10)
	if err != nil {
		t.Fatalf("SimilaritySearch: %v", err)
	}
	if len(results) != 1 {
		t.Errorf("want 1 result (clamped), got %d", len(results))
	}
}

// ---------------------------------------------------------------------------
// SimilaritySearchByVector
// ---------------------------------------------------------------------------

func TestSimilaritySearchByVector(t *testing.T) {
	dim := 4
	e := newFakeEmbedder(dim)
	e.set("vec_doc", unitVec(dim, 3)) // [0,0,0,1]
	store := vectorstore.NewInMemoryVectorStore(e)
	store.AddDocuments(context.Background(), []schema.Document{{PageContent: "vec_doc"}})

	// Query with exact same vector
	queryVec := unitVec(dim, 3)
	results, err := store.SimilaritySearchByVector(context.Background(), queryVec, 1)
	if err != nil {
		t.Fatalf("SimilaritySearchByVector: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("want 1 result, got %d", len(results))
	}
	if results[0].Score < 0.99 {
		t.Errorf("expected score ~1.0, got %f", results[0].Score)
	}
}

// ---------------------------------------------------------------------------
// Delete
// ---------------------------------------------------------------------------

func TestDelete_RemovesDocument(t *testing.T) {
	dim := 4
	e := newFakeEmbedder(dim)
	e.set("keep_doc", unitVec(dim, 0))
	e.set("delete_doc", unitVec(dim, 1))
	e.set("query", unitVec(dim, 1)) // matches delete_doc direction

	store := vectorstore.NewInMemoryVectorStore(e)
	store.AddDocuments(context.Background(), []schema.Document{
		{PageContent: "keep_doc", Metadata: map[string]any{"id": "keep"}},
		{PageContent: "delete_doc", Metadata: map[string]any{"id": "del"}},
	})

	if err := store.Delete(context.Background(), []string{"del"}); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if store.Len() != 1 {
		t.Errorf("Len after delete: want 1, got %d", store.Len())
	}

	// The deleted doc should not appear in results
	results, _ := store.SimilaritySearch(context.Background(), "query", 5)
	for _, r := range results {
		if r.PageContent == "delete_doc" {
			t.Error("deleted document still appears in results")
		}
	}
}

func TestDelete_NonexistentID(t *testing.T) {
	dim := 4
	e := newFakeEmbedder(dim)
	e.set("doc", unitVec(dim, 0))
	store := vectorstore.NewInMemoryVectorStore(e)
	store.AddDocuments(context.Background(), []schema.Document{
		{PageContent: "doc", Metadata: map[string]any{"id": "d1"}},
	})

	// Deleting a nonexistent ID should not error or corrupt the store
	if err := store.Delete(context.Background(), []string{"nonexistent"}); err != nil {
		t.Fatalf("Delete nonexistent: %v", err)
	}
	if store.Len() != 1 {
		t.Errorf("Len: should still be 1, got %d", store.Len())
	}
}

// ---------------------------------------------------------------------------
// Cosine similarity edge cases
// ---------------------------------------------------------------------------

func TestSimilaritySearch_ZeroVector(t *testing.T) {
	dim := 4
	e := newFakeEmbedder(dim)
	// Zero vector embedding — should not cause division-by-zero
	e.set("zero_doc", make([]float64, dim))
	e.set("zero_q", make([]float64, dim))
	store := vectorstore.NewInMemoryVectorStore(e)
	store.AddDocuments(context.Background(), []schema.Document{{PageContent: "zero_doc"}})

	// Should not panic
	results, err := store.SimilaritySearch(context.Background(), "zero_q", 1)
	if err != nil {
		t.Fatalf("SimilaritySearch with zero vector: %v", err)
	}
	_ = results // zero vectors have similarity 0 — that's fine
}

func TestSimilaritySearch_ScoreOrdering(t *testing.T) {
	dim := 3
	e := newFakeEmbedder(dim)
	// doc_a is perfectly aligned with query, doc_b is orthogonal
	e.set("doc_a", []float64{1, 0, 0})
	e.set("doc_b", []float64{0, 1, 0})
	e.set("doc_c", []float64{0, 0, 1})
	e.set("query", []float64{1, 0, 0}) // aligns with doc_a

	store := vectorstore.NewInMemoryVectorStore(e)
	for _, txt := range []string{"doc_a", "doc_b", "doc_c"} {
		store.AddDocuments(context.Background(), []schema.Document{{PageContent: txt}})
	}

	results, err := store.SimilaritySearch(context.Background(), "query", 3)
	if err != nil {
		t.Fatalf("SimilaritySearch: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("want 3 results, got %d", len(results))
	}
	// Results should be ordered by descending score
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("results not sorted: [%d].Score=%f > [%d].Score=%f",
				i, results[i].Score, i-1, results[i-1].Score)
		}
	}
	if results[0].PageContent != "doc_a" {
		t.Errorf("top result should be doc_a, got %q", results[0].PageContent)
	}
}
