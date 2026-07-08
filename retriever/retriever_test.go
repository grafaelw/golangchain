package retriever

import (
	"context"
	"testing"

	"github.com/grafaelw/golangchain/schema"
)

func mkDocs(bodies ...string) []schema.Document {
	out := make([]schema.Document, len(bodies))
	for i, b := range bodies {
		out[i] = schema.Document{
			PageContent: b,
			// Use the body as the id so cross-retriever dedup works.
			Metadata: map[string]any{"id": b},
		}
	}
	return out
}

var _ = mkDocsWithIdx

func mkDocsWithIdx(bodies ...string) []schema.Document {
	out := make([]schema.Document, len(bodies))
	for i, b := range bodies {
		out[i] = schema.Document{PageContent: b, Metadata: map[string]any{"id": i}}
	}
	return out
}

func TestBM25Retriever(t *testing.T) {
	docs := mkDocs(
		"the quick brown fox jumps over the lazy dog",
		"a lazy cat sleeps in the sun",
		"quantum entanglement in condensed matter",
		"foxes are cunning and quick",
	)
	r := NewBM25Retriever(docs, 2)
	hits, err := r.GetRelevantDocuments(context.Background(), "quick fox")
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 {
		t.Fatal("no hits")
	}
	// The fox-mentioning docs should rank top-1 and top-2.
	if !contains(hits[0].PageContent, "fox") {
		t.Errorf("top hit unexpected: %q", hits[0].PageContent)
	}
}

type fakeRetriever struct{ docs []schema.Document }

func (f fakeRetriever) GetRelevantDocuments(_ context.Context, _ string) ([]schema.Document, error) {
	return f.docs, nil
}

func TestEnsembleRRF(t *testing.T) {
	a := fakeRetriever{docs: mkDocs("apple", "banana", "cherry")}
	b := fakeRetriever{docs: mkDocs("banana", "date", "apple")}
	e := NewEnsembleRetriever([]Retriever{a, b}, nil, 3)
	hits, err := e.GetRelevantDocuments(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 3 {
		t.Fatalf("expected 3 unique hits, got %d", len(hits))
	}
	// banana appears near the top of both lists → should win.
	if hits[0].PageContent != "banana" {
		t.Errorf("expected banana first, got %q", hits[0].PageContent)
	}
}

func TestKeywordCompressor(t *testing.T) {
	docs := mkDocs("Go is a programming language", "cats are furry", "python is a language")
	c := KeywordCompressor{}
	got, err := c.Compress(context.Background(), docs, "language")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 matches, got %d", len(got))
	}
}

func contains(s, sub string) bool {
	return len(sub) == 0 || (len(s) >= len(sub) && indexOf(s, sub) >= 0)
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
