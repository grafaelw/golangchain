package textsplitter

import (
	"strings"
	"testing"

	"github.com/grafaelw/golangchain/schema"
)

func TestCharacterSplitter(t *testing.T) {
	s := NewCharacterSplitter("\n\n", WithChunkSize(20), WithChunkOverlap(0))
	chunks := s.SplitText("aaaa\n\nbbbb\n\ncccc\n\ndddd")
	if len(chunks) == 0 {
		t.Fatal("expected chunks")
	}
	for _, c := range chunks {
		if RuneLen(c) > 20 {
			t.Errorf("chunk exceeds size: %q (%d)", c, RuneLen(c))
		}
	}
}

func TestRecursiveSplitterShrinksHugeText(t *testing.T) {
	long := strings.Repeat("word ", 500) // 2500 chars
	s := NewRecursiveCharacterSplitter(nil, WithChunkSize(120), WithChunkOverlap(20))
	chunks := s.SplitText(long)
	if len(chunks) < 5 {
		t.Fatalf("expected many chunks, got %d", len(chunks))
	}
	for i, c := range chunks {
		if RuneLen(c) > 140 { // chunk size + a little slack for kept separators
			t.Errorf("chunk %d too big: %d chars", i, RuneLen(c))
		}
	}
}

func TestMarkdownSplitterPrefersHeadings(t *testing.T) {
	md := "# Title\n\nintro paragraph.\n\n## Section\n\nbody one.\n\n## Another\n\nbody two."
	s := NewMarkdownSplitter(WithChunkSize(40), WithChunkOverlap(0))
	chunks := s.SplitText(md)
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d: %#v", len(chunks), chunks)
	}
}

func TestSplitDocumentsPreservesMetadata(t *testing.T) {
	s := NewRecursiveCharacterSplitter(nil, WithChunkSize(20), WithChunkOverlap(0))
	docs := s.SplitDocuments([]schema.Document{{
		PageContent: "aaaaa bbbbb ccccc ddddd eeeee fffff",
		Metadata:    map[string]any{"source": "test.txt"},
	}})
	if len(docs) < 2 {
		t.Fatalf("expected >=2 chunks, got %d", len(docs))
	}
	for i, d := range docs {
		if d.Metadata["source"] != "test.txt" {
			t.Errorf("chunk %d lost source metadata", i)
		}
		if _, ok := d.Metadata["chunk"]; !ok {
			t.Errorf("chunk %d missing chunk index", i)
		}
	}
}
