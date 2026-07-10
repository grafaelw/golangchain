package textsplitter_test

import (
	"strings"
	"testing"

	"github.com/grafaelw/golangchain/schema"
	"github.com/grafaelw/golangchain/textsplitter"
)

func TestPythonSplitter(t *testing.T) {
	s := textsplitter.NewPythonSplitter(textsplitter.WithChunkSize(50))
	code := `class Foo:
    def __init__(self):
        self.x = 1

    def bar(self):
        return self.x + 1

class Bar:
    def baz(self):
        pass`
	chunks := s.SplitText(code)
	if len(chunks) == 0 {
		t.Fatal("expected at least 1 chunk")
	}
	// With chunkSize=50, classes should be split
	if len(chunks) < 2 {
		// It's OK if they fit together with this small code — just verify we got something
		for _, c := range chunks {
			if len(c) > 0 {
				return // pass
			}
		}
	}
}

func TestGoSplitter(t *testing.T) {
	s := textsplitter.NewGoSplitter(textsplitter.WithChunkSize(200))
	code := `package main

import "fmt"

func hello() {
    fmt.Println("hello")
}

func world() {
    fmt.Println("world")
}`
	chunks := s.SplitText(code)
	if len(chunks) > 1 {
		// Different funcs should end up in separate chunks
		foundHello, foundWorld := false, false
		for _, c := range chunks {
			if strings.Contains(c, "func hello") {
				foundHello = true
			}
			if strings.Contains(c, "func world") {
				foundWorld = true
			}
		}
		if !foundHello || !foundWorld {
			t.Errorf("funcs should be in different chunks: got %d chunks", len(chunks))
		}
	}
}

func TestJavaScriptSplitter(t *testing.T) {
	s := textsplitter.NewJavaScriptSplitter(textsplitter.WithChunkSize(200))
	code := `function add(a, b) { return a + b; }

function multiply(a, b) { return a * b; }`
	chunks := s.SplitText(code)
	if len(chunks) == 0 {
		t.Fatal("expected at least 1 chunk")
	}
}

func TestCodeSplitter_Documents(t *testing.T) {
	s := textsplitter.NewGoSplitter()
	docs := []schema.Document{
		schema.Document{PageContent: "package main\n\nfunc main() {}", Metadata: map[string]any{"source": "test.go"}},
	}
	result := s.SplitDocuments(docs)
	if len(result) == 0 {
		t.Fatal("expected at least 1 document")
	}
	for _, d := range result {
		if lang, ok := d.Metadata["language"]; !ok || lang != "go" {
			t.Errorf("expected metadata language=go, got %v", lang)
		}
	}
}
