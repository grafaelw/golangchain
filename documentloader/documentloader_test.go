package documentloader

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestTextAndDirectoryLoader(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "b.md"), []byte("# heading\n\ntext"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "ignored.bin"), []byte("xx"), 0o644); err != nil {
		t.Fatal(err)
	}

	docs, err := NewDirectoryLoader(dir).Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 2 {
		t.Fatalf("expected 2 docs, got %d: %#v", len(docs), docs)
	}
}

func TestCSVLoader(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "data.csv")
	if err := os.WriteFile(path, []byte("name,age\nAlice,30\nBob,25\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	docs, err := NewCSVLoader(path).Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 2 {
		t.Fatalf("expected 2 rows, got %d", len(docs))
	}
	if docs[0].Metadata["name"] != "Alice" {
		t.Fatalf("metadata missing: %#v", docs[0].Metadata)
	}
}

func TestHTMLLoaderStripsTags(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "page.html")
	body := `<html><head><title>My Page</title><style>x{}</style></head><body><script>evil()</script><p>Hello <b>world</b></p></body></html>`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	docs, err := NewHTMLLoader(path).Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 doc, got %d", len(docs))
	}
	if title, _ := docs[0].Metadata["title"].(string); title != "My Page" {
		t.Errorf("title not extracted: %q", title)
	}
	if got := docs[0].PageContent; got == "" || contains(got, "evil()") || contains(got, "<p>") {
		t.Errorf("bad content: %q", got)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
