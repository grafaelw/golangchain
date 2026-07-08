// Package documentloader — see doc.go for the package overview.
package documentloader

import (
	"context"
	"encoding/csv"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// Loader interface
// ---------------------------------------------------------------------------

// Loader converts a source into one or more Documents.
type Loader interface {
	Load(ctx context.Context) ([]schema.Document, error)
}

// ---------------------------------------------------------------------------
// TextLoader
// ---------------------------------------------------------------------------

// TextLoader reads a single UTF-8 text file into one Document.
type TextLoader struct {
	Path     string
	Encoding string // reserved; only UTF-8 supported today
}

// NewTextLoader creates a TextLoader for the given path.
func NewTextLoader(path string) *TextLoader { return &TextLoader{Path: path} }

func (l *TextLoader) Load(_ context.Context) ([]schema.Document, error) {
	data, err := os.ReadFile(l.Path)
	if err != nil {
		return nil, fmt.Errorf("documentloader: read %q: %w", l.Path, err)
	}
	return []schema.Document{{
		PageContent: string(data),
		Metadata: map[string]any{
			"source": l.Path,
			"mime":   "text/plain",
			"size":   len(data),
		},
	}}, nil
}

// ---------------------------------------------------------------------------
// MarkdownLoader
// ---------------------------------------------------------------------------

// MarkdownLoader reads a Markdown file as text.
// Pair with textsplitter.NewMarkdownSplitter for structural chunking.
type MarkdownLoader struct{ Path string }

// NewMarkdownLoader creates a MarkdownLoader.
func NewMarkdownLoader(path string) *MarkdownLoader { return &MarkdownLoader{Path: path} }

func (l *MarkdownLoader) Load(ctx context.Context) ([]schema.Document, error) {
	docs, err := (&TextLoader{Path: l.Path}).Load(ctx)
	if err != nil {
		return nil, err
	}
	for i := range docs {
		docs[i].Metadata["mime"] = "text/markdown"
	}
	return docs, nil
}

// ---------------------------------------------------------------------------
// CSVLoader
// ---------------------------------------------------------------------------

// CSVLoader loads a CSV file, emitting one Document per row. Column names
// from the header row are used as metadata keys; the concatenated
// "key: value" pairs form the PageContent. Set ContentColumn to emit only
// that column as content instead.
type CSVLoader struct {
	Path          string
	Delimiter     rune   // default ','
	ContentColumn string // if set, PageContent is that column's value only
}

// NewCSVLoader creates a CSVLoader with default delimiter ','.
func NewCSVLoader(path string) *CSVLoader { return &CSVLoader{Path: path, Delimiter: ','} }

func (l *CSVLoader) Load(_ context.Context) ([]schema.Document, error) {
	f, err := os.Open(l.Path)
	if err != nil {
		return nil, fmt.Errorf("documentloader: open %q: %w", l.Path, err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	if l.Delimiter != 0 {
		r.Comma = l.Delimiter
	}
	rows, err := r.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("documentloader: csv %q: %w", l.Path, err)
	}
	if len(rows) < 2 {
		return nil, nil
	}
	header := rows[0]
	docs := make([]schema.Document, 0, len(rows)-1)
	for i, row := range rows[1:] {
		md := map[string]any{
			"source": l.Path,
			"mime":   "text/csv",
			"row":    i,
		}
		var content strings.Builder
		for j, v := range row {
			key := fmt.Sprintf("col_%d", j)
			if j < len(header) {
				key = header[j]
			}
			md[key] = v
			if l.ContentColumn == "" {
				fmt.Fprintf(&content, "%s: %s\n", key, v)
			} else if key == l.ContentColumn {
				content.WriteString(v)
			}
		}
		docs = append(docs, schema.Document{
			PageContent: strings.TrimRight(content.String(), "\n"),
			Metadata:    md,
		})
	}
	return docs, nil
}

// ---------------------------------------------------------------------------
// HTMLLoader
// ---------------------------------------------------------------------------

// HTMLLoader loads an HTML file and strips tags, keeping only visible text
// (drops <script>, <style>, <head>, <noscript>). The <title> element, if
// present, is captured into metadata.
type HTMLLoader struct{ Path string }

// NewHTMLLoader creates an HTMLLoader.
func NewHTMLLoader(path string) *HTMLLoader { return &HTMLLoader{Path: path} }

func (l *HTMLLoader) Load(_ context.Context) ([]schema.Document, error) {
	data, err := os.ReadFile(l.Path)
	if err != nil {
		return nil, fmt.Errorf("documentloader: read %q: %w", l.Path, err)
	}
	text, title := stripHTML(string(data))
	md := map[string]any{
		"source": l.Path,
		"mime":   "text/html",
		"size":   len(data),
	}
	if title != "" {
		md["title"] = title
	}
	return []schema.Document{{PageContent: text, Metadata: md}}, nil
}

// stripHTML removes tags and script/style/head blocks. It is intentionally
// dependency-free; for production-grade extraction consider integrating a
// dedicated parser via a Loader that delegates to it.
func stripHTML(src string) (text, title string) {
	// Capture <title> BEFORE dropping <head>.
	lowerAll := strings.ToLower(src)
	if i := strings.Index(lowerAll, "<title>"); i >= 0 {
		if j := strings.Index(lowerAll[i:], "</title>"); j > 0 {
			title = strings.TrimSpace(src[i+len("<title>") : i+j])
		}
	}
	lower := lowerAll
	for _, tag := range []string{"script", "style", "head", "noscript"} {
		src, lower = stripBlock(src, lower, "<"+tag, "</"+tag+">")
	}

	// Strip remaining tags.
	var b strings.Builder
	inTag := false
	for _, r := range src {
		switch {
		case r == '<':
			inTag = true
		case r == '>':
			inTag = false
			b.WriteByte(' ')
		case !inTag:
			b.WriteRune(r)
		}
	}
	// Collapse whitespace.
	return collapseSpace(b.String()), title
}

func stripBlock(src, lower, open, close string) (string, string) {
	for {
		i := strings.Index(lower, open)
		if i < 0 {
			return src, lower
		}
		j := strings.Index(lower[i:], close)
		if j < 0 {
			// unterminated — drop the rest
			return src[:i], lower[:i]
		}
		end := i + j + len(close)
		src = src[:i] + src[end:]
		lower = lower[:i] + lower[end:]
	}
}

func collapseSpace(s string) string {
	var b strings.Builder
	space := false
	for _, r := range s {
		if r == ' ' || r == '\t' || r == '\r' || r == '\n' {
			if !space {
				b.WriteByte(' ')
				space = true
			}
			continue
		}
		space = false
		b.WriteRune(r)
	}
	return strings.TrimSpace(b.String())
}

// ---------------------------------------------------------------------------
// HTTPLoader
// ---------------------------------------------------------------------------

// HTTPLoader fetches a URL and returns one Document. The MIME type of the
// response drives whether the body is stripped as HTML or kept as text.
type HTTPLoader struct {
	URL     string
	Client  *http.Client
	Headers map[string]string
}

// NewHTTPLoader creates an HTTPLoader with a 30-second default client.
func NewHTTPLoader(u string) *HTTPLoader {
	return &HTTPLoader{URL: u, Client: &http.Client{Timeout: 30 * time.Second}}
}

func (l *HTTPLoader) Load(ctx context.Context) ([]schema.Document, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, l.URL, nil)
	if err != nil {
		return nil, fmt.Errorf("documentloader: build request: %w", err)
	}
	for k, v := range l.Headers {
		req.Header.Set(k, v)
	}
	resp, err := l.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("documentloader: GET %q: %w", l.URL, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("documentloader: GET %q: status %d", l.URL, resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("documentloader: read body: %w", err)
	}
	mime := resp.Header.Get("Content-Type")
	content := string(body)
	title := ""
	if strings.Contains(mime, "html") {
		content, title = stripHTML(content)
	}
	md := map[string]any{
		"source": l.URL,
		"mime":   mime,
		"size":   len(body),
		"status": resp.StatusCode,
	}
	if title != "" {
		md["title"] = title
	}
	return []schema.Document{{PageContent: content, Metadata: md}}, nil
}

// ---------------------------------------------------------------------------
// DirectoryLoader
// ---------------------------------------------------------------------------

// DirectoryLoader walks a directory and dispatches each file to a loader
// selected by extension. Unknown extensions are ignored unless a Fallback
// loader factory is configured.
type DirectoryLoader struct {
	Root      string
	Glob      string // optional glob pattern for filepath.Match against the base name
	Recursive bool
	// Handlers maps a lowercase extension (e.g. ".txt") to a factory that
	// builds a Loader for the file. The default handler set covers .txt,
	// .md, .markdown, .csv, .html, .htm.
	Handlers map[string]func(path string) Loader
	// Fallback is used when no handler matches. If nil, unknown files are skipped.
	Fallback func(path string) Loader
}

// NewDirectoryLoader constructs a DirectoryLoader with the default handler set.
func NewDirectoryLoader(root string) *DirectoryLoader {
	return &DirectoryLoader{
		Root:      root,
		Recursive: true,
		Handlers: map[string]func(string) Loader{
			".txt":      func(p string) Loader { return NewTextLoader(p) },
			".md":       func(p string) Loader { return NewMarkdownLoader(p) },
			".markdown": func(p string) Loader { return NewMarkdownLoader(p) },
			".csv":      func(p string) Loader { return NewCSVLoader(p) },
			".html":     func(p string) Loader { return NewHTMLLoader(p) },
			".htm":      func(p string) Loader { return NewHTMLLoader(p) },
		},
	}
}

func (l *DirectoryLoader) Load(ctx context.Context) ([]schema.Document, error) {
	var out []schema.Document
	walker := func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			if path != l.Root && !l.Recursive {
				return filepath.SkipDir
			}
			return nil
		}
		if l.Glob != "" {
			match, err := filepath.Match(l.Glob, filepath.Base(path))
			if err != nil || !match {
				return err
			}
		}
		ext := strings.ToLower(filepath.Ext(path))
		factory, ok := l.Handlers[ext]
		if !ok {
			if l.Fallback == nil {
				return nil
			}
			factory = l.Fallback
		}
		docs, err := factory(path).Load(ctx)
		if err != nil {
			return fmt.Errorf("documentloader: %s: %w", path, err)
		}
		out = append(out, docs...)
		return nil
	}
	return out, filepath.Walk(l.Root, walker)
}
