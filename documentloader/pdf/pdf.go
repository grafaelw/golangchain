// Package pdf provides a Loader that extracts plain text from PDF files.
package pdf

import (
	"context"
	"fmt"
	"strings"

	pdflib "github.com/ledongthuc/pdf"

	"github.com/grafaelw/golangchain/documentloader"
	"github.com/grafaelw/golangchain/schema"
)

// Loader extracts text content from PDF files.
type Loader struct {
	Path string
}

// New creates a PDF loader for the given path.
func New(path string) *Loader {
	return &Loader{Path: path}
}

// Load opens the PDF and extracts text from every page into a single Document.
func (l *Loader) Load(_ context.Context) ([]schema.Document, error) {
	f, r, err := pdflib.Open(l.Path)
	if err != nil {
		return nil, fmt.Errorf("pdf: open %q: %w", l.Path, err)
	}
	defer func() { _ = f.Close() }()

	var buf strings.Builder
	numPages := r.NumPage()
	for i := 1; i <= numPages; i++ {
		page := r.Page(i)
		if page.V.IsNull() {
			continue
		}
		text, err := page.GetPlainText(nil)
		if err != nil {
			continue
		}
		buf.WriteString(text)
	}
	content := strings.TrimSpace(buf.String())
	return []schema.Document{{
		PageContent: content,
		Metadata: map[string]any{
			"source": l.Path,
			"mime":   "application/pdf",
			"pages":  numPages,
		},
	}}, nil
}

var _ documentloader.Loader = (*Loader)(nil)
