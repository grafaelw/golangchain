// Package docx provides a Loader that extracts text from DOCX files.
// Uses only the standard library (archive/zip + encoding/xml).
package docx

import (
	"archive/zip"
	"context"
	"encoding/xml"
	"fmt"
	"io"
	"strings"

	"github.com/grafaelw/golangchain/documentloader"
	"github.com/grafaelw/golangchain/schema"
)

// Loader extracts text from DOCX (Office Open XML) files.
type Loader struct {
	Path string
}

// New creates a DOCX loader.
func New(path string) *Loader {
	return &Loader{Path: path}
}

// Load opens the DOCX archive and extracts text from word/document.xml.
func (l *Loader) Load(_ context.Context) ([]schema.Document, error) {
	r, err := zip.OpenReader(l.Path)
	if err != nil {
		return nil, fmt.Errorf("docx: open %q: %w", l.Path, err)
	}
	defer func() { _ = r.Close() }()

	var content strings.Builder
	for _, f := range r.File {
		if f.Name == "word/document.xml" {
			rc, err := f.Open()
			if err != nil {
				return nil, fmt.Errorf("docx: read document.xml: %w", err)
			}
			data, err := io.ReadAll(rc)
			_ = rc.Close()
			if err != nil {
				return nil, fmt.Errorf("docx: read document.xml: %w", err)
			}
			text, err := extractDOCXText(data)
			if err != nil {
				return nil, fmt.Errorf("docx: parse: %w", err)
			}
			content.WriteString(text)
			break
		}
	}
	return []schema.Document{{
		PageContent: strings.TrimSpace(content.String()),
		Metadata: map[string]any{
			"source": l.Path,
			"mime":   "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
		},
	}}, nil
}

// ---------------------------------------------------------------------------
// XML parsing
// ---------------------------------------------------------------------------

type docxBody struct {
	XMLName    xml.Name        `xml:"body"`
	Paragraphs []docxParagraph `xml:"p"`
}

type docxParagraph struct {
	Runs []docxRun `xml:"r"`
}

type docxRun struct {
	Texts []docxText `xml:"t"`
}

type docxText struct {
	Value string `xml:",chardata"`
}

func extractDOCXText(raw []byte) (string, error) {
	var b docxBody
	if err := xml.Unmarshal(raw, &b); err != nil {
		return "", err
	}
	var out strings.Builder
	for _, p := range b.Paragraphs {
		var line strings.Builder
		for _, r := range p.Runs {
			for _, t := range r.Texts {
				line.WriteString(t.Value)
			}
		}
		txt := strings.TrimSpace(line.String())
		if txt != "" {
			out.WriteString(txt)
			out.WriteString("\n")
		}
	}
	return out.String(), nil
}

var _ documentloader.Loader = (*Loader)(nil)
