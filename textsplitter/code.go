package textsplitter

import (
	"strings"

	"github.com/grafaelw/golangchain/schema"
)

// CodeSplitter splits source code by language-aware separators.
// It extends RecursiveCharacterSplitter with language-specific tokens
// that respect class/function boundaries.
type CodeSplitter struct {
	*RecursiveCharacterSplitter
	language string
}

// NewPythonSplitter creates a code splitter for Python.
func NewPythonSplitter(opts ...Option) *CodeSplitter {
	separators := []string{
		"\nclass ", "\ndef ", "\n\tdef ", "\n    def ",
		"\nasync def ", "\n@", "\n\n", "\n", " ", "",
	}
	return &CodeSplitter{
		RecursiveCharacterSplitter: NewRecursiveCharacterSplitter(separators, opts...),
		language:                   "python",
	}
}

// NewJavaScriptSplitter creates a code splitter for JavaScript/TypeScript.
func NewJavaScriptSplitter(opts ...Option) *CodeSplitter {
	separators := []string{
		"\nfunction ", "\nconst ", "\nlet ", "\nvar ",
		"\nclass ", "\nexport ", "\nimport ", "\ninterface ",
		"\ntype ", "\n\n", "\n", " ", "",
	}
	return &CodeSplitter{
		RecursiveCharacterSplitter: NewRecursiveCharacterSplitter(separators, opts...),
		language:                   "javascript",
	}
}

// NewGoSplitter creates a code splitter for Go.
func NewGoSplitter(opts ...Option) *CodeSplitter {
	separators := []string{
		"\nfunc ", "\ntype ", "\nvar ", "\nconst ",
		"\npackage ", "\nimport (", "\n\n", "\n", " ", "",
	}
	return &CodeSplitter{
		RecursiveCharacterSplitter: NewRecursiveCharacterSplitter(separators, opts...),
		language:                   "go",
	}
}

// SplitText splits source code into chunks.
func (c *CodeSplitter) SplitText(text string) []string {
	return c.RecursiveCharacterSplitter.SplitText(text)
}

// SplitDocuments splits documents containing source code.
func (c *CodeSplitter) SplitDocuments(docs []schema.Document) []schema.Document {
	var out []schema.Document
	for _, doc := range docs {
		chunks := c.RecursiveCharacterSplitter.SplitText(doc.PageContent)
		for _, chunk := range chunks {
			meta := make(map[string]any)
			for k, v := range doc.Metadata {
				meta[k] = v
			}
			meta["language"] = c.language
			out = append(out, schema.Document{
				PageContent: strings.TrimSpace(chunk),
				Metadata:    meta,
			})
		}
	}
	return out
}
