// Package textsplitter — see doc.go for the package overview.
package textsplitter

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// Splitter interface
// ---------------------------------------------------------------------------

// Splitter chunks a piece of text into smaller pieces.
type Splitter interface {
	// SplitText returns the text broken into chunks.
	SplitText(text string) []string
	// SplitDocuments splits each document's PageContent and returns one
	// Document per chunk, cloning and extending the source Metadata with
	// "chunk" (0-based index) and "source_chunks" (total count).
	SplitDocuments(docs []schema.Document) []schema.Document
}

// ---------------------------------------------------------------------------
// Length function
// ---------------------------------------------------------------------------

// LenFunc computes the "length" of a chunk. The default RuneLen measures
// UTF-8 runes; provide your own for a tokenizer-aware count.
type LenFunc func(string) int

// RuneLen counts UTF-8 runes.
func RuneLen(s string) int { return utf8.RuneCountInString(s) }

// ByteLen counts bytes.
func ByteLen(s string) int { return len(s) }

// ApproxTokenLen approximates a BPE token count as runes/4 (rounded up).
// Good enough for splitting; not accurate for billing.
func ApproxTokenLen(s string) int {
	n := utf8.RuneCountInString(s)
	return (n + 3) / 4
}

// ---------------------------------------------------------------------------
// Common option plumbing
// ---------------------------------------------------------------------------

// Option configures a splitter at construction time.
type Option func(*config)

type config struct {
	chunkSize    int
	chunkOverlap int
	lenFunc      LenFunc
	keepSep      bool
}

func newConfig(opts ...Option) *config {
	c := &config{chunkSize: 1000, chunkOverlap: 200, lenFunc: RuneLen, keepSep: true}
	for _, o := range opts {
		o(c)
	}
	if c.chunkOverlap >= c.chunkSize {
		c.chunkOverlap = c.chunkSize / 4
	}
	return c
}

// WithChunkSize sets the maximum length of each chunk (default 1000).
func WithChunkSize(n int) Option { return func(c *config) { c.chunkSize = n } }

// WithChunkOverlap sets how much content the next chunk should repeat from
// the previous one (default 200). Must be smaller than chunk size.
func WithChunkOverlap(n int) Option { return func(c *config) { c.chunkOverlap = n } }

// WithLenFunc replaces the length function (default RuneLen).
func WithLenFunc(fn LenFunc) Option { return func(c *config) { c.lenFunc = fn } }

// WithKeepSeparator controls whether separators are retained in the output
// chunks (default true, matching LangChain behaviour).
func WithKeepSeparator(keep bool) Option { return func(c *config) { c.keepSep = keep } }

// ---------------------------------------------------------------------------
// CharacterSplitter
// ---------------------------------------------------------------------------

// CharacterSplitter splits on a single separator string.
type CharacterSplitter struct {
	cfg *config
	sep string
}

// NewCharacterSplitter constructs a CharacterSplitter. The default separator
// is "\n\n" (paragraph break).
func NewCharacterSplitter(separator string, opts ...Option) *CharacterSplitter {
	if separator == "" {
		separator = "\n\n"
	}
	return &CharacterSplitter{cfg: newConfig(opts...), sep: separator}
}

func (s *CharacterSplitter) SplitText(text string) []string {
	parts := splitWithSep(text, s.sep, s.cfg.keepSep)
	return mergeSplits(parts, s.sep, s.cfg)
}

func (s *CharacterSplitter) SplitDocuments(docs []schema.Document) []schema.Document {
	return splitDocuments(s, docs)
}

// ---------------------------------------------------------------------------
// RecursiveCharacterSplitter
// ---------------------------------------------------------------------------

// RecursiveCharacterSplitter tries a list of separators in order. For each
// oversize piece it recurses with the next separator, all the way down to
// character-level splitting.
type RecursiveCharacterSplitter struct {
	cfg  *config
	seps []string
}

// DefaultRecursiveSeparators mirrors LangChain's recommended defaults for
// English prose: paragraph, line, sentence, word, character.
var DefaultRecursiveSeparators = []string{"\n\n", "\n", ". ", " ", ""}

// NewRecursiveCharacterSplitter constructs a RecursiveCharacterSplitter.
// Pass nil for separators to use DefaultRecursiveSeparators.
func NewRecursiveCharacterSplitter(separators []string, opts ...Option) *RecursiveCharacterSplitter {
	if separators == nil {
		separators = DefaultRecursiveSeparators
	}
	return &RecursiveCharacterSplitter{cfg: newConfig(opts...), seps: separators}
}

func (s *RecursiveCharacterSplitter) SplitText(text string) []string {
	return s.split(text, s.seps)
}

func (s *RecursiveCharacterSplitter) split(text string, seps []string) []string {
	// Pick the first separator that appears (or the last as a fallback).
	sep := seps[len(seps)-1]
	remaining := seps[len(seps):] // empty slice; recursion fallback is char-by-char
	for i, candidate := range seps {
		if candidate == "" {
			sep = candidate
			remaining = seps[i+1:]
			break
		}
		if strings.Contains(text, candidate) {
			sep = candidate
			remaining = seps[i+1:]
			break
		}
	}

	parts := splitWithSep(text, sep, s.cfg.keepSep)

	// Recurse into oversize parts.
	var goodParts []string
	for _, p := range parts {
		if s.cfg.lenFunc(p) < s.cfg.chunkSize {
			goodParts = append(goodParts, p)
			continue
		}
		if len(remaining) == 0 {
			// No further separator: hard-cut by length.
			goodParts = append(goodParts, hardCut(p, s.cfg)...)
			continue
		}
		goodParts = append(goodParts, s.split(p, remaining)...)
	}

	return mergeSplits(goodParts, sep, s.cfg)
}

func (s *RecursiveCharacterSplitter) SplitDocuments(docs []schema.Document) []schema.Document {
	return splitDocuments(s, docs)
}

// ---------------------------------------------------------------------------
// MarkdownSplitter — Markdown-aware separators
// ---------------------------------------------------------------------------

// MarkdownSeparators are recursive splitters tuned for Markdown documents.
// Order: H1..H6 headings, code fence, horizontal rule, list bullet,
// paragraph, line, sentence, word, character.
var MarkdownSeparators = []string{
	"\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### ",
	"\n```\n",
	"\n---\n",
	"\n- ", "\n* ",
	"\n\n", "\n", ". ", " ", "",
}

// NewMarkdownSplitter returns a RecursiveCharacterSplitter configured with
// Markdown-aware separators.
func NewMarkdownSplitter(opts ...Option) *RecursiveCharacterSplitter {
	return NewRecursiveCharacterSplitter(MarkdownSeparators, opts...)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

func splitWithSep(text, sep string, keep bool) []string {
	if sep == "" {
		// Split into single runes.
		out := make([]string, 0, len(text))
		for _, r := range text {
			out = append(out, string(r))
		}
		return out
	}
	if !keep {
		return strings.Split(text, sep)
	}
	// Keep-separator mode: attach the separator to the tail of each
	// non-final piece so joins remain lossless.
	parts := strings.SplitAfter(text, sep)
	out := parts[:0]
	for _, p := range parts {
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// mergeSplits packs consecutive pieces into chunks up to chunkSize, adding
// overlap between adjacent chunks. Pieces individually larger than chunkSize
// are emitted verbatim (they were produced by the deepest recursion level).
func mergeSplits(parts []string, sep string, cfg *config) []string {
	if len(parts) == 0 {
		return nil
	}
	var (
		out     []string
		current []string
		total   int
	)
	sepLen := cfg.lenFunc(sep)

	flush := func() {
		if len(current) == 0 {
			return
		}
		out = append(out, strings.TrimSpace(strings.Join(current, "")))
	}

	for _, p := range parts {
		plen := cfg.lenFunc(p)
		if total+plen+sepLen*len(current) > cfg.chunkSize && len(current) > 0 {
			flush()
			// Build overlap tail for the next chunk.
			current, total = overlapTail(current, sep, cfg)
		}
		current = append(current, p)
		total += plen
	}
	flush()
	return out
}

// overlapTail returns the trailing pieces from prior chunk that fit inside
// cfg.chunkOverlap, so they seed the next chunk.
func overlapTail(prev []string, sep string, cfg *config) ([]string, int) {
	if cfg.chunkOverlap <= 0 {
		return nil, 0
	}
	sepLen := cfg.lenFunc(sep)
	var tail []string
	total := 0
	for i := len(prev) - 1; i >= 0; i-- {
		plen := cfg.lenFunc(prev[i])
		if total+plen+sepLen*len(tail) > cfg.chunkOverlap {
			break
		}
		tail = append([]string{prev[i]}, tail...)
		total += plen
	}
	return tail, total
}

// hardCut slices s into fixed-length pieces when no separator can shrink it.
func hardCut(s string, cfg *config) []string {
	if cfg.lenFunc(s) <= cfg.chunkSize {
		return []string{s}
	}
	runes := []rune(s)
	step := cfg.chunkSize
	if step <= 0 {
		step = 1
	}
	overlap := cfg.chunkOverlap
	if overlap < 0 {
		overlap = 0
	}
	if overlap >= step {
		overlap = step / 4
	}
	var out []string
	for i := 0; i < len(runes); i += step - overlap {
		end := i + step
		if end > len(runes) {
			end = len(runes)
		}
		out = append(out, string(runes[i:end]))
		if end == len(runes) {
			break
		}
	}
	return out
}

// splitDocuments is shared by all splitter implementations.
func splitDocuments(s Splitter, docs []schema.Document) []schema.Document {
	var out []schema.Document
	for _, doc := range docs {
		chunks := s.SplitText(doc.PageContent)
		for i, c := range chunks {
			md := cloneMetadata(doc.Metadata)
			md["chunk"] = i
			md["source_chunks"] = len(chunks)
			out = append(out, schema.Document{
				PageContent: c,
				Metadata:    md,
			})
		}
	}
	return out
}

func cloneMetadata(m map[string]any) map[string]any {
	if m == nil {
		return map[string]any{}
	}
	cp := make(map[string]any, len(m)+2)
	for k, v := range m {
		cp[k] = v
	}
	return cp
}

// ---------------------------------------------------------------------------
// Debug helpers
// ---------------------------------------------------------------------------

// Summary returns a short human-readable description of a chunk set,
// useful for logging or examples.
func Summary(chunks []string) string {
	if len(chunks) == 0 {
		return "0 chunks"
	}
	return fmt.Sprintf("%d chunks, first=%d chars, last=%d chars",
		len(chunks), len(chunks[0]), len(chunks[len(chunks)-1]))
}
