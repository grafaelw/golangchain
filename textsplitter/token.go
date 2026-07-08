package textsplitter

import "github.com/grafaelw/golangchain/schema"

// ---------------------------------------------------------------------------
// TokenSplitter
// ---------------------------------------------------------------------------

// TokenSplitter splits text into chunks by approximate token count.
// Uses ~4 chars per token as the default approximation.
type TokenSplitter struct {
	chunkSize    int
	chunkOverlap int
	charsPerTok  int
	separator    string
}

// TokenOption configures a TokenSplitter.
type TokenOption func(*TokenSplitter)

// WithTokenChunkSize sets the maximum token count per chunk (default 512).
func WithTokenChunkSize(n int) TokenOption {
	return func(s *TokenSplitter) { s.chunkSize = n }
}

// WithTokenChunkOverlap sets the token overlap between consecutive chunks (default 50).
func WithTokenChunkOverlap(n int) TokenOption {
	return func(s *TokenSplitter) { s.chunkOverlap = n }
}

// WithTokenCharsPerToken sets the character-to-token ratio (default 4).
func WithTokenCharsPerToken(n int) TokenOption {
	return func(s *TokenSplitter) { s.charsPerTok = n }
}

// WithTokenSeparator sets the separator to split on first (default " ").
func WithTokenSeparator(sep string) TokenOption {
	return func(s *TokenSplitter) { s.separator = sep }
}

// NewTokenSplitter creates a token-based splitter.
func NewTokenSplitter(opts ...TokenOption) *TokenSplitter {
	s := &TokenSplitter{
		chunkSize:    512,
		chunkOverlap: 50,
		charsPerTok:  4,
		separator:    " ",
	}
	for _, o := range opts {
		o(s)
	}
	if s.chunkOverlap >= s.chunkSize {
		s.chunkOverlap = s.chunkSize / 4
	}
	return s
}

// SplitText splits text into chunks of approximately chunkSize tokens.
func (s *TokenSplitter) SplitText(text string) []string {
	tokenLen := func(str string) int {
		return (RuneLen(str) + s.charsPerTok - 1) / s.charsPerTok
	}
	cfg := &config{
		chunkSize:    s.chunkSize,
		chunkOverlap: s.chunkOverlap,
		lenFunc:      tokenLen,
		keepSep:      false,
	}
	parts := splitWithSep(text, s.separator, cfg.keepSep)
	return mergeSplits(parts, s.separator, cfg)
}

// SplitDocuments splits documents using the token-based splitter.
func (s *TokenSplitter) SplitDocuments(docs []schema.Document) []schema.Document {
	return splitDocuments(s, docs)
}
