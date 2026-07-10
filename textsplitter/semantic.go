package textsplitter

import (
	"context"
	"fmt"
	"math"
	"strings"

	"github.com/grafaelw/golangchain/schema"
)

// SemanticChunker splits text at points where adjacent passages have low
// semantic similarity, indicating a topic boundary. It requires an embedder
// to compute sentence-level embeddings.
//
//	embedder, _ := embeddings.NewOpenAIEmbedder(key, "text-embedding-3-small")
//	chunker := textsplitter.NewSemanticChunker(embedder,
//	    textsplitter.WithChunkSize(512),
//	    textsplitter.WithChunkOverlap(50),
//	)
type SemanticChunker struct {
	embedder  Embedder
	chunkSize int
	overlap   int
	threshold float64
	lenFunc   LenFunc
}

// SemanticOption configures a SemanticChunker.
type SemanticOption func(*SemanticChunker)

// WithSemanticThreshold sets the similarity threshold for split points.
func WithSemanticThreshold(t float64) SemanticOption {
	return func(s *SemanticChunker) { s.threshold = t }
}

// Embedder is a minimal interface for computing embeddings.
type Embedder interface {
	EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error)
}

// NewSemanticChunker creates a chunker that splits at topic boundaries.
func NewSemanticChunker(embedder Embedder, opts ...Option) *SemanticChunker {
	s := &SemanticChunker{
		embedder:  embedder,
		chunkSize: 1000,
		overlap:   100,
		threshold: 0.5,
		lenFunc:   ApproxTokenLen,
	}
	// Apply shared text splitter options where relevant
	cfg := newConfig(opts...)
	s.chunkSize = cfg.chunkSize
	s.overlap = cfg.chunkOverlap
	s.lenFunc = cfg.lenFunc
	return s
}

// SplitText splits text at semantic boundaries. Falls back to original text on error.
func (s *SemanticChunker) SplitText(text string) []string {
	chunks, err := s.SplitContext(context.Background(), text)
	if err != nil {
		return []string{text}
	}
	return chunks
}

// SplitContext splits text using the given context for embedding calls.
func (s *SemanticChunker) SplitContext(ctx context.Context, text string) ([]string, error) {
	sentences := splitSentences(text)
	if len(sentences) <= 1 {
		return []string{text}, nil
	}

	embeddings, err := s.embedder.EmbedDocuments(ctx, sentences)
	if err != nil {
		return nil, fmt.Errorf("semantic chunker: embed: %w", err)
	}

	var breaks []int
	for i := 1; i < len(embeddings); i++ {
		sim := calcCosineSimilarity(embeddings[i-1], embeddings[i])
		if sim < s.threshold {
			breaks = append(breaks, i)
		}
	}

	var chunks []string
	start := 0
	for _, brk := range breaks {
		chunk := strings.Join(sentences[start:brk], " ")
		if s.lenFunc(chunk) > s.chunkSize {
			subSplitter := NewCharacterSplitter(" ", WithChunkSize(s.chunkSize), WithChunkOverlap(s.overlap), WithLenFunc(s.lenFunc))
			subs := subSplitter.SplitText(chunk)
			chunks = append(chunks, subs...)
		} else {
			chunks = append(chunks, chunk)
		}
		start = brk
	}
	if start < len(sentences) {
		chunks = append(chunks, strings.Join(sentences[start:], " "))
	}

	return chunks, nil
}

// SplitDocuments splits each document at semantic boundaries.
func (s *SemanticChunker) SplitDocuments(docs []schema.Document) []schema.Document {
	var out []schema.Document
	for _, doc := range docs {
		chunks := s.SplitText(doc.PageContent)
		for _, chunk := range chunks {
			cp := schema.Document{
				PageContent: chunk,
				Metadata:    copyMap(doc.Metadata),
			}
			out = append(out, cp)
		}
	}
	return out
}

func splitSentences(text string) []string {
	var sentences []string
	var current strings.Builder
	for i := 0; i < len(text); i++ {
		ch := text[i]
		current.WriteByte(ch)
		if ch == '.' || ch == '!' || ch == '?' {
			if i+1 < len(text) && (text[i+1] == ' ' || text[i+1] == '\n' || text[i+1] == '"') {
				t := strings.TrimSpace(current.String())
				if t != "" {
					sentences = append(sentences, t)
				}
				current.Reset()
			}
		}
		if ch == '\n' && strings.TrimSpace(current.String()) != "" {
			t := strings.TrimSpace(current.String())
			if t != "" {
				sentences = append(sentences, t)
			}
			current.Reset()
		}
	}
	if strings.TrimSpace(current.String()) != "" {
		sentences = append(sentences, strings.TrimSpace(current.String()))
	}
	return sentences
}

func calcCosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	d := math.Sqrt(na) * math.Sqrt(nb)
	if d == 0 {
		return 0
	}
	return dot / d
}

func copyMap(src map[string]any) map[string]any {
	if src == nil {
		return nil
	}
	dst := make(map[string]any, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}
