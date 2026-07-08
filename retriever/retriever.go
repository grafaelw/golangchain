// Package retriever — see doc.go for the package overview.
package retriever

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"unicode"

	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/schema"
	"github.com/grafaelw/golangchain/vectorstore"
)

// ---------------------------------------------------------------------------
// Retriever interface
// ---------------------------------------------------------------------------

// Retriever returns the k documents most relevant to a natural-language query.
// Implementations must be safe for concurrent use.
type Retriever interface {
	GetRelevantDocuments(ctx context.Context, query string) ([]schema.Document, error)
}

// ---------------------------------------------------------------------------
// VectorStoreRetriever
// ---------------------------------------------------------------------------

// VectorStoreRetriever adapts a VectorStore to the Retriever interface.
type VectorStoreRetriever struct {
	Store vectorstore.VectorStore
	K     int
}

// NewVectorStoreRetriever creates a Retriever backed by store returning k documents.
func NewVectorStoreRetriever(store vectorstore.VectorStore, k int) *VectorStoreRetriever {
	return &VectorStoreRetriever{Store: store, K: k}
}

func (r *VectorStoreRetriever) GetRelevantDocuments(ctx context.Context, query string) ([]schema.Document, error) {
	return r.Store.SimilaritySearch(ctx, query, r.K)
}

// ---------------------------------------------------------------------------
// BM25Retriever — classical lexical retrieval
// ---------------------------------------------------------------------------

// BM25Retriever is a self-contained BM25 index over an in-memory corpus.
// No embeddings are required, making it a fast complementary retriever for
// EnsembleRetriever (hybrid search).
//
//	r := retriever.NewBM25Retriever(docs, 5)
//	hits, _ := r.GetRelevantDocuments(ctx, "quick brown fox")
type BM25Retriever struct {
	docs    []schema.Document
	terms   []map[string]int // per-doc term frequency
	docLen  []int
	avgLen  float64
	df      map[string]int
	total   int
	k       int
	k1, b   float64
	stopSet map[string]struct{}
}

// BM25Option configures a BM25Retriever.
type BM25Option func(*BM25Retriever)

// WithBM25K1 sets the BM25 k1 parameter (default 1.5).
func WithBM25K1(v float64) BM25Option { return func(r *BM25Retriever) { r.k1 = v } }

// WithBM25B sets the BM25 b parameter (default 0.75).
func WithBM25B(v float64) BM25Option { return func(r *BM25Retriever) { r.b = v } }

// WithBM25Stopwords replaces the default English stopword list.
func WithBM25Stopwords(words []string) BM25Option {
	return func(r *BM25Retriever) {
		set := make(map[string]struct{}, len(words))
		for _, w := range words {
			set[strings.ToLower(w)] = struct{}{}
		}
		r.stopSet = set
	}
}

// DefaultBM25Stopwords is a small English stopword list.
var DefaultBM25Stopwords = []string{
	"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
	"he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
	"were", "will", "with",
}

// NewBM25Retriever indexes the given documents. k is the number of hits to return.
func NewBM25Retriever(docs []schema.Document, k int, opts ...BM25Option) *BM25Retriever {
	r := &BM25Retriever{k: k, k1: 1.5, b: 0.75}
	WithBM25Stopwords(DefaultBM25Stopwords)(r)
	for _, o := range opts {
		o(r)
	}
	r.build(docs)
	return r
}

func (r *BM25Retriever) build(docs []schema.Document) {
	r.docs = docs
	r.total = len(docs)
	r.terms = make([]map[string]int, len(docs))
	r.docLen = make([]int, len(docs))
	r.df = make(map[string]int)

	var totalLen int
	for i, d := range docs {
		toks := r.tokenize(d.PageContent)
		tf := make(map[string]int, len(toks))
		for _, t := range toks {
			tf[t]++
		}
		r.terms[i] = tf
		r.docLen[i] = len(toks)
		totalLen += len(toks)
		for term := range tf {
			r.df[term]++
		}
	}
	if len(docs) > 0 {
		r.avgLen = float64(totalLen) / float64(len(docs))
	}
}

func (r *BM25Retriever) tokenize(s string) []string {
	var out []string
	var b strings.Builder
	flush := func() {
		if b.Len() == 0 {
			return
		}
		w := strings.ToLower(b.String())
		b.Reset()
		if _, stop := r.stopSet[w]; stop {
			return
		}
		out = append(out, w)
	}
	for _, ch := range s {
		if unicode.IsLetter(ch) || unicode.IsDigit(ch) {
			b.WriteRune(ch)
		} else {
			flush()
		}
	}
	flush()
	return out
}

// Score returns the BM25 score of a single document for a query.
func (r *BM25Retriever) Score(docIdx int, query string) float64 {
	if docIdx < 0 || docIdx >= len(r.docs) {
		return 0
	}
	toks := r.tokenize(query)
	var score float64
	dl := float64(r.docLen[docIdx])
	tf := r.terms[docIdx]
	for _, t := range toks {
		f := float64(tf[t])
		if f == 0 {
			continue
		}
		n := float64(r.df[t])
		idf := math.Log(1 + (float64(r.total)-n+0.5)/(n+0.5))
		norm := f * (r.k1 + 1) / (f + r.k1*(1-r.b+r.b*dl/r.avgLen))
		score += idf * norm
	}
	return score
}

func (r *BM25Retriever) GetRelevantDocuments(_ context.Context, query string) ([]schema.Document, error) {
	type scored struct {
		i     int
		score float64
	}
	scores := make([]scored, 0, len(r.docs))
	for i := range r.docs {
		s := r.Score(i, query)
		if s > 0 {
			scores = append(scores, scored{i, s})
		}
	}
	sort.Slice(scores, func(a, b int) bool { return scores[a].score > scores[b].score })

	k := r.k
	if k > len(scores) {
		k = len(scores)
	}
	out := make([]schema.Document, k)
	for i := 0; i < k; i++ {
		d := r.docs[scores[i].i]
		d.Score = scores[i].score
		out[i] = d
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// EnsembleRetriever — Reciprocal Rank Fusion
// ---------------------------------------------------------------------------

// EnsembleRetriever combines several retrievers using Reciprocal Rank Fusion:
// score(doc) = Σ_i weight_i / (k + rank_i(doc)). The default RRF k is 60.
//
// Documents are deduplicated by identity: their metadata "id" if present,
// otherwise the (trimmed, lowercased) first 200 chars of PageContent.
type EnsembleRetriever struct {
	Retrievers []Retriever
	Weights    []float64
	K          int     // top-k results to return (default: max of children's k, else 5)
	RRFK       float64 // rank normaliser (default 60)
}

// NewEnsembleRetriever constructs an EnsembleRetriever. If weights is nil,
// each retriever receives equal weight.
func NewEnsembleRetriever(retrievers []Retriever, weights []float64, k int) *EnsembleRetriever {
	if weights == nil {
		weights = make([]float64, len(retrievers))
		for i := range weights {
			weights[i] = 1
		}
	}
	if k <= 0 {
		k = 5
	}
	return &EnsembleRetriever{Retrievers: retrievers, Weights: weights, K: k, RRFK: 60}
}

func (e *EnsembleRetriever) GetRelevantDocuments(ctx context.Context, query string) ([]schema.Document, error) {
	if len(e.Retrievers) != len(e.Weights) {
		return nil, fmt.Errorf("retriever: ensemble: retrievers/weights length mismatch")
	}

	type result struct {
		docs []schema.Document
		err  error
	}
	results := make([]result, len(e.Retrievers))
	var wg sync.WaitGroup
	for i, r := range e.Retrievers {
		wg.Add(1)
		go func(idx int, ret Retriever) {
			defer wg.Done()
			docs, err := ret.GetRelevantDocuments(ctx, query)
			results[idx] = result{docs: docs, err: err}
		}(i, r)
	}
	wg.Wait()

	scores := map[string]float64{}
	docByKey := map[string]schema.Document{}
	for i, res := range results {
		if res.err != nil {
			return nil, fmt.Errorf("retriever: ensemble: source %d: %w", i, res.err)
		}
		for rank, d := range res.docs {
			key := docKey(d)
			scores[key] += e.Weights[i] / (e.RRFK + float64(rank+1))
			if _, ok := docByKey[key]; !ok {
				docByKey[key] = d
			}
		}
	}

	type scored struct {
		key   string
		score float64
	}
	ranked := make([]scored, 0, len(scores))
	for k, s := range scores {
		ranked = append(ranked, scored{k, s})
	}
	sort.Slice(ranked, func(i, j int) bool { return ranked[i].score > ranked[j].score })

	k := e.K
	if k > len(ranked) {
		k = len(ranked)
	}
	out := make([]schema.Document, k)
	for i := 0; i < k; i++ {
		d := docByKey[ranked[i].key]
		d.Score = ranked[i].score
		out[i] = d
	}
	return out, nil
}

func docKey(d schema.Document) string {
	if v, ok := d.Metadata["id"]; ok {
		return fmt.Sprint(v)
	}
	s := strings.ToLower(strings.TrimSpace(d.PageContent))
	if len(s) > 200 {
		s = s[:200]
	}
	return s
}

// ---------------------------------------------------------------------------
// MultiQueryRetriever
// ---------------------------------------------------------------------------

// MultiQueryRetriever asks an LLM to produce N alternative phrasings of the
// input query, runs the base retriever for each, and returns the deduplicated
// union. Useful when the user's phrasing is ambiguous or when recall matters
// more than precision.
type MultiQueryRetriever struct {
	Base      Retriever
	LLM       llm.LLM
	N         int    // number of alternative queries to generate (default 3)
	Prompt    string // template — must contain {{ .query }} and {{ .n }}
	IncludeQ  bool   // include the original query in the union (default true)
	MaxUnique int    // cap on returned docs (default 10)
}

// DefaultMultiQueryPrompt is the query-rewrite prompt.
const DefaultMultiQueryPrompt = `You are a search-query expander. Produce {{ .n }} alternative phrasings of the user's question below. Each on its own line, no numbering, no explanations.

Question: {{ .query }}`

// NewMultiQueryRetriever wraps base with an LLM-driven query expander.
func NewMultiQueryRetriever(base Retriever, model llm.LLM) *MultiQueryRetriever {
	return &MultiQueryRetriever{
		Base:      base,
		LLM:       model,
		N:         3,
		Prompt:    DefaultMultiQueryPrompt,
		IncludeQ:  true,
		MaxUnique: 10,
	}
}

func (m *MultiQueryRetriever) GetRelevantDocuments(ctx context.Context, query string) ([]schema.Document, error) {
	prompt := strings.ReplaceAll(m.Prompt, "{{ .query }}", query)
	prompt = strings.ReplaceAll(prompt, "{{ .n }}", fmt.Sprintf("%d", m.N))

	gen, err := m.LLM.Generate(ctx, []schema.Message{schema.NewHumanMessage(prompt)})
	if err != nil {
		return nil, fmt.Errorf("retriever: multiquery: llm: %w", err)
	}

	queries := parseQueries(gen.Text)
	if m.IncludeQ {
		queries = append([]string{query}, queries...)
	}

	seen := map[string]schema.Document{}
	for _, q := range queries {
		docs, err := m.Base.GetRelevantDocuments(ctx, q)
		if err != nil {
			return nil, fmt.Errorf("retriever: multiquery: base %q: %w", q, err)
		}
		for _, d := range docs {
			seen[docKey(d)] = d
		}
	}

	out := make([]schema.Document, 0, len(seen))
	for _, d := range seen {
		out = append(out, d)
	}
	if m.MaxUnique > 0 && len(out) > m.MaxUnique {
		out = out[:m.MaxUnique]
	}
	return out, nil
}

func parseQueries(text string) []string {
	var out []string
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// Strip common list prefixes like "1. ", "- ", "* ".
		line = strings.TrimLeft(line, "-*0123456789. )")
		line = strings.TrimSpace(line)
		if line != "" {
			out = append(out, line)
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// ContextualCompressionRetriever
// ---------------------------------------------------------------------------

// DocumentCompressor filters or shrinks a retrieved document set given the
// original query. Returning an empty slice signals "drop entirely".
type DocumentCompressor interface {
	Compress(ctx context.Context, docs []schema.Document, query string) ([]schema.Document, error)
}

// ContextualCompressionRetriever runs a base retriever and then passes the
// results through a DocumentCompressor before returning them.
type ContextualCompressionRetriever struct {
	Base       Retriever
	Compressor DocumentCompressor
}

// NewContextualCompressionRetriever wraps base with compressor.
func NewContextualCompressionRetriever(base Retriever, c DocumentCompressor) *ContextualCompressionRetriever {
	return &ContextualCompressionRetriever{Base: base, Compressor: c}
}

func (c *ContextualCompressionRetriever) GetRelevantDocuments(ctx context.Context, query string) ([]schema.Document, error) {
	docs, err := c.Base.GetRelevantDocuments(ctx, query)
	if err != nil {
		return nil, err
	}
	return c.Compressor.Compress(ctx, docs, query)
}

// ---------------------------------------------------------------------------
// Built-in compressors
// ---------------------------------------------------------------------------

// KeywordCompressor drops documents that don't contain any of the query's
// whitespace-tokenised words (case-insensitive). It is dependency-free and
// useful as a cheap first-pass filter.
type KeywordCompressor struct{ MinHits int } // default 1

func (k KeywordCompressor) Compress(_ context.Context, docs []schema.Document, query string) ([]schema.Document, error) {
	terms := strings.Fields(strings.ToLower(query))
	min := k.MinHits
	if min <= 0 {
		min = 1
	}
	out := docs[:0:0]
	for _, d := range docs {
		body := strings.ToLower(d.PageContent)
		hits := 0
		for _, t := range terms {
			if strings.Contains(body, t) {
				hits++
			}
		}
		if hits >= min {
			out = append(out, d)
		}
	}
	return out, nil
}

// LLMRelevanceCompressor asks an LLM to answer YES/NO for each document.
// Documents receiving a NO are dropped.
type LLMRelevanceCompressor struct {
	LLM    llm.LLM
	Prompt string // must contain {{ .query }} and {{ .doc }}
}

// DefaultRelevancePrompt is the yes/no scoring prompt.
const DefaultRelevancePrompt = `Given the user question and the candidate document, answer with a single word: YES if the document is relevant to answering the question, otherwise NO.

Question: {{ .query }}

Document:
{{ .doc }}

Answer:`

// NewLLMRelevanceCompressor constructs a compressor backed by model.
func NewLLMRelevanceCompressor(model llm.LLM) *LLMRelevanceCompressor {
	return &LLMRelevanceCompressor{LLM: model, Prompt: DefaultRelevancePrompt}
}

func (c *LLMRelevanceCompressor) Compress(ctx context.Context, docs []schema.Document, query string) ([]schema.Document, error) {
	out := docs[:0:0]
	for _, d := range docs {
		p := strings.ReplaceAll(c.Prompt, "{{ .query }}", query)
		p = strings.ReplaceAll(p, "{{ .doc }}", d.PageContent)
		gen, err := c.LLM.Generate(ctx, []schema.Message{schema.NewHumanMessage(p)})
		if err != nil {
			return nil, fmt.Errorf("retriever: compressor: %w", err)
		}
		if strings.HasPrefix(strings.ToUpper(strings.TrimSpace(gen.Text)), "YES") {
			out = append(out, d)
		}
	}
	return out, nil
}
