// This file adds retrieval-augmented and summarization chains that live
// alongside LLMChain, SequentialChain, MapChain, and RouterChain.
//
//   - RetrievalQAChain:      RAG — retrieve documents, stuff them into a
//                            prompt, ask the LLM.
//   - MapReduceSummarizer:   summarise each chunk in parallel, then reduce.
//   - RefineSummarizer:      seed a summary and iteratively refine it chunk
//                            by chunk. Best when order matters.

package chain

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/retriever"
	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// RetrievalQAChain — classical RAG
// ---------------------------------------------------------------------------

// RetrievalQAChain performs a "stuff"-style RAG: fetch top-k documents from
// the retriever, format them into a context block, and ask the LLM.
//
// Input to Invoke must be a string (the question) or map with key "question".
// The output is a string answer.
type RetrievalQAChain struct {
	Retriever    retriever.Retriever
	LLM          llm.LLM
	LLMOptions   []llm.Option
	Prompt       string // must contain {{ .context }} and {{ .question }}
	ReturnSource bool   // if true, Invoke returns map{"answer", "sources"}
	Name         string
}

// DefaultRAGPrompt is the "stuff" RAG prompt.
const DefaultRAGPrompt = `Use ONLY the following context to answer the question. If the answer isn't in the context, say you don't know.

Context:
{{ .context }}

Question: {{ .question }}
Answer:`

// NewRetrievalQAChain wires a retriever to an LLM using the default RAG prompt.
func NewRetrievalQAChain(r retriever.Retriever, model llm.LLM, opts ...llm.Option) *RetrievalQAChain {
	return &RetrievalQAChain{
		Retriever:  r,
		LLM:        model,
		LLMOptions: opts,
		Prompt:     DefaultRAGPrompt,
		Name:       "RetrievalQAChain",
	}
}

func (c *RetrievalQAChain) Invoke(ctx context.Context, input any) (any, error) {
	question, err := extractQuestion(input)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", c.Name, err)
	}
	docs, err := c.Retriever.GetRelevantDocuments(ctx, question)
	if err != nil {
		return nil, fmt.Errorf("%s: retrieve: %w", c.Name, err)
	}
	prompt := renderRAGPrompt(c.Prompt, question, docs)
	gen, err := c.LLM.Generate(ctx, []schema.Message{schema.NewHumanMessage(prompt)}, c.LLMOptions...)
	if err != nil {
		return nil, fmt.Errorf("%s: llm: %w", c.Name, err)
	}
	answer := strings.TrimSpace(gen.Text)
	if c.ReturnSource {
		return map[string]any{"answer": answer, "sources": docs}, nil
	}
	return answer, nil
}

func (c *RetrievalQAChain) Stream(ctx context.Context, input any) (<-chan StreamChunk, error) {
	question, err := extractQuestion(input)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", c.Name, err)
	}
	docs, err := c.Retriever.GetRelevantDocuments(ctx, question)
	if err != nil {
		return nil, fmt.Errorf("%s: retrieve: %w", c.Name, err)
	}
	prompt := renderRAGPrompt(c.Prompt, question, docs)
	llmCh, err := c.LLM.Stream(ctx, []schema.Message{schema.NewHumanMessage(prompt)}, c.LLMOptions...)
	if err != nil {
		return nil, err
	}
	out := make(chan StreamChunk, 32)
	go func() {
		defer close(out)
		for chunk := range llmCh {
			if chunk.Err != nil {
				out <- StreamChunk{Err: chunk.Err}
				return
			}
			out <- StreamChunk{Value: chunk.Text, Done: chunk.Done}
		}
	}()
	return out, nil
}

func (c *RetrievalQAChain) Pipe(next Runnable) Runnable {
	return &pipeRunnable{first: c, second: next}
}

func (c *RetrievalQAChain) Batch(ctx context.Context, inputs []any) ([]any, error) {
	return RunBatch(ctx, c, inputs)
}

func extractQuestion(input any) (string, error) {
	switch v := input.(type) {
	case string:
		return v, nil
	case map[string]any:
		if q, ok := v["question"].(string); ok {
			return q, nil
		}
		if q, ok := v["input"].(string); ok {
			return q, nil
		}
		return "", fmt.Errorf("map input missing string \"question\"")
	default:
		return "", fmt.Errorf("expected string or map, got %T", input)
	}
}

func renderRAGPrompt(tmpl, question string, docs []schema.Document) string {
	var ctx strings.Builder
	for i, d := range docs {
		fmt.Fprintf(&ctx, "[%d] %s\n\n", i+1, d.PageContent)
	}
	p := strings.ReplaceAll(tmpl, "{{ .context }}", strings.TrimSpace(ctx.String()))
	p = strings.ReplaceAll(p, "{{ .question }}", question)
	return p
}

// ---------------------------------------------------------------------------
// MapReduceSummarizer
// ---------------------------------------------------------------------------

// MapReduceSummarizer summarises each document in parallel (map), then feeds
// the intermediate summaries into a reduce prompt for a final summary.
//
// Input to Invoke must be []schema.Document (or []string, converted).
type MapReduceSummarizer struct {
	LLM          llm.LLM
	LLMOptions   []llm.Option
	MapPrompt    string // must contain {{ .content }}
	ReducePrompt string // must contain {{ .summaries }}
	Concurrency  int
	Name         string
}

// DefaultMapPrompt is used to summarise each chunk.
const DefaultMapPrompt = `Summarise the following text concisely:

{{ .content }}

Concise summary:`

// DefaultReducePrompt combines chunk summaries into one.
const DefaultReducePrompt = `You are given several partial summaries of a longer document. Combine them into one coherent, concise summary.

Partial summaries:
{{ .summaries }}

Final summary:`

// NewMapReduceSummarizer creates a summarizer with the default prompts and
// concurrency of 4.
func NewMapReduceSummarizer(model llm.LLM, opts ...llm.Option) *MapReduceSummarizer {
	return &MapReduceSummarizer{
		LLM:          model,
		LLMOptions:   opts,
		MapPrompt:    DefaultMapPrompt,
		ReducePrompt: DefaultReducePrompt,
		Concurrency:  4,
		Name:         "MapReduceSummarizer",
	}
}

func (s *MapReduceSummarizer) Invoke(ctx context.Context, input any) (any, error) {
	docs, err := toDocuments(input)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", s.Name, err)
	}
	if len(docs) == 0 {
		return "", nil
	}
	if len(docs) == 1 {
		return s.summarise(ctx, s.MapPrompt, "content", docs[0].PageContent)
	}

	// Map phase
	summaries := make([]string, len(docs))
	sem := make(chan struct{}, max(1, s.Concurrency))
	var wg sync.WaitGroup
	var firstErr error
	var errMu sync.Mutex

	for i, d := range docs {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, content string) {
			defer wg.Done()
			defer func() { <-sem }()
			out, err := s.summarise(ctx, s.MapPrompt, "content", content)
			if err != nil {
				errMu.Lock()
				if firstErr == nil {
					firstErr = err
				}
				errMu.Unlock()
				return
			}
			summaries[i] = out
		}(i, d.PageContent)
	}
	wg.Wait()
	if firstErr != nil {
		return nil, fmt.Errorf("%s: map: %w", s.Name, firstErr)
	}

	// Reduce phase
	var joined strings.Builder
	for i, sm := range summaries {
		fmt.Fprintf(&joined, "[%d] %s\n\n", i+1, sm)
	}
	return s.summarise(ctx, s.ReducePrompt, "summaries", strings.TrimSpace(joined.String()))
}

func (s *MapReduceSummarizer) summarise(ctx context.Context, tmpl, key, value string) (string, error) {
	p := strings.ReplaceAll(tmpl, "{{ ."+key+" }}", value)
	gen, err := s.LLM.Generate(ctx, []schema.Message{schema.NewHumanMessage(p)}, s.LLMOptions...)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(gen.Text), nil
}

func (s *MapReduceSummarizer) Stream(ctx context.Context, input any) (<-chan StreamChunk, error) {
	ch := make(chan StreamChunk, 1)
	go func() {
		defer close(ch)
		out, err := s.Invoke(ctx, input)
		if err != nil {
			ch <- StreamChunk{Err: err}
			return
		}
		ch <- StreamChunk{Value: out, Done: true}
	}()
	return ch, nil
}

func (s *MapReduceSummarizer) Pipe(next Runnable) Runnable {
	return &pipeRunnable{first: s, second: next}
}

func (s *MapReduceSummarizer) Batch(ctx context.Context, inputs []any) ([]any, error) {
	return RunBatch(ctx, s, inputs)
}

// ---------------------------------------------------------------------------
// RefineSummarizer
// ---------------------------------------------------------------------------

// RefineSummarizer produces an initial summary from the first document, then
// walks through the remaining documents in order, refining the running summary.
// Preserves narrative order — pair with a splitter that keeps chunks ordered.
type RefineSummarizer struct {
	LLM           llm.LLM
	LLMOptions    []llm.Option
	InitialPrompt string // must contain {{ .content }}
	RefinePrompt  string // must contain {{ .existing }} and {{ .content }}
	Name          string
}

// DefaultInitialPrompt bootstraps the running summary from the first chunk.
const DefaultInitialPrompt = `Write a concise summary of the following text:

{{ .content }}

Concise summary:`

// DefaultRefinePrompt updates an existing summary with the next chunk.
const DefaultRefinePrompt = `You have an existing summary:

{{ .existing }}

Refine the summary above to incorporate the additional context below. Return only the updated summary — do not describe your changes.

Additional context:
{{ .content }}

Updated summary:`

// NewRefineSummarizer creates a RefineSummarizer with defaults.
func NewRefineSummarizer(model llm.LLM, opts ...llm.Option) *RefineSummarizer {
	return &RefineSummarizer{
		LLM:           model,
		LLMOptions:    opts,
		InitialPrompt: DefaultInitialPrompt,
		RefinePrompt:  DefaultRefinePrompt,
		Name:          "RefineSummarizer",
	}
}

func (s *RefineSummarizer) Invoke(ctx context.Context, input any) (any, error) {
	docs, err := toDocuments(input)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", s.Name, err)
	}
	if len(docs) == 0 {
		return "", nil
	}
	current, err := s.callTemplate(ctx, s.InitialPrompt, map[string]string{"content": docs[0].PageContent})
	if err != nil {
		return nil, fmt.Errorf("%s: initial: %w", s.Name, err)
	}
	for i, d := range docs[1:] {
		current, err = s.callTemplate(ctx, s.RefinePrompt, map[string]string{
			"existing": current,
			"content":  d.PageContent,
		})
		if err != nil {
			return nil, fmt.Errorf("%s: refine %d: %w", s.Name, i+1, err)
		}
	}
	return current, nil
}

func (s *RefineSummarizer) callTemplate(ctx context.Context, tmpl string, vars map[string]string) (string, error) {
	p := tmpl
	for k, v := range vars {
		p = strings.ReplaceAll(p, "{{ ."+k+" }}", v)
	}
	gen, err := s.LLM.Generate(ctx, []schema.Message{schema.NewHumanMessage(p)}, s.LLMOptions...)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(gen.Text), nil
}

func (s *RefineSummarizer) Stream(ctx context.Context, input any) (<-chan StreamChunk, error) {
	ch := make(chan StreamChunk, 1)
	go func() {
		defer close(ch)
		out, err := s.Invoke(ctx, input)
		if err != nil {
			ch <- StreamChunk{Err: err}
			return
		}
		ch <- StreamChunk{Value: out, Done: true}
	}()
	return ch, nil
}

func (s *RefineSummarizer) Pipe(next Runnable) Runnable {
	return &pipeRunnable{first: s, second: next}
}

func (s *RefineSummarizer) Batch(ctx context.Context, inputs []any) ([]any, error) {
	return RunBatch(ctx, s, inputs)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func toDocuments(input any) ([]schema.Document, error) {
	switch v := input.(type) {
	case []schema.Document:
		return v, nil
	case []string:
		out := make([]schema.Document, len(v))
		for i, s := range v {
			out[i] = schema.Document{PageContent: s}
		}
		return out, nil
	case string:
		return []schema.Document{{PageContent: v}}, nil
	case schema.Document:
		return []schema.Document{v}, nil
	default:
		return nil, fmt.Errorf("expected []schema.Document, got %T", input)
	}
}
