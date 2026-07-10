// This file adds retrieval-augmented and summarization chains that live
// alongside LLMChain, SequentialChain, MapChain, and RouterChain.
//
//   - RetrievalQAChain:             RAG — retrieve documents, stuff them into a
//                                   prompt, ask the LLM.
//   - ConversationalRetrievalChain: Multi-turn RAG with question reformulation
//                                   and chat history.
//   - MapReduceSummarizer:          summarise each chunk in parallel, then reduce.
//   - RefineSummarizer:             seed a summary and iteratively refine it chunk
//                                   by chunk. Best when order matters.

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

func (c *RetrievalQAChain) Stream(ctx context.Context, input any) (<-chan schema.StreamChunk, error) {
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
	out := make(chan schema.StreamChunk, 32)
	go func() {
		defer close(out)
		for chunk := range llmCh {
			if chunk.Err != nil {
				out <- chunk
				return
			}
			chunk.Value = chunk.Text
			out <- chunk
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
// ConversationalRetrievalChain — multi-turn RAG with question reformulation
// ---------------------------------------------------------------------------

// ConversationalRetrievalChain extends RAG to multi-turn conversations.
// Each new question is reformulated into a standalone query using the chat
// history, then documents are retrieved and an answer is generated with the
// full context.
//
//	chain := NewConversationalRetrievalChain(retriever, model,
//	    WithMemory(memory.NewConversationBufferMemory()),
//	)
//	ans, _ := chain.Invoke(ctx, "Explain more about that.")
type ConversationalRetrievalChain struct {
	Retriever      retriever.Retriever
	LLM            llm.LLM
	LLMOptions     []llm.Option
	Memory         Memory
	CondensePrompt string
	AnswerPrompt   string
	ReturnSource   bool
	Name           string
}

// Memory is a minimal interface used by ConversationalRetrievalChain.
// Any memory.Memory implementation satisfies this interface.
type Memory interface {
	LoadMemoryVariables(ctx context.Context) (map[string]any, error)
	SaveContext(ctx context.Context, humanInput, aiOutput string) error
}

// DefaultCondenseQuestionPrompt reformulates a follow-up question given chat history.
const DefaultCondenseQuestionPrompt = `Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question that captures all needed context from the conversation.

Chat History:
{{ .chat_history }}

Follow-up question: {{ .question }}

Standalone question:`

// DefaultConversationalAnswerPrompt answers a question with retrieved context.
const DefaultConversationalAnswerPrompt = `You are a helpful assistant. Answer the question using only the provided context. If the answer is not in the context, say you don't know.

Context:
{{ .context }}

Question: {{ .question }}

Answer:`

// ConvRetrievalOption configures a ConversationalRetrievalChain.
type ConvRetrievalOption func(*ConversationalRetrievalChain)

// WithMemory sets the conversation memory for the chain.
func WithMemory(mem Memory) ConvRetrievalOption {
	return func(c *ConversationalRetrievalChain) { c.Memory = mem }
}

// WithCondensePrompt overrides the question reformulation prompt.
func WithCondensePrompt(p string) ConvRetrievalOption {
	return func(c *ConversationalRetrievalChain) { c.CondensePrompt = p }
}

// WithAnswerPrompt overrides the answer-generation prompt.
func WithAnswerPrompt(p string) ConvRetrievalOption {
	return func(c *ConversationalRetrievalChain) { c.AnswerPrompt = p }
}

// NewConversationalRetrievalChain creates a multi-turn RAG chain.
func NewConversationalRetrievalChain(r retriever.Retriever, model llm.LLM, opts ...ConvRetrievalOption) *ConversationalRetrievalChain {
	c := &ConversationalRetrievalChain{
		Retriever:      r,
		LLM:            model,
		CondensePrompt: DefaultCondenseQuestionPrompt,
		AnswerPrompt:   DefaultConversationalAnswerPrompt,
		Name:           "ConversationalRetrievalChain",
	}
	for _, o := range opts {
		o(c)
	}
	return c
}

func (c *ConversationalRetrievalChain) Invoke(ctx context.Context, input any) (any, error) {
	question, err := extractQuestion(input)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", c.Name, err)
	}

	standalone := question

	if c.Memory != nil {
		vars, err := c.Memory.LoadMemoryVariables(ctx)
		if err != nil {
			return nil, fmt.Errorf("%s: load memory: %w", c.Name, err)
		}
		history, ok := vars["history"]
		if ok {
			historyStr := formatChatHistory(history)
			if historyStr != "" {
				p := strings.ReplaceAll(c.CondensePrompt, "{{ .chat_history }}", historyStr)
				p = strings.ReplaceAll(p, "{{ .question }}", question)
				gen, err := c.LLM.Generate(ctx, []schema.Message{schema.NewHumanMessage(p)}, c.LLMOptions...)
				if err != nil {
					return nil, fmt.Errorf("%s: condense: %w", c.Name, err)
				}
				standalone = strings.TrimSpace(gen.Text)
			}
		}
	}

	docs, err := c.Retriever.GetRelevantDocuments(ctx, standalone)
	if err != nil {
		return nil, fmt.Errorf("%s: retrieve: %w", c.Name, err)
	}

	prompt := renderRAGPrompt(c.AnswerPrompt, question, docs)
	gen, err := c.LLM.Generate(ctx, []schema.Message{schema.NewHumanMessage(prompt)}, c.LLMOptions...)
	if err != nil {
		return nil, fmt.Errorf("%s: llm: %w", c.Name, err)
	}
	answer := strings.TrimSpace(gen.Text)

	if c.Memory != nil {
		_ = c.Memory.SaveContext(ctx, question, answer)
	}

	if c.ReturnSource {
		return map[string]any{"answer": answer, "sources": docs, "standalone_question": standalone}, nil
	}
	return answer, nil
}

func (c *ConversationalRetrievalChain) Stream(ctx context.Context, input any) (<-chan schema.StreamChunk, error) {
	question, err := extractQuestion(input)
	if err != nil {
		return nil, err
	}

	standalone := question
	if c.Memory != nil {
		vars, err := c.Memory.LoadMemoryVariables(ctx)
		if err != nil {
			return nil, err
		}
		if history, ok := vars["history"]; ok {
			historyStr := formatChatHistory(history)
			if historyStr != "" {
				p := strings.ReplaceAll(c.CondensePrompt, "{{ .chat_history }}", historyStr)
				p = strings.ReplaceAll(p, "{{ .question }}", question)
				gen, err := c.LLM.Generate(ctx, []schema.Message{schema.NewHumanMessage(p)}, c.LLMOptions...)
				if err != nil {
					return nil, err
				}
				standalone = strings.TrimSpace(gen.Text)
			}
		}
	}

	docs, err := c.Retriever.GetRelevantDocuments(ctx, standalone)
	if err != nil {
		return nil, err
	}

	prompt := renderRAGPrompt(c.AnswerPrompt, question, docs)
	llmCh, err := c.LLM.Stream(ctx, []schema.Message{schema.NewHumanMessage(prompt)}, c.LLMOptions...)
	if err != nil {
		return nil, err
	}

	out := make(chan schema.StreamChunk, 32)
	go func() {
		defer close(out)
		var answer strings.Builder
		for chunk := range llmCh {
			if chunk.Err != nil {
				out <- chunk
				return
			}
			answer.WriteString(chunk.Text)
			chunk.Value = chunk.Text
			out <- chunk
		}
		if c.Memory != nil {
			_ = c.Memory.SaveContext(ctx, question, answer.String())
		}
	}()
	return out, nil
}

func (c *ConversationalRetrievalChain) Pipe(next Runnable) Runnable {
	return &pipeRunnable{first: c, second: next}
}

func (c *ConversationalRetrievalChain) Batch(ctx context.Context, inputs []any) ([]any, error) {
	return RunBatch(ctx, c, inputs)
}

func formatChatHistory(history any) string {
	switch v := history.(type) {
	case string:
		return v
	case []schema.Message:
		var sb strings.Builder
		for _, m := range v {
			sb.WriteString(string(m.Role))
			sb.WriteString(": ")
			sb.WriteString(m.Content)
			sb.WriteString("\n")
		}
		return strings.TrimSpace(sb.String())
	case []any:
		var sb strings.Builder
		for _, item := range v {
			fmt.Fprint(&sb, item)
			sb.WriteString("\n")
		}
		return strings.TrimSpace(sb.String())
	default:
		return fmt.Sprint(v)
	}
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

func (s *MapReduceSummarizer) Stream(ctx context.Context, input any) (<-chan schema.StreamChunk, error) {
	ch := make(chan schema.StreamChunk, 1)
	go func() {
		defer close(ch)
		out, err := s.Invoke(ctx, input)
		if err != nil {
			ch <- schema.StreamChunk{Err: err}
			return
		}
		ch <- schema.StreamChunk{Value: out, Done: true}
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

func (s *RefineSummarizer) Stream(ctx context.Context, input any) (<-chan schema.StreamChunk, error) {
	ch := make(chan schema.StreamChunk, 1)
	go func() {
		defer close(ch)
		out, err := s.Invoke(ctx, input)
		if err != nil {
			ch <- schema.StreamChunk{Err: err}
			return
		}
		ch <- schema.StreamChunk{Value: out, Done: true}
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
