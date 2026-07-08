// This file adds two production-oriented tracing outputs alongside the
// in-memory Tracer and PrettyHandler:
//
//   - JSONLinesExporter: emits each traced event as a single JSON object per
//     line to any io.Writer. This is a light-weight, backend-agnostic format
//     that ingests cleanly into Loki, ClickHouse, DuckDB, jq, or a shell
//     pipeline. It is also the recommended way to feed golangchain traces
//     into an OpenTelemetry collector via the log/file receiver.
//
//   - FeedbackStore: an in-process store of user- or evaluator-supplied
//     Feedback records keyed by run ID, mirroring LangSmith's Feedback API.
//     The store is safe for concurrent use and can be flushed to JSON Lines.

package tracing

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync"
	"time"

	"github.com/grafaelw/golangchain/callbacks"
	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// Event
// ---------------------------------------------------------------------------

// EventKind identifies the specific callback method that produced an event.
type EventKind string

const (
	EventLLMStart        EventKind = "llm_start"
	EventLLMEnd          EventKind = "llm_end"
	EventLLMStream       EventKind = "llm_stream"
	EventChainStart      EventKind = "chain_start"
	EventChainEnd        EventKind = "chain_end"
	EventToolStart       EventKind = "tool_start"
	EventToolEnd         EventKind = "tool_end"
	EventAgentAction     EventKind = "agent_action"
	EventAgentFinish     EventKind = "agent_finish"
	EventGraphNodeStart  EventKind = "graph_node_start"
	EventGraphNodeEnd    EventKind = "graph_node_end"
	EventGraphCheckpoint EventKind = "graph_checkpoint"
	EventError           EventKind = "error"
)

// Event is a single record emitted by JSONLinesExporter. Compatible with any
// log-file receiver that understands JSON Lines.
type Event struct {
	Time     time.Time `json:"time"`
	Kind     EventKind `json:"kind"`
	RunID    string    `json:"run_id,omitempty"`
	ParentID string    `json:"parent_id,omitempty"`
	Name     string    `json:"name,omitempty"`
	Source   string    `json:"source,omitempty"`
	Data     any       `json:"data,omitempty"`
	Error    string    `json:"error,omitempty"`
}

// ---------------------------------------------------------------------------
// JSONLinesExporter
// ---------------------------------------------------------------------------

// JSONLinesExporter is a callbacks.Handler that writes each event as one
// JSON object per line to Out. Writes are serialised behind an internal
// mutex, so it is safe to share a single exporter across goroutines.
type JSONLinesExporter struct {
	callbacks.NoOpHandler
	Out io.Writer
	mu  sync.Mutex
}

// NewJSONLinesExporter constructs an exporter over out.
func NewJSONLinesExporter(out io.Writer) *JSONLinesExporter {
	return &JSONLinesExporter{Out: out}
}

// NewFileJSONLinesExporter opens (or creates+appends) a file and returns
// an exporter over it, along with the io.Closer for the file so callers
// can shut it down cleanly.
func NewFileJSONLinesExporter(path string) (*JSONLinesExporter, io.Closer, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, nil, fmt.Errorf("tracing: open %q: %w", path, err)
	}
	return NewJSONLinesExporter(f), f, nil
}

func (e *JSONLinesExporter) emit(ctx context.Context, ev Event) {
	ev.Time = time.Now().UTC()
	if ev.RunID == "" {
		ev.RunID = callbacks.RunIDFromContext(ctx)
	}
	if ev.ParentID == "" {
		ev.ParentID = callbacks.ParentRunIDFromContext(ctx)
	}
	data, err := json.Marshal(ev)
	if err != nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	_, _ = e.Out.Write(append(data, '\n'))
}

func (e *JSONLinesExporter) OnLLMStart(ctx context.Context, model string, msgs []schema.Message) {
	e.emit(ctx, Event{Kind: EventLLMStart, Name: model, Data: map[string]any{"messages": len(msgs)}})
}
func (e *JSONLinesExporter) OnLLMEnd(ctx context.Context, model string, gen *schema.Generation) {
	e.emit(ctx, Event{Kind: EventLLMEnd, Name: model, Data: map[string]any{
		"stop":        gen.StopReason,
		"usage":       gen.Usage,
		"text_length": len(gen.Text),
	}})
}
func (e *JSONLinesExporter) OnLLMStream(ctx context.Context, model string, c schema.StreamChunk) {
	if !c.Done {
		return
	}
	e.emit(ctx, Event{Kind: EventLLMStream, Name: model, Data: map[string]any{"done": true}})
}
func (e *JSONLinesExporter) OnChainStart(ctx context.Context, name string, inputs map[string]any) {
	e.emit(ctx, Event{Kind: EventChainStart, Name: name, Data: map[string]any{"input_keys": keys(inputs)}})
}
func (e *JSONLinesExporter) OnChainEnd(ctx context.Context, name string, outputs map[string]any) {
	e.emit(ctx, Event{Kind: EventChainEnd, Name: name, Data: map[string]any{"output_keys": keys(outputs)}})
}
func (e *JSONLinesExporter) OnToolStart(ctx context.Context, name, input string) {
	e.emit(ctx, Event{Kind: EventToolStart, Name: name, Data: map[string]any{"input": input}})
}
func (e *JSONLinesExporter) OnToolEnd(ctx context.Context, name, output string) {
	e.emit(ctx, Event{Kind: EventToolEnd, Name: name, Data: map[string]any{"output_length": len(output)}})
}
func (e *JSONLinesExporter) OnAgentAction(ctx context.Context, a schema.AgentAction) {
	e.emit(ctx, Event{Kind: EventAgentAction, Name: a.Tool, Data: map[string]any{"input": a.ToolInput}})
}
func (e *JSONLinesExporter) OnAgentFinish(ctx context.Context, f schema.AgentFinish) {
	e.emit(ctx, Event{Kind: EventAgentFinish, Data: map[string]any{"output_length": len(f.Output)}})
}
func (e *JSONLinesExporter) OnGraphNodeStart(ctx context.Context, graph, node string) {
	e.emit(ctx, Event{Kind: EventGraphNodeStart, Name: graph, Source: node})
}
func (e *JSONLinesExporter) OnGraphNodeEnd(ctx context.Context, graph, node string) {
	e.emit(ctx, Event{Kind: EventGraphNodeEnd, Name: graph, Source: node})
}
func (e *JSONLinesExporter) OnGraphCheckpoint(ctx context.Context, graph, threadID string) {
	e.emit(ctx, Event{Kind: EventGraphCheckpoint, Name: graph, Source: threadID})
}
func (e *JSONLinesExporter) OnError(ctx context.Context, source string, err error) {
	e.emit(ctx, Event{Kind: EventError, Source: source, Error: err.Error()})
}

func keys(m map[string]any) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

// ---------------------------------------------------------------------------
// Feedback
// ---------------------------------------------------------------------------

// Feedback captures a single evaluation signal attached to a run
// (thumbs-up, numeric score, free-form comment, evaluator result, …).
type Feedback struct {
	RunID     string    `json:"run_id"`
	Key       string    `json:"key"`
	Score     *float64  `json:"score,omitempty"`
	Value     any       `json:"value,omitempty"`
	Comment   string    `json:"comment,omitempty"`
	Source    string    `json:"source,omitempty"` // "user", "evaluator", …
	CreatedAt time.Time `json:"created_at"`
}

// FeedbackStore is an in-process, thread-safe store of Feedback keyed by RunID.
type FeedbackStore struct {
	mu   sync.RWMutex
	byID map[string][]Feedback
}

// NewFeedbackStore constructs an empty FeedbackStore.
func NewFeedbackStore() *FeedbackStore {
	return &FeedbackStore{byID: make(map[string][]Feedback)}
}

// Add records feedback (auto-sets CreatedAt if zero).
func (s *FeedbackStore) Add(f Feedback) {
	if f.CreatedAt.IsZero() {
		f.CreatedAt = time.Now().UTC()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.byID[f.RunID] = append(s.byID[f.RunID], f)
}

// Get returns all feedback for a run.
func (s *FeedbackStore) Get(runID string) []Feedback {
	s.mu.RLock()
	defer s.mu.RUnlock()
	cp := make([]Feedback, len(s.byID[runID]))
	copy(cp, s.byID[runID])
	return cp
}

// All returns every feedback record across every run.
func (s *FeedbackStore) All() []Feedback {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var out []Feedback
	for _, list := range s.byID {
		out = append(out, list...)
	}
	return out
}

// WriteJSONL dumps every stored feedback as JSON Lines to w.
func (s *FeedbackStore) WriteJSONL(w io.Writer) error {
	enc := json.NewEncoder(w)
	for _, f := range s.All() {
		if err := enc.Encode(f); err != nil {
			return err
		}
	}
	return nil
}
