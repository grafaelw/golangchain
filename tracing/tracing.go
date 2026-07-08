// Package tracing provides LangSmith-inspired observability for golangchain.
//
// It ships four ready-made callbacks.Handler implementations:
//
//   - [TracerHandler] — records every run (LLM call, chain, tool, graph node)
//     into an in-memory tree of [Run] objects. Use [Tracer.Runs] to inspect
//     results, [Tracer.ExportJSON] to dump them, or [Tracer.Summary] for a
//     compact text report.
//
//   - [PrettyHandler] — writes a live, colour-coded trace to any io.Writer
//     (typically os.Stderr) as operations execute, giving a terminal
//     experience similar to LangSmith's run view.
//
//   - [JSONLinesExporter] — writes one JSON [Event] per line to any
//     io.Writer or file. Ideal for shipping traces into jq, Loki,
//     ClickHouse, DuckDB, or an OpenTelemetry collector's file receiver.
//     Use [NewFileJSONLinesExporter] for append-only file logging.
//
// A [FeedbackStore] is also provided for capturing user- or evaluator-supplied
// [Feedback] records keyed by run ID, mirroring LangSmith's Feedback API.
//
// Quick-start:
//
//	import (
//	    "os"
//	    "github.com/grafaelw/golangchain/callbacks"
//	    "github.com/grafaelw/golangchain/tracing"
//	)
//
//	cb := callbacks.NewCallbackManager(
//	    tracing.NewPrettyHandler(os.Stderr),   // live output
//	    tracing.NewJSONLinesExporter(logFile), // structured, per-line
//	)
//
//	// — or — record programmatically:
//	tracer := tracing.NewTracer()
//	cb := callbacks.NewCallbackManager(tracer.Handler())
//	// … run your chain …
//	tracer.ExportJSON(os.Stdout)
package tracing

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/grafaelw/golangchain/callbacks"
	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// Run — a single traced operation
// ---------------------------------------------------------------------------

// RunType classifies what kind of component produced a Run.
type RunType string

const (
	RunTypeLLM   RunType = "llm"
	RunTypeChain RunType = "chain"
	RunTypeTool  RunType = "tool"
	RunTypeAgent RunType = "agent"
	RunTypeGraph RunType = "graph"
)

// Run represents one traced operation with timing, inputs/outputs, and token
// usage. Runs form a tree via ParentID; use [Tracer.Runs] to get the roots.
type Run struct {
	ID        string            `json:"id"`
	ParentID  string            `json:"parent_id,omitempty"`
	Type      RunType           `json:"type"`
	Name      string            `json:"name"`
	Inputs    map[string]any    `json:"inputs,omitempty"`
	Outputs   map[string]any    `json:"outputs,omitempty"`
	Error     string            `json:"error,omitempty"`
	StartTime time.Time         `json:"start_time"`
	EndTime   time.Time         `json:"end_time,omitempty"`
	Usage     schema.TokenUsage `json:"usage,omitempty"`
	Children  []*Run            `json:"children,omitempty"`
}

// Duration returns the wall-clock time for this run. If the run has not
// finished yet, it returns the elapsed time since start.
func (r *Run) Duration() time.Duration {
	if r.EndTime.IsZero() {
		return time.Since(r.StartTime)
	}
	return r.EndTime.Sub(r.StartTime)
}

// IsFinished reports whether the run has ended (success or error).
func (r *Run) IsFinished() bool { return !r.EndTime.IsZero() }

// ---------------------------------------------------------------------------
// Tracer — collects Runs from callback events
// ---------------------------------------------------------------------------

// Tracer records all runs emitted by golangchain components and assembles
// them into a hierarchical tree. It is safe for concurrent use.
type Tracer struct {
	mu   sync.Mutex
	runs map[string]*Run
}

// NewTracer creates a new, empty Tracer.
func NewTracer() *Tracer {
	return &Tracer{runs: make(map[string]*Run)}
}

// Handler returns a [TracerHandler] that feeds events into this Tracer.
// Add it to a [callbacks.CallbackManager] to start recording.
func (t *Tracer) Handler() *TracerHandler {
	return &TracerHandler{tracer: t}
}

func (t *Tracer) startRun(id, parentID string, rt RunType, name string, inputs map[string]any) {
	if id == "" {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.runs[id] = &Run{
		ID:        id,
		ParentID:  parentID,
		Type:      rt,
		Name:      name,
		Inputs:    inputs,
		StartTime: time.Now(),
	}
}

func (t *Tracer) endRun(id string, outputs map[string]any, usage schema.TokenUsage) {
	if id == "" {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if r, ok := t.runs[id]; ok {
		r.EndTime = time.Now()
		r.Outputs = outputs
		r.Usage = usage
	}
}

func (t *Tracer) failRun(id, errStr string) {
	if id == "" {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if r, ok := t.runs[id]; ok {
		r.EndTime = time.Now()
		r.Error = errStr
	}
}

// Runs returns the complete run forest (roots with Children populated).
// The returned slice contains only top-level (parentless) runs; each Run's
// Children field contains its direct children, recursively.
func (t *Tracer) Runs() []*Run {
	t.mu.Lock()
	defer t.mu.Unlock()

	// Build parent → children map
	childMap := make(map[string][]*Run, len(t.runs))
	for _, r := range t.runs {
		childMap[r.ParentID] = append(childMap[r.ParentID], r)
	}
	// Populate each run's Children slice
	for _, r := range t.runs {
		r.Children = childMap[r.ID]
	}
	return childMap[""]
}

// ExportJSON writes the full run tree as indented JSON to w.
func (t *Tracer) ExportJSON(w io.Writer) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(t.Runs())
}

// Reset clears all recorded runs.
func (t *Tracer) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.runs = make(map[string]*Run)
}

// TotalTokens returns the sum of token usage across all LLM runs.
func (t *Tracer) TotalTokens() schema.TokenUsage {
	t.mu.Lock()
	defer t.mu.Unlock()
	var total schema.TokenUsage
	for _, r := range t.runs {
		if r.Type == RunTypeLLM {
			total.PromptTokens += r.Usage.PromptTokens
			total.CompletionTokens += r.Usage.CompletionTokens
			total.TotalTokens += r.Usage.TotalTokens
		}
	}
	return total
}

// Summary returns a human-readable multi-line text summary of all finished
// runs, ordered by start time, with indentation reflecting the run hierarchy.
func (t *Tracer) Summary() string {
	roots := t.Runs()
	var sb strings.Builder
	for _, r := range roots {
		writeSummary(&sb, r, 0)
	}
	return sb.String()
}

func writeSummary(sb *strings.Builder, r *Run, depth int) {
	pad := strings.Repeat("  ", depth)
	status := "ok"
	if r.Error != "" {
		status = "ERR: " + r.Error
	}
	tok := ""
	if r.Usage.TotalTokens > 0 {
		tok = fmt.Sprintf("  ↑%d ↓%d tok", r.Usage.PromptTokens, r.Usage.CompletionTokens)
	}
	fmt.Fprintf(sb, "%s[%s] %s  %s%s  %s\n",
		pad, r.Type, r.Name, fmtDur(r.Duration()), tok, status)
	for _, child := range r.Children {
		writeSummary(sb, child, depth+1)
	}
}

func fmtDur(d time.Duration) string {
	switch {
	case d < time.Millisecond:
		return fmt.Sprintf("%dµs", d.Microseconds())
	case d < time.Second:
		return fmt.Sprintf("%dms", d.Milliseconds())
	default:
		return fmt.Sprintf("%.2fs", d.Seconds())
	}
}

// ---------------------------------------------------------------------------
// TracerHandler — implements callbacks.Handler
// ---------------------------------------------------------------------------

// TracerHandler is a [callbacks.Handler] that feeds all lifecycle events into
// a [Tracer]. Each event type maps to a Run start or end:
//
//   - OnChainStart / OnChainEnd   → RunTypeChain
//   - OnLLMStart   / OnLLMEnd     → RunTypeLLM
//   - OnToolStart  / OnToolEnd    → RunTypeTool
//   - OnGraphNodeStart / OnGraphNodeEnd → RunTypeGraph
//   - OnError                     → marks the active run as failed
type TracerHandler struct {
	callbacks.NoOpHandler
	tracer *Tracer
}

func (h *TracerHandler) OnChainStart(ctx context.Context, name string, inputs map[string]any) {
	h.tracer.startRun(
		callbacks.RunIDFromContext(ctx),
		callbacks.ParentRunIDFromContext(ctx),
		RunTypeChain, name, inputs,
	)
}

func (h *TracerHandler) OnChainEnd(ctx context.Context, _ string, outputs map[string]any) {
	h.tracer.endRun(callbacks.RunIDFromContext(ctx), outputs, schema.TokenUsage{})
}

func (h *TracerHandler) OnLLMStart(ctx context.Context, model string, msgs []schema.Message) {
	h.tracer.startRun(
		callbacks.RunIDFromContext(ctx),
		callbacks.ParentRunIDFromContext(ctx),
		RunTypeLLM, model, map[string]any{"messages": summariseMsgs(msgs)},
	)
}

func (h *TracerHandler) OnLLMEnd(ctx context.Context, _ string, gen *schema.Generation) {
	h.tracer.endRun(
		callbacks.RunIDFromContext(ctx),
		map[string]any{"text": gen.Text, "stop_reason": gen.StopReason},
		gen.Usage,
	)
}

func (h *TracerHandler) OnToolStart(ctx context.Context, name, input string) {
	h.tracer.startRun(
		callbacks.RunIDFromContext(ctx),
		callbacks.ParentRunIDFromContext(ctx),
		RunTypeTool, name, map[string]any{"input": input},
	)
}

func (h *TracerHandler) OnToolEnd(ctx context.Context, _ string, output string) {
	h.tracer.endRun(
		callbacks.RunIDFromContext(ctx),
		map[string]any{"output": output},
		schema.TokenUsage{},
	)
}

func (h *TracerHandler) OnGraphNodeStart(ctx context.Context, graphName, nodeName string) {
	h.tracer.startRun(
		callbacks.RunIDFromContext(ctx),
		callbacks.ParentRunIDFromContext(ctx),
		RunTypeGraph,
		fmt.Sprintf("%s/%s", graphName, nodeName),
		nil,
	)
}

func (h *TracerHandler) OnGraphNodeEnd(ctx context.Context, _, _ string) {
	h.tracer.endRun(callbacks.RunIDFromContext(ctx), nil, schema.TokenUsage{})
}

func (h *TracerHandler) OnError(ctx context.Context, _ string, err error) {
	h.tracer.failRun(callbacks.RunIDFromContext(ctx), err.Error())
}

// summariseMsgs converts a message slice into a lightweight representation
// suitable for JSON storage without embedding large content verbatim.
func summariseMsgs(msgs []schema.Message) []map[string]string {
	out := make([]map[string]string, len(msgs))
	for i, m := range msgs {
		content := m.Content
		if len(content) > 200 {
			content = content[:200] + "…"
		}
		out[i] = map[string]string{
			"role":    string(m.Role),
			"content": content,
		}
	}
	return out
}
