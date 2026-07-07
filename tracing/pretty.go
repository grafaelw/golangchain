package tracing

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/grafaelw/golangchain/callbacks"
	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// ANSI colour helpers
// ---------------------------------------------------------------------------

const (
	ansiReset   = "\033[0m"
	ansiBold    = "\033[1m"
	ansiDim     = "\033[2m"
	ansiBlue    = "\033[34m" // LLM
	ansiCyan    = "\033[36m" // Chain
	ansiYellow  = "\033[33m" // Tool
	ansiGreen   = "\033[32m" // success / agent finish
	ansiMagenta = "\033[35m" // Graph node
	ansiRed     = "\033[31m" // Error
	ansiGray    = "\033[90m" // dim metadata
)

// ---------------------------------------------------------------------------
// PrettyHandler
// ---------------------------------------------------------------------------

type spanInfo struct {
	depth     int
	startTime time.Time
	name      string
	runType   RunType
}

// PrettyHandler is a [callbacks.Handler] that writes a live, colour-coded
// trace to an [io.Writer] as operations execute — similar to what LangSmith
// shows in its run timeline, but directly in your terminal.
//
// Each component type is colour-coded:
//   - LLM calls        → blue
//   - Chain runs       → cyan
//   - Tool calls       → yellow
//   - Graph nodes      → magenta
//   - Errors           → red
//   - Success markers  → green
//
// Nesting is reflected with two-space indentation per level, derived from
// the run-ID hierarchy injected by the framework.
//
// Usage:
//
//	cb := callbacks.NewCallbackManager(
//	    tracing.NewPrettyHandler(os.Stderr),
//	)
type PrettyHandler struct {
	callbacks.NoOpHandler
	mu      sync.Mutex
	active  map[string]*spanInfo // runID → open span
	w       io.Writer
	noColor bool
}

// NewPrettyHandler creates a PrettyHandler that writes to w.
// Pass os.Stderr to keep trace output separate from program stdout.
// If w is nil, os.Stderr is used.
func NewPrettyHandler(w io.Writer) *PrettyHandler {
	if w == nil {
		w = os.Stderr
	}
	return &PrettyHandler{
		active: make(map[string]*spanInfo),
		w:      w,
	}
}

// WithoutColor disables ANSI colour codes. Useful for CI pipelines or log
// files that do not support terminal escape sequences.
func (h *PrettyHandler) WithoutColor() *PrettyHandler {
	h.noColor = true
	return h
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

func (h *PrettyHandler) c(code, s string) string {
	if h.noColor {
		return s
	}
	return code + s + ansiReset
}

// depthOf computes the display depth for a new child span.
//   - parentID == ""              → depth 0  (root span)
//   - parentID found in active   → parent.depth + 1
//   - parentID set but not found → depth 1  (parent not tracked, e.g. agent
//     executor run ID)
func (h *PrettyHandler) depthOf(parentID string) int {
	if parentID == "" {
		return 0
	}
	if p, ok := h.active[parentID]; ok {
		return p.depth + 1
	}
	return 1
}

func (h *PrettyHandler) pad(depth int) string {
	return strings.Repeat("  ", depth)
}

func (h *PrettyHandler) fmtDuration(d time.Duration) string {
	var s string
	switch {
	case d < time.Millisecond:
		s = fmt.Sprintf("%dµs", d.Microseconds())
	case d < time.Second:
		s = fmt.Sprintf("%dms", d.Milliseconds())
	default:
		s = fmt.Sprintf("%.2fs", d.Seconds())
	}
	if h.noColor {
		return s
	}
	switch {
	case d < 200*time.Millisecond:
		return ansiGreen + s + ansiReset
	case d < time.Second:
		return ansiYellow + s + ansiReset
	default:
		return ansiRed + s + ansiReset
	}
}

// truncate shortens s to at most n runes, replacing newlines with spaces.
func truncate(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.TrimSpace(s)
	if utf8.RuneCountInString(s) <= n {
		return s
	}
	runes := []rune(s)
	return string(runes[:n]) + "…"
}

func (h *PrettyHandler) openSpan(runID, parentID string, rt RunType, name string) int {
	depth := h.depthOf(parentID)
	h.active[runID] = &spanInfo{
		depth:     depth,
		startTime: time.Now(),
		name:      name,
		runType:   rt,
	}
	return depth
}

func (h *PrettyHandler) closeSpan(runID string) *spanInfo {
	info := h.active[runID]
	if info != nil {
		delete(h.active, runID)
	}
	return info
}

// ---------------------------------------------------------------------------
// LLM events
// ---------------------------------------------------------------------------

func (h *PrettyHandler) OnLLMStart(ctx context.Context, model string, msgs []schema.Message) {
	runID := callbacks.RunIDFromContext(ctx)
	parentID := callbacks.ParentRunIDFromContext(ctx)

	h.mu.Lock()
	depth := h.openSpan(runID, parentID, RunTypeLLM, model)
	p := h.pad(depth)
	h.mu.Unlock()

	fmt.Fprintf(h.w, "%s%s %s  %s\n",
		p,
		h.c(ansiBold+ansiBlue, "> LLM"),
		h.c(ansiBlue, model),
		h.c(ansiGray, fmt.Sprintf("[%d messages]", len(msgs))),
	)
}

func (h *PrettyHandler) OnLLMEnd(ctx context.Context, model string, gen *schema.Generation) {
	runID := callbacks.RunIDFromContext(ctx)

	h.mu.Lock()
	info := h.closeSpan(runID)
	h.mu.Unlock()

	if info == nil {
		return
	}
	dur := time.Since(info.startTime)
	p := h.pad(info.depth)

	tok := ""
	if gen.Usage.TotalTokens > 0 {
		tok = h.c(ansiGray, fmt.Sprintf("  ↑%d ↓%d tok",
			gen.Usage.PromptTokens, gen.Usage.CompletionTokens))
	}
	preview := ""
	if gen.Text != "" {
		preview = h.c(ansiDim, fmt.Sprintf(`  "%s"`, truncate(gen.Text, 80)))
	}

	fmt.Fprintf(h.w, "%s%s %s  %s%s%s\n",
		p,
		h.c(ansiGreen, "✓ LLM"),
		h.c(ansiBlue, model),
		h.fmtDuration(dur),
		tok, preview,
	)
}

// ---------------------------------------------------------------------------
// Chain events
// ---------------------------------------------------------------------------

func (h *PrettyHandler) OnChainStart(ctx context.Context, name string, _ map[string]any) {
	runID := callbacks.RunIDFromContext(ctx)
	parentID := callbacks.ParentRunIDFromContext(ctx)

	h.mu.Lock()
	depth := h.openSpan(runID, parentID, RunTypeChain, name)
	p := h.pad(depth)
	h.mu.Unlock()

	fmt.Fprintf(h.w, "%s%s %s\n",
		p,
		h.c(ansiBold+ansiCyan, "> Chain"),
		h.c(ansiCyan, name),
	)
}

func (h *PrettyHandler) OnChainEnd(ctx context.Context, name string, _ map[string]any) {
	runID := callbacks.RunIDFromContext(ctx)

	h.mu.Lock()
	info := h.closeSpan(runID)
	h.mu.Unlock()

	if info == nil {
		return
	}
	dur := time.Since(info.startTime)
	p := h.pad(info.depth)

	fmt.Fprintf(h.w, "%s%s %s  %s\n",
		p,
		h.c(ansiGreen, "✓ Chain"),
		h.c(ansiCyan, name),
		h.fmtDuration(dur),
	)
}

// ---------------------------------------------------------------------------
// Tool events
// ---------------------------------------------------------------------------

func (h *PrettyHandler) OnToolStart(ctx context.Context, name, input string) {
	runID := callbacks.RunIDFromContext(ctx)
	parentID := callbacks.ParentRunIDFromContext(ctx)

	h.mu.Lock()
	depth := h.openSpan(runID, parentID, RunTypeTool, name)
	p := h.pad(depth)
	h.mu.Unlock()

	fmt.Fprintf(h.w, "%s%s %s  %s\n",
		p,
		h.c(ansiBold+ansiYellow, "> Tool"),
		h.c(ansiYellow, name),
		h.c(ansiGray, fmt.Sprintf(`"%s"`, truncate(input, 60))),
	)
}

func (h *PrettyHandler) OnToolEnd(ctx context.Context, name, output string) {
	runID := callbacks.RunIDFromContext(ctx)

	h.mu.Lock()
	info := h.closeSpan(runID)
	h.mu.Unlock()

	if info == nil {
		return
	}
	dur := time.Since(info.startTime)
	p := h.pad(info.depth)

	fmt.Fprintf(h.w, "%s%s %s  %s  %s\n",
		p,
		h.c(ansiGreen, "✓ Tool"),
		h.c(ansiYellow, name),
		h.fmtDuration(dur),
		h.c(ansiDim, fmt.Sprintf(`"%s"`, truncate(output, 80))),
	)
}

// ---------------------------------------------------------------------------
// Agent events  (point-in-time, not spans)
// ---------------------------------------------------------------------------

func (h *PrettyHandler) OnAgentAction(ctx context.Context, action schema.AgentAction) {
	// Determine depth from the agent's context run ID.
	parentID := callbacks.RunIDFromContext(ctx)
	h.mu.Lock()
	depth := h.depthOf(parentID)
	h.mu.Unlock()

	p := h.pad(depth)
	fmt.Fprintf(h.w, "%s%s  %s(%s)\n",
		p,
		h.c(ansiBold+ansiGreen, "• Action"),
		h.c(ansiYellow, action.Tool),
		h.c(ansiGray, truncate(action.ToolInput, 60)),
	)
}

func (h *PrettyHandler) OnAgentFinish(ctx context.Context, finish schema.AgentFinish) {
	parentID := callbacks.RunIDFromContext(ctx)
	h.mu.Lock()
	depth := h.depthOf(parentID)
	h.mu.Unlock()

	p := h.pad(depth)
	fmt.Fprintf(h.w, "%s%s  %s\n",
		p,
		h.c(ansiGreen, "✓ Agent"),
		h.c(ansiDim, fmt.Sprintf(`"%s"`, truncate(finish.Output, 100))),
	)
}

// ---------------------------------------------------------------------------
// Graph events
// ---------------------------------------------------------------------------

func (h *PrettyHandler) OnGraphNodeStart(ctx context.Context, graphName, nodeName string) {
	runID := callbacks.RunIDFromContext(ctx)
	parentID := callbacks.ParentRunIDFromContext(ctx)

	h.mu.Lock()
	depth := h.openSpan(runID, parentID, RunTypeGraph, nodeName)
	p := h.pad(depth)
	h.mu.Unlock()

	fmt.Fprintf(h.w, "%s%s %s  %s\n",
		p,
		h.c(ansiBold+ansiMagenta, "> Node"),
		h.c(ansiMagenta, nodeName),
		h.c(ansiGray, graphName),
	)
}

func (h *PrettyHandler) OnGraphNodeEnd(ctx context.Context, _, nodeName string) {
	runID := callbacks.RunIDFromContext(ctx)

	h.mu.Lock()
	info := h.closeSpan(runID)
	h.mu.Unlock()

	if info == nil {
		return
	}
	dur := time.Since(info.startTime)
	p := h.pad(info.depth)

	fmt.Fprintf(h.w, "%s%s %s  %s\n",
		p,
		h.c(ansiGreen, "✓ Node"),
		h.c(ansiMagenta, nodeName),
		h.fmtDuration(dur),
	)
}

func (h *PrettyHandler) OnGraphCheckpoint(_ context.Context, _, threadID string) {
	fmt.Fprintf(h.w, "%s  thread=%s\n",
		h.c(ansiGray, "• Checkpoint"),
		threadID,
	)
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

func (h *PrettyHandler) OnError(ctx context.Context, source string, err error) {
	runID := callbacks.RunIDFromContext(ctx)

	h.mu.Lock()
	info := h.closeSpan(runID)
	depth := 0
	if info != nil {
		depth = info.depth
	}
	h.mu.Unlock()

	p := h.pad(depth)
	fmt.Fprintf(h.w, "%s%s  [%s] %v\n",
		p,
		h.c(ansiRed, "✗ Error"),
		source, err,
	)
}
