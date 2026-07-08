package tracing

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

// Issue represents a detected problem in a trace.
type Issue struct {
	RunID    string `json:"run_id"`
	Severity string `json:"severity"`
	Title    string `json:"title"`
	Detail   string `json:"detail"`
	Fix      string `json:"fix,omitempty"`
}

// Engine analyzes traces and detects issues.
type Engine struct {
	tracer *Tracer
	issues []Issue
	mu     sync.RWMutex
}

// NewEngine creates an Engine that analyzes the given tracer.
func NewEngine(tracer *Tracer) *Engine {
	return &Engine{tracer: tracer}
}

// Analyze scans all runs in the tracer and detects:
//   - Tool call failures (error observations)
//   - High latency LLM calls (>10s)
//   - Repeated tool calls (same tool called 3+ times with similar input)
//   - Empty LLM responses
//   - Missing tool results
//   - Chain errors
//
// Returns detected issues.
func (e *Engine) Analyze(ctx context.Context) []Issue {
	_ = ctx

	roots := e.tracer.Runs()
	allRuns := flattenRuns(roots)

	e.mu.Lock()
	defer e.mu.Unlock()
	e.issues = nil

	for _, r := range allRuns {
		if r.Error != "" {
			switch r.Type {
			case RunTypeTool:
				e.issues = append(e.issues, Issue{
					RunID:    r.ID,
					Severity: "error",
					Title:    "Tool call failed",
					Detail:   fmt.Sprintf("Tool %q returned error: %s", r.Name, r.Error),
					Fix:      "Check tool input parameters and ensure external service is reachable.",
				})
			case RunTypeChain:
				e.issues = append(e.issues, Issue{
					RunID:    r.ID,
					Severity: "error",
					Title:    "Chain execution failed",
					Detail:   fmt.Sprintf("Chain %q failed: %s", r.Name, r.Error),
					Fix:      "Inspect chain steps and verify upstream dependencies.",
				})
			case RunTypeLLM:
				e.issues = append(e.issues, Issue{
					RunID:    r.ID,
					Severity: "error",
					Title:    "LLM call failed",
					Detail:   fmt.Sprintf("Model %q returned error: %s", r.Name, r.Error),
					Fix:      "Verify API key, model name, and rate limits.",
				})
			default:
				e.issues = append(e.issues, Issue{
					RunID:    r.ID,
					Severity: "error",
					Title:    fmt.Sprintf("%s run failed", r.Type),
					Detail:   fmt.Sprintf("%s %q: %s", r.Type, r.Name, r.Error),
				})
			}
			continue
		}

		if r.Type == RunTypeLLM && r.IsFinished() {
			if r.Duration() > 10*time.Second {
				e.issues = append(e.issues, Issue{
					RunID:    r.ID,
					Severity: "warning",
					Title:    "High latency LLM call",
					Detail:   fmt.Sprintf("Model %q took %s to respond.", r.Name, r.Duration()),
					Fix:      "Consider using a faster model, reducing prompt size, or enabling streaming.",
				})
			}
			if isLLMResponseEmpty(r) {
				e.issues = append(e.issues, Issue{
					RunID:    r.ID,
					Severity: "warning",
					Title:    "Empty LLM response",
					Detail:   fmt.Sprintf("Model %q returned an empty response.", r.Name),
					Fix:      "Check prompt content and model configuration.",
				})
			}
		}

		if r.Type == RunTypeTool && r.IsFinished() && isMissingToolResult(r) {
			e.issues = append(e.issues, Issue{
				RunID:    r.ID,
				Severity: "warning",
				Title:    "Missing tool result",
				Detail:   fmt.Sprintf("Tool %q completed with no output.", r.Name),
				Fix:      "Verify the tool implementation returns a result.",
			})
		}
	}

	e.detectRepeatedToolCalls(allRuns)

	return e.issues
}

// Report returns a human-readable summary of all issues.
func (e *Engine) Report() string {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if len(e.issues) == 0 {
		return "No issues detected.\n"
	}

	var sb strings.Builder
	fmt.Fprintf(&sb, "Trace Analysis — %d issue(s) found\n\n", len(e.issues))

	severityOrder := map[string]int{"error": 0, "warning": 1, "info": 2}
	sevLabel := map[string]string{"error": "ERROR", "warning": "WARN ", "info": "INFO "}

	for sev := 0; sev <= 2; sev++ {
		for _, issue := range e.issues {
			if severityOrder[issue.Severity] != sev {
				continue
			}
			fmt.Fprintf(&sb, "[%s] %s\n", sevLabel[issue.Severity], issue.Title)
			fmt.Fprintf(&sb, "      Run: %s\n", issue.RunID)
			fmt.Fprintf(&sb, "      %s\n", issue.Detail)
			if issue.Fix != "" {
				fmt.Fprintf(&sb, "      Fix: %s\n", issue.Fix)
			}
			sb.WriteString("\n")
		}
	}
	return sb.String()
}

// Issues returns all detected issues.
func (e *Engine) Issues() []Issue {
	e.mu.RLock()
	defer e.mu.RUnlock()
	out := make([]Issue, len(e.issues))
	copy(out, e.issues)
	return out
}

func flattenRuns(runs []*Run) []*Run {
	var out []*Run
	for _, r := range runs {
		out = append(out, r)
		out = append(out, flattenRuns(r.Children)...)
	}
	return out
}

func isLLMResponseEmpty(r *Run) bool {
	if r.Outputs == nil {
		return true
	}
	text, _ := r.Outputs["text"].(string)
	return text == ""
}

func isMissingToolResult(r *Run) bool {
	if r.Outputs == nil {
		return true
	}
	output, ok := r.Outputs["output"].(string)
	return !ok || output == ""
}

type engineToolCall struct {
	name  string
	input string
	runID string
}

func (e *Engine) detectRepeatedToolCalls(allRuns []*Run) {
	var calls []engineToolCall
	for _, r := range allRuns {
		if r.Type != RunTypeTool {
			continue
		}
		input := ""
		if r.Inputs != nil {
			input, _ = r.Inputs["input"].(string)
		}
		calls = append(calls, engineToolCall{name: r.Name, input: input, runID: r.ID})
	}

	byName := make(map[string][]engineToolCall)
	for _, c := range calls {
		byName[c.name] = append(byName[c.name], c)
	}

	for _, group := range byName {
		if len(group) < 3 {
			continue
		}
		similar := findSimilarGroup(group)
		if len(similar) >= 3 {
			runIDs := make([]string, len(similar))
			for i, c := range similar {
				runIDs[i] = c.runID
			}
			e.issues = append(e.issues, Issue{
				RunID:    similar[0].runID,
				Severity: "info",
				Title:    "Repeated tool calls",
				Detail:   fmt.Sprintf("Tool %q was called %d times with similar input. Run IDs: %s", group[0].name, len(similar), strings.Join(runIDs, ", ")),
				Fix:      "Consider caching tool results or deduplicating calls.",
			})
		}
	}
}

func findSimilarGroup(calls []engineToolCall) []engineToolCall {
	trimmed := make([]string, len(calls))
	for i, c := range calls {
		trimmed[i] = strings.TrimSpace(strings.ToLower(c.input))
	}
	counts := make(map[string]int)
	for _, t := range trimmed {
		counts[t]++
	}
	var bestInput string
	maxCount := 0
	for input, count := range counts {
		if count > maxCount {
			maxCount = count
			bestInput = input
		}
	}
	if maxCount < 3 {
		return nil
	}
	var result []engineToolCall
	for i, c := range calls {
		if trimmed[i] == bestInput {
			result = append(result, c)
		}
	}
	return result
}
