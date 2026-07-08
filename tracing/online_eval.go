package tracing

import (
	"context"
	"sync"

	"github.com/grafaelw/golangchain/callbacks"
	"github.com/grafaelw/golangchain/schema"
)

// Score represents an evaluation score.
type Score struct {
	Evaluator string  `json:"evaluator"`
	Name      string  `json:"name"`
	Value     float64 `json:"value"`
	Comment   string  `json:"comment,omitempty"`
}

// EvalFunc evaluates a run and returns scores.
type EvalFunc func(ctx context.Context, run *Run) ([]Score, error)

// OnlineEvaluator evaluates agent traces as they complete.
type OnlineEvaluator struct {
	evaluators []EvalFunc
	mu         sync.RWMutex
	scores     map[string][]Score
}

// NewOnlineEvaluator creates an OnlineEvaluator with optional evaluators.
func NewOnlineEvaluator(evaluators ...EvalFunc) *OnlineEvaluator {
	return &OnlineEvaluator{
		evaluators: evaluators,
		scores:     make(map[string][]Score),
	}
}

// Evaluate runs all evaluators against a completed run.
func (o *OnlineEvaluator) Evaluate(ctx context.Context, run *Run) []Score {
	o.mu.RLock()
	evals := make([]EvalFunc, len(o.evaluators))
	copy(evals, o.evaluators)
	o.mu.RUnlock()

	var all []Score
	for _, e := range evals {
		scores, err := e(ctx, run)
		if err != nil {
			continue
		}
		all = append(all, scores...)
	}

	o.mu.Lock()
	o.scores[run.ID] = all
	o.mu.Unlock()

	return all
}

// AddEvaluator adds a new evaluator.
func (o *OnlineEvaluator) AddEvaluator(e EvalFunc) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.evaluators = append(o.evaluators, e)
}

// Scores returns all scores for a run ID.
func (o *OnlineEvaluator) Scores(runID string) []Score {
	o.mu.RLock()
	defer o.mu.RUnlock()
	out := make([]Score, len(o.scores[runID]))
	copy(out, o.scores[runID])
	return out
}

// OnlineEvalHandler auto-evaluates runs as they complete.
type OnlineEvalHandler struct {
	callbacks.NoOpHandler
	tracer    *Tracer
	evaluator *OnlineEvaluator
}

// NewOnlineEvalHandler wraps a Tracer and OnlineEvaluator into a handler
// that auto-evaluates runs on completion.
func NewOnlineEvalHandler(tracer *Tracer, evaluator *OnlineEvaluator) *OnlineEvalHandler {
	return &OnlineEvalHandler{tracer: tracer, evaluator: evaluator}
}

func (h *OnlineEvalHandler) OnChainEnd(ctx context.Context, _ string, _ map[string]any) {
	run := h.findRun(ctx)
	if run == nil {
		return
	}
	h.evaluator.Evaluate(ctx, run)
}

func (h *OnlineEvalHandler) OnLLMEnd(ctx context.Context, _ string, _ *schema.Generation) {
	run := h.findRun(ctx)
	if run == nil {
		return
	}
	h.evaluator.Evaluate(ctx, run)
}

func (h *OnlineEvalHandler) OnToolEnd(ctx context.Context, _ string, _ string) {
	run := h.findRun(ctx)
	if run == nil {
		return
	}
	h.evaluator.Evaluate(ctx, run)
}

func (h *OnlineEvalHandler) findRun(ctx context.Context) *Run {
	runID := callbacks.RunIDFromContext(ctx)
	if runID == "" {
		return nil
	}
	return findRunByID(h.tracer.Runs(), runID)
}

func findRunByID(runs []*Run, id string) *Run {
	for _, r := range runs {
		if r.ID == id {
			return r
		}
		if found := findRunByID(r.Children, id); found != nil {
			return found
		}
	}
	return nil
}
