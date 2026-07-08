// Package eval — see doc.go for the package overview.
package eval

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"reflect"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

// Example is a single dataset row.
type Example struct {
	ID       string         `json:"id,omitempty"`
	Input    any            `json:"input"`
	Expected any            `json:"expected,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// Dataset is an ordered list of Examples.
type Dataset []Example

// LoadJSONL reads a JSON Lines file into a Dataset.
func LoadJSONL(path string) (Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("eval: open %q: %w", path, err)
	}
	defer func() { _ = f.Close() }()

	dec := json.NewDecoder(f)
	var out Dataset
	for {
		var ex Example
		if err := dec.Decode(&ex); err != nil {
			if errors.Is(err, io.EOF) {
				return out, nil
			}
			return out, fmt.Errorf("eval: decode: %w", err)
		}
		out = append(out, ex)
	}
}

// SaveJSONL writes a Dataset as JSON Lines to path.
func (d Dataset) SaveJSONL(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("eval: create %q: %w", path, err)
	}
	defer func() { _ = f.Close() }()
	enc := json.NewEncoder(f)
	for _, ex := range d {
		if err := enc.Encode(ex); err != nil {
			return err
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

// PredictFunc turns an example input into a prediction (any type).
type PredictFunc func(ctx context.Context, input any) (any, error)

// ---------------------------------------------------------------------------
// Score & Evaluator
// ---------------------------------------------------------------------------

// Score is the result of a single evaluator on a single example.
type Score struct {
	Name    string  `json:"name"`
	Value   float64 `json:"value"` // 1.0 = pass, 0.0 = fail (for boolean evaluators)
	Comment string  `json:"comment,omitempty"`
}

// Evaluator scores a single (example, prediction) pair.
type Evaluator interface {
	Name() string
	Evaluate(ctx context.Context, example Example, prediction any) (Score, error)
}

// ---------------------------------------------------------------------------
// Built-in evaluators
// ---------------------------------------------------------------------------

// ExactMatch compares the string form of prediction to Example.Expected.
type ExactMatch struct{ CaseInsensitive bool }

func (e ExactMatch) Name() string { return "exact_match" }
func (e ExactMatch) Evaluate(_ context.Context, ex Example, pred any) (Score, error) {
	got, want := fmt.Sprint(pred), fmt.Sprint(ex.Expected)
	if e.CaseInsensitive {
		got, want = strings.ToLower(got), strings.ToLower(want)
	}
	if strings.TrimSpace(got) == strings.TrimSpace(want) {
		return Score{Name: e.Name(), Value: 1}, nil
	}
	return Score{Name: e.Name(), Value: 0, Comment: fmt.Sprintf("expected %q, got %q", want, got)}, nil
}

// Contains checks that prediction contains Example.Expected as a substring.
type Contains struct{ CaseInsensitive bool }

func (c Contains) Name() string { return "contains" }
func (c Contains) Evaluate(_ context.Context, ex Example, pred any) (Score, error) {
	got, want := fmt.Sprint(pred), fmt.Sprint(ex.Expected)
	if c.CaseInsensitive {
		got, want = strings.ToLower(got), strings.ToLower(want)
	}
	if strings.Contains(got, want) {
		return Score{Name: c.Name(), Value: 1}, nil
	}
	return Score{Name: c.Name(), Value: 0, Comment: fmt.Sprintf("substring %q not found", want)}, nil
}

// Regex checks that prediction matches Pattern.
type Regex struct {
	Pattern string
	re      *regexp.Regexp
}

// NewRegex compiles pattern; returns an error if it is invalid.
func NewRegex(pattern string) (*Regex, error) {
	re, err := regexp.Compile(pattern)
	if err != nil {
		return nil, fmt.Errorf("eval: regex %q: %w", pattern, err)
	}
	return &Regex{Pattern: pattern, re: re}, nil
}

func (r *Regex) Name() string { return "regex" }
func (r *Regex) Evaluate(_ context.Context, _ Example, pred any) (Score, error) {
	if r.re.MatchString(fmt.Sprint(pred)) {
		return Score{Name: r.Name(), Value: 1}, nil
	}
	return Score{Name: r.Name(), Value: 0, Comment: "no regex match"}, nil
}

// JSONEqual JSON-marshals both sides and compares them structurally.
type JSONEqual struct{}

func (JSONEqual) Name() string { return "json_equal" }
func (JSONEqual) Evaluate(_ context.Context, ex Example, pred any) (Score, error) {
	pa, err := roundTripJSON(pred)
	if err != nil {
		return Score{Name: "json_equal", Value: 0, Comment: fmt.Sprintf("marshal pred: %v", err)}, nil
	}
	pb, err := roundTripJSON(ex.Expected)
	if err != nil {
		return Score{Name: "json_equal", Value: 0, Comment: fmt.Sprintf("marshal expected: %v", err)}, nil
	}
	if reflect.DeepEqual(pa, pb) {
		return Score{Name: "json_equal", Value: 1}, nil
	}
	return Score{Name: "json_equal", Value: 0, Comment: "JSON differs"}, nil
}

func roundTripJSON(v any) (any, error) {
	data, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	var out any
	return out, json.Unmarshal(data, &out)
}

// LLMAsJudge asks a model to rate the prediction against the reference.
// The prompt must instruct the judge to output a JSON object with fields
// {"score": <0..1>, "comment": "..."}.
type LLMAsJudge struct {
	LLM    llm.LLM
	Prompt string
}

// DefaultJudgePrompt is a minimalist grading prompt.
const DefaultJudgePrompt = `You are grading a model's answer.

Question or input:
{{ .input }}

Reference (may be empty):
{{ .expected }}

Candidate answer:
{{ .prediction }}

Return ONLY a JSON object with fields "score" (0..1) and "comment" (string, short).`

// NewLLMAsJudge wraps model as an evaluator.
func NewLLMAsJudge(model llm.LLM) *LLMAsJudge {
	return &LLMAsJudge{LLM: model, Prompt: DefaultJudgePrompt}
}

func (j *LLMAsJudge) Name() string { return "llm_as_judge" }

func (j *LLMAsJudge) Evaluate(ctx context.Context, ex Example, pred any) (Score, error) {
	p := strings.ReplaceAll(j.Prompt, "{{ .input }}", fmt.Sprint(ex.Input))
	p = strings.ReplaceAll(p, "{{ .expected }}", fmt.Sprint(ex.Expected))
	p = strings.ReplaceAll(p, "{{ .prediction }}", fmt.Sprint(pred))
	gen, err := j.LLM.Generate(ctx, []schema.Message{schema.NewHumanMessage(p)})
	if err != nil {
		return Score{Name: j.Name()}, fmt.Errorf("eval: judge llm: %w", err)
	}
	var parsed struct {
		Score   float64 `json:"score"`
		Comment string  `json:"comment"`
	}
	// Try to isolate a JSON object if the model added prose around it.
	text := strings.TrimSpace(gen.Text)
	if i, k := strings.Index(text, "{"), strings.LastIndex(text, "}"); i >= 0 && k > i {
		text = text[i : k+1]
	}
	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		return Score{Name: j.Name(), Value: 0, Comment: "unparseable judge output: " + gen.Text}, nil
	}
	return Score{Name: j.Name(), Value: parsed.Score, Comment: parsed.Comment}, nil
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

// Result is the full record for one example: its prediction, wall time, error,
// and each evaluator's score.
type Result struct {
	Example    Example       `json:"example"`
	Prediction any           `json:"prediction,omitempty"`
	Latency    time.Duration `json:"latency"`
	Err        string        `json:"error,omitempty"`
	Scores     []Score       `json:"scores"`
}

// Report aggregates results across a run.
type Report struct {
	Results []Result       `json:"results"`
	Summary map[string]any `json:"summary"`
}

// Aggregate returns per-evaluator mean scores plus overall counts.
func (r Report) Aggregate() map[string]any {
	means := map[string]float64{}
	counts := map[string]int{}
	failed := 0
	for _, res := range r.Results {
		if res.Err != "" {
			failed++
		}
		for _, s := range res.Scores {
			means[s.Name] += s.Value
			counts[s.Name]++
		}
	}
	for k, n := range counts {
		if n > 0 {
			means[k] /= float64(n)
		}
	}
	out := map[string]any{
		"n":           len(r.Results),
		"errors":      failed,
		"mean_scores": means,
	}
	return out
}

// Run executes predict over dataset, applies each evaluator, and returns a Report.
// Set concurrency to 0 or 1 for sequential execution.
func Run(ctx context.Context, dataset Dataset, predict PredictFunc, evaluators []Evaluator, concurrency int) (Report, error) {
	if concurrency < 1 {
		concurrency = 1
	}
	results := make([]Result, len(dataset))
	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup

	for i, ex := range dataset {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, ex Example) {
			defer wg.Done()
			defer func() { <-sem }()

			start := time.Now()
			pred, err := predict(ctx, ex.Input)
			res := Result{Example: ex, Prediction: pred, Latency: time.Since(start)}
			if err != nil {
				res.Err = err.Error()
				results[i] = res
				return
			}
			for _, e := range evaluators {
				s, err := e.Evaluate(ctx, ex, pred)
				if err != nil {
					s = Score{Name: e.Name(), Value: 0, Comment: "evaluator error: " + err.Error()}
				}
				res.Scores = append(res.Scores, s)
			}
			results[i] = res
		}(i, ex)
	}
	wg.Wait()

	report := Report{Results: results}
	report.Summary = report.Aggregate()
	return report, nil
}
