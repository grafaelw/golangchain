package eval

import (
	"context"
	"testing"
)

func TestExactMatchAndContains(t *testing.T) {
	ex := Example{Input: "hi", Expected: "hello world"}

	if s, _ := (ExactMatch{}).Evaluate(context.Background(), ex, "hello world"); s.Value != 1 {
		t.Fatalf("exact match: got %v", s)
	}
	if s, _ := (ExactMatch{}).Evaluate(context.Background(), ex, "wrong"); s.Value != 0 {
		t.Fatalf("exact match should fail")
	}
	if s, _ := (Contains{}).Evaluate(context.Background(), Example{Expected: "hello"}, "well, hello there"); s.Value != 1 {
		t.Fatalf("contains failed: %v", s)
	}
}

func TestJSONEqual(t *testing.T) {
	got, _ := (JSONEqual{}).Evaluate(context.Background(),
		Example{Expected: map[string]any{"a": 1, "b": []any{"x"}}},
		map[string]any{"b": []any{"x"}, "a": 1},
	)
	if got.Value != 1 {
		t.Fatalf("json equal: got %v", got)
	}
}

func TestRunAggregates(t *testing.T) {
	dataset := Dataset{
		{Input: "1", Expected: "a"},
		{Input: "2", Expected: "b"},
		{Input: "3", Expected: "c"},
	}
	predict := func(_ context.Context, in any) (any, error) {
		// Correct for first two, wrong for third
		switch in {
		case "1":
			return "a", nil
		case "2":
			return "b", nil
		default:
			return "z", nil
		}
	}
	report, err := Run(context.Background(), dataset, predict, []Evaluator{ExactMatch{}}, 2)
	if err != nil {
		t.Fatal(err)
	}
	means := report.Summary["mean_scores"].(map[string]float64)
	if got := means["exact_match"]; got < 0.66 || got > 0.67 {
		t.Fatalf("mean expected ~0.6667, got %v", got)
	}
}
