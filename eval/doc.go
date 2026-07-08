// Package eval is a lightweight, fully local evaluation harness inspired by
// LangSmith's evaluators and datasets. No SaaS backend is required.
//
// # Concepts
//
//   - [Example]    — one row: Input + optional Expected + Metadata.
//   - [Dataset]    — an ordered slice of Examples, with JSONL load/save.
//   - [PredictFunc] — turns an Example.Input into a prediction (any type).
//   - [Evaluator]  — scores a single (Example, prediction) pair into a [Score].
//   - [Report]     — the aggregated result of a run, with per-evaluator means.
//
// # Built-in evaluators
//
//   - [ExactMatch]   — string equality (optionally case-insensitive).
//   - [Contains]     — substring containment.
//   - [Regex]        — regexp match against the prediction.
//   - [JSONEqual]    — JSON round-trip + reflect.DeepEqual.
//   - [LLMAsJudge]   — LLM-based grader returning JSON {"score","comment"}.
//
// # Quick start
//
//	dataset, _ := eval.LoadJSONL("qa.jsonl")
//	report, _  := eval.Run(ctx, dataset,
//	    func(ctx context.Context, in any) (any, error) {
//	        return chain.Invoke(ctx, in)
//	    },
//	    []eval.Evaluator{eval.Contains{CaseInsensitive: true}},
//	    /*concurrency*/ 4,
//	)
//	fmt.Printf("%+v\n", report.Summary)
package eval
