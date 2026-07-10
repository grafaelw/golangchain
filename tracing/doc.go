// Package tracing provides LangSmith-inspired observability:
// hierarchical run trees, live terminal output, JSON-lines export,
// human feedback collection, online evaluation, and OTel integration.
//
// # Core concepts
//
//   - Tracer — in-memory collector that builds a tree of Run nodes as
//     callbacks fire. Supports Summary(), TotalTokens(), and ExportJSON().
//   - PrettyHandler — colour-coded live terminal tracer (stderr by default).
//   - JSONLinesExporter — structured per-line JSON event export compatible
//     with jq, Loki, OTel file receivers, and LangSmith ingestion pipelines.
//   - FeedbackStore — user/evaluator Feedback keyed by run ID.
//   - AnnotationQueue — human-review queue for flagged runs.
//   - OnlineEvaluator — auto-scores runs as they complete via EvalFunc.
//   - Engine — post-hoc trace analysis detecting tool failures, high
//     latency, repeated calls, and empty responses.
//
// # Usage
//
//	tr := tracing.NewTracer()
//	cb := callbacks.NewCallbackManager(
//	    tracing.NewPrettyHandler(os.Stderr),
//	    tr.Handler(),
//	)
//	// Use cb in your agents, chains, and graphs.
//	fmt.Println(tr.Summary())
//	_ = tr.ExportJSON(os.Stdout)
package tracing
