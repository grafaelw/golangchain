// Package serve exposes any chain.Runnable, agent.AgentExecutor, or
// graph.CompiledGraph over HTTP with a small, LangServe-like contract.
//
// # Endpoints
//
// For every registered base path, two endpoints are added:
//
//	POST <base>/invoke   body: {"input": <any>}  → {"output": <any>}
//	POST <base>/stream   body: {"input": <any>}  → text/event-stream (SSE)
//
// A default endpoint is always available:
//
//	GET  /health                                  → {"status": "ok"}
//
// # Registration
//
//   - [Mux.HandleRunnable] — for any chain.Runnable.
//   - [Mux.HandleAgent]    — for an *agent.AgentExecutor; streams AgentEvents
//     over SSE with typed payloads (thought / tool_call / tool_result / …).
//   - [HandleGraph]        — generic helper for graph.CompiledGraph[S];
//     honours the "X-Thread-ID" request header for checkpointing.
//
// # Quick start
//
//	mux := serve.NewMux()
//	mux.HandleRunnable("/qa",    ragChain)
//	mux.HandleAgent   ("/agent", executor)
//	serve.HandleGraph (mux, "/plan", compiled, func() State { return State{} })
//	http.ListenAndServe(":8080", mux)
//
// # Middleware
//
// [WithContextValue] is provided as a minimal example; wrap the Mux with any
// standard http.Handler middleware for auth, logging, or tracing.
package serve
