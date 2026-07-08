// Package serve — see doc.go for the package overview.
package serve

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/grafaelw/golangchain/agent"
	"github.com/grafaelw/golangchain/chain"
	"github.com/grafaelw/golangchain/graph"
)

// ---------------------------------------------------------------------------
// Mux
// ---------------------------------------------------------------------------

// Mux is a thin wrapper over http.ServeMux with helpers for registering
// golangchain endpoints. Use it directly as an http.Handler.
type Mux struct{ *http.ServeMux }

// NewMux constructs a Mux with a default /health endpoint.
func NewMux() *Mux {
	m := &Mux{ServeMux: http.NewServeMux()}
	m.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	return m
}

// ---------------------------------------------------------------------------
// Runnable handler
// ---------------------------------------------------------------------------

// HandleRunnable registers POST base/invoke and POST base/stream for r.
func (m *Mux) HandleRunnable(base string, r chain.Runnable) {
	m.HandleFunc(base+"/invoke", func(w http.ResponseWriter, req *http.Request) {
		invokeRunnable(w, req, r)
	})
	m.HandleFunc(base+"/stream", func(w http.ResponseWriter, req *http.Request) {
		streamRunnable(w, req, r)
	})
}

func invokeRunnable(w http.ResponseWriter, req *http.Request, r chain.Runnable) {
	input, err := decodeInput(req)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	out, err := r.Invoke(req.Context(), input)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"output": out})
}

func streamRunnable(w http.ResponseWriter, req *http.Request, r chain.Runnable) {
	input, err := decodeInput(req)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	ch, err := r.Stream(req.Context(), input)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	prepareSSE(w)
	flusher, _ := w.(http.Flusher)
	for chunk := range ch {
		if chunk.Err != nil {
			writeSSE(w, "error", map[string]string{"error": chunk.Err.Error()})
			return
		}
		event := "chunk"
		if chunk.Done {
			event = "done"
		}
		writeSSE(w, event, map[string]any{"value": chunk.Value})
		if flusher != nil {
			flusher.Flush()
		}
	}
}

// ---------------------------------------------------------------------------
// Agent handler
// ---------------------------------------------------------------------------

// HandleAgent registers POST base/invoke and POST base/stream for an AgentExecutor.
// Input body: {"input": "<question>"}.
func (m *Mux) HandleAgent(base string, ex *agent.AgentExecutor) {
	m.HandleFunc(base+"/invoke", func(w http.ResponseWriter, req *http.Request) {
		var body struct {
			Input string `json:"input"`
		}
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		out, err := ex.Run(req.Context(), body.Input)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"output": out})
	})
	m.HandleFunc(base+"/stream", func(w http.ResponseWriter, req *http.Request) {
		var body struct {
			Input string `json:"input"`
		}
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		prepareSSE(w)
		flusher, _ := w.(http.Flusher)
		for ev := range ex.Stream(req.Context(), body.Input) {
			payload := map[string]any{"type": ev.Type}
			switch ev.Type {
			case agent.EventThought:
				payload["thought"] = ev.Thought
			case agent.EventToolCall:
				payload["action"] = ev.Action
			case agent.EventToolResult:
				payload["observation"] = ev.Observation
			case agent.EventFinalAnswer:
				payload["answer"] = ev.Answer
			case agent.EventError:
				if ev.Err != nil {
					payload["error"] = ev.Err.Error()
				}
			}
			writeSSE(w, string(ev.Type), payload)
			if flusher != nil {
				flusher.Flush()
			}
		}
	})
}

// ---------------------------------------------------------------------------
// Graph handler
// ---------------------------------------------------------------------------

// HandleGraph registers POST base/invoke and POST base/stream for a
// CompiledGraph. newState is a factory that returns a zero-value state of
// the graph's generic type S so the request body can be JSON-decoded into it.
func HandleGraph[S any](m *Mux, base string, g *graph.CompiledGraph[S], newState func() S) {
	m.HandleFunc(base+"/invoke", func(w http.ResponseWriter, req *http.Request) {
		state := newState()
		if err := decodeState(req, &state); err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		out, err := g.Invoke(req.Context(), state, graphOpts[S](req)...)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"output": out})
	})
	m.HandleFunc(base+"/stream", func(w http.ResponseWriter, req *http.Request) {
		state := newState()
		if err := decodeState(req, &state); err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		prepareSSE(w)
		flusher, _ := w.(http.Flusher)
		for ev := range g.Stream(req.Context(), state, graphOpts[S](req)...) {
			payload := map[string]any{
				"type":  ev.Type,
				"node":  ev.Node,
				"state": ev.State,
			}
			if ev.Err != nil {
				payload["error"] = ev.Err.Error()
			}
			writeSSE(w, string(ev.Type), payload)
			if flusher != nil {
				flusher.Flush()
			}
		}
	})
}

func graphOpts[S any](req *http.Request) []graph.RunOption[S] {
	var opts []graph.RunOption[S]
	if id := req.Header.Get("X-Thread-ID"); id != "" {
		opts = append(opts, graph.WithThreadID[S](id))
	}
	return opts
}

// ---------------------------------------------------------------------------
// Middleware — inject a per-request context value (extend as needed).
// ---------------------------------------------------------------------------

// WithContextValue returns middleware that stores key=value in each request's
// context before delegating to next.
func WithContextValue(key, value any, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx := context.WithValue(req.Context(), key, value)
		next.ServeHTTP(w, req.WithContext(ctx))
	})
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

func decodeInput(req *http.Request) (any, error) {
	var body struct {
		Input any `json:"input"`
	}
	if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("decode body: %w", err)
	}
	return body.Input, nil
}

func decodeState(req *http.Request, state any) error {
	var body struct {
		Input json.RawMessage `json:"input"`
	}
	if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
		return fmt.Errorf("decode body: %w", err)
	}
	if len(body.Input) == 0 {
		return nil
	}
	return json.Unmarshal(body.Input, state)
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, code int, err error) {
	writeJSON(w, code, map[string]string{"error": err.Error()})
}

func prepareSSE(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
}

func writeSSE(w http.ResponseWriter, event string, data any) {
	payload, _ := json.Marshal(data)
	_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, payload)
}
