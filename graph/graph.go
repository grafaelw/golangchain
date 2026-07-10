// Package graph implements a LangGraph-equivalent StateGraph engine for Go.
//
// A StateGraph is a directed graph where:
//   - Nodes are functions that receive the current state and return an updated state.
//   - Edges are unconditional or conditional transitions between nodes.
//   - State is a user-defined generic type S, merged via a user-supplied StateReducer.
//   - Cycles are fully supported (enabling agent loops).
//   - Parallel branches are supported via AddParallelEdges.
//   - State can be checkpointed at each node for resumable runs and human-in-the-loop.
//
// Typical usage:
//
//	type State struct {
//	    Messages []schema.Message
//	    Next     string
//	}
//
//	g := graph.NewStateGraph(func(cur, upd State) State {
//	    cur.Messages = append(cur.Messages, upd.Messages...)
//	    cur.Next = upd.Next
//	    return cur
//	})
//
//	g.AddNode("agent", agentNode)
//	g.AddNode("tools", toolsNode)
//	g.AddEdge(graph.START, "agent")
//	g.AddConditionalEdges("agent", routerFn, map[string]string{
//	    "use_tools": "tools",
//	    "done":      graph.END,
//	})
//	g.AddEdge("tools", "agent") // loop back
//
//	compiled, _ := g.Compile(graph.WithCheckpointer(graph.NewMemoryCheckpointer()))
//	result, _ := compiled.Invoke(ctx, State{...}, graph.WithThreadID("thread-1"))
package graph

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/grafaelw/golangchain/callbacks"
)

// ErrGraphMaxSteps is returned when a compiled graph exceeds the maximum
// number of node executions during a single invocation.
var ErrGraphMaxSteps = errors.New("graph exceeded max steps")

// ---------------------------------------------------------------------------
// Sentinel node names
// ---------------------------------------------------------------------------

const (
	// START is the virtual entry node. Add edges FROM Start to your first real node.
	START = "__start__"
	// END is the virtual terminal node. Route to END to finish graph execution.
	END = "__end__"
)

// ---------------------------------------------------------------------------
// Core function types
// ---------------------------------------------------------------------------

// NodeFunc is a graph node: it receives the current state and returns a
// (partial) state update. The update is merged into the current state via
// the StateReducer.
type NodeFunc[S any] func(ctx context.Context, state S) (S, error)

// ConditionFunc maps the current state to a routing key. The key is looked up
// in the edge mapping supplied to AddConditionalEdges.
type ConditionFunc[S any] func(ctx context.Context, state S) string

// StateReducer merges a partial state update into the current full state.
// For simple struct states a common reducer just overwrites fields:
//
//	func(cur, upd MyState) MyState {
//	    if upd.Messages != nil { cur.Messages = append(cur.Messages, upd.Messages...) }
//	    if upd.Next != "" { cur.Next = upd.Next }
//	    return cur
//	}
type StateReducer[S any] func(current S, update S) S

// ---------------------------------------------------------------------------
// Edge types (internal)
// ---------------------------------------------------------------------------

type edgeKind int

const (
	edgeUnconditional edgeKind = iota
	edgeConditional
	edgeParallel
)

type edge[S any] struct {
	kind      edgeKind
	to        string            // for unconditional
	condition ConditionFunc[S]  // for conditional
	mapping   map[string]string // for conditional: route key → node name
	branches  []string          // for parallel
	interrupt bool              // pause before traversing this edge
}

// ---------------------------------------------------------------------------
// StateGraph — graph definition (mutable, not thread-safe during build)
// ---------------------------------------------------------------------------

// StateGraph is the mutable builder for a typed state graph.
// Call Compile() to produce an immutable, executable CompiledGraph.
type StateGraph[S any] struct {
	nodes   map[string]NodeFunc[S]
	edges   map[string][]edge[S] // from → edges
	reducer StateReducer[S]
	name    string
}

// NewStateGraph constructs an empty StateGraph with the given reducer.
func NewStateGraph[S any](reducer StateReducer[S]) *StateGraph[S] {
	return &StateGraph[S]{
		nodes:   make(map[string]NodeFunc[S]),
		edges:   make(map[string][]edge[S]),
		reducer: reducer,
		name:    "StateGraph",
	}
}

// WithName sets a human-readable name used in callbacks and checkpoints.
func (g *StateGraph[S]) WithName(name string) *StateGraph[S] {
	g.name = name
	return g
}

// AddNode registers a node function under the given name.
// Node names must be unique and must not be START or END.
func (g *StateGraph[S]) AddNode(name string, fn NodeFunc[S]) error {
	if name == START || name == END {
		return fmt.Errorf("graph: %q is a reserved node name", name)
	}
	if _, exists := g.nodes[name]; exists {
		return fmt.Errorf("graph: node %q already registered", name)
	}
	g.nodes[name] = fn
	return nil
}

// MustAddNode is like AddNode but panics on error.
func (g *StateGraph[S]) MustAddNode(name string, fn NodeFunc[S]) {
	if err := g.AddNode(name, fn); err != nil {
		panic(err)
	}
}

// AddEdge adds an unconditional edge from → to.
// Use START as from to define the graph's entry point(s).
// Use END as to to terminate a branch.
func (g *StateGraph[S]) AddEdge(from, to string) error {
	if from == END {
		return fmt.Errorf("graph: cannot add edge FROM %q", END)
	}
	g.edges[from] = append(g.edges[from], edge[S]{kind: edgeUnconditional, to: to})
	return nil
}

// MustAddEdge is like AddEdge but panics on error.
func (g *StateGraph[S]) MustAddEdge(from, to string) {
	if err := g.AddEdge(from, to); err != nil {
		panic(err)
	}
}

// AddConditionalEdges adds a conditional edge from node `from`.
// At runtime, condition(state) is called and its return value is looked up
// in mapping to determine the next node. A special key "*" in mapping acts
// as a default/catch-all.
func (g *StateGraph[S]) AddConditionalEdges(from string, condition ConditionFunc[S], mapping map[string]string) error {
	if from == END {
		return fmt.Errorf("graph: cannot add edge FROM %q", END)
	}
	if len(mapping) == 0 {
		return fmt.Errorf("graph: conditional edge from %q has empty mapping", from)
	}
	g.edges[from] = append(g.edges[from], edge[S]{
		kind:      edgeConditional,
		condition: condition,
		mapping:   mapping,
	})
	return nil
}

// MustAddConditionalEdges is like AddConditionalEdges but panics on error.
func (g *StateGraph[S]) MustAddConditionalEdges(from string, condition ConditionFunc[S], mapping map[string]string) {
	if err := g.AddConditionalEdges(from, condition, mapping); err != nil {
		panic(err)
	}
}

// AddParallelEdges fans out from `from` to all `branches` simultaneously.
// Each branch node runs concurrently; the results are merged via the reducer
// in an unspecified order (make your reducer commutative for parallel branches).
func (g *StateGraph[S]) AddParallelEdges(from string, branches []string) error {
	if from == END {
		return fmt.Errorf("graph: cannot add edge FROM %q", END)
	}
	if len(branches) < 2 {
		return fmt.Errorf("graph: parallel edges from %q need at least 2 branches", from)
	}
	g.edges[from] = append(g.edges[from], edge[S]{kind: edgeParallel, branches: branches})
	return nil
}

// AddSubgraph adds a CompiledGraph as a named node.
func (g *StateGraph[S]) AddSubgraph(name string, subgraph *CompiledGraph[S]) error {
	return g.AddNode(name, SubgraphNode(subgraph))
}

// MustAddSubgraph is like AddSubgraph but panics on error.
func (g *StateGraph[S]) MustAddSubgraph(name string, subgraph *CompiledGraph[S]) {
	if err := g.AddSubgraph(name, subgraph); err != nil {
		panic(err)
	}
}

// AddInterruptEdge marks an unconditional edge as an interrupt point.
// Execution pauses before traversing this edge, saving a checkpoint.
// The next invocation (with the same thread ID) resumes from this point.
//
//	g.AddInterruptEdge("agent", "approval_required") // pause before approval
func (g *StateGraph[S]) AddInterruptEdge(from, to string) {
	e := edge[S]{kind: edgeUnconditional, to: to, interrupt: true}
	g.edges[from] = append(g.edges[from], e)
}

// ---------------------------------------------------------------------------
// Compile options
// ---------------------------------------------------------------------------

// CompileOption configures CompiledGraph behaviour.
type CompileOption[S any] func(*compiledConfig[S])

type compiledConfig[S any] struct {
	checkpointer Checkpointer[S]
	callbacks    *callbacks.CallbackManager
	maxSteps     int
}

// WithCheckpointer attaches a Checkpointer for state persistence and
// human-in-the-loop support.
func WithCheckpointer[S any](cp Checkpointer[S]) CompileOption[S] {
	return func(c *compiledConfig[S]) { c.checkpointer = cp }
}

// WithGraphCallbacks attaches a CallbackManager to the graph.
func WithGraphCallbacks[S any](cm *callbacks.CallbackManager) CompileOption[S] {
	return func(c *compiledConfig[S]) { c.callbacks = cm }
}

// WithMaxSteps sets the maximum number of node executions before the graph
// aborts with an error (prevents infinite loops in buggy graphs; default 100).
func WithMaxSteps[S any](n int) CompileOption[S] {
	return func(c *compiledConfig[S]) { c.maxSteps = n }
}

// ---------------------------------------------------------------------------
// Compile
// ---------------------------------------------------------------------------

// Compile validates the graph and returns an immutable CompiledGraph.
// Validation checks:
//   - At least one edge from START exists.
//   - Every non-END destination node is registered.
//   - No edges exist from END.
func (g *StateGraph[S]) Compile(opts ...CompileOption[S]) (*CompiledGraph[S], error) {
	cfg := &compiledConfig[S]{maxSteps: 100}
	for _, o := range opts {
		o(cfg)
	}

	// Validate: must have entry edge from START
	if _, ok := g.edges[START]; !ok {
		return nil, fmt.Errorf("graph: no entry edge from %q — call AddEdge(%q, firstNode)", START, START)
	}

	// Validate: all destinations are registered nodes or END
	for from, edges := range g.edges {
		for _, e := range edges {
			switch e.kind {
			case edgeUnconditional:
				if e.to != END {
					if _, ok := g.nodes[e.to]; !ok {
						return nil, fmt.Errorf("graph: edge from %q to unknown node %q", from, e.to)
					}
				}
			case edgeConditional:
				for key, to := range e.mapping {
					if to != END {
						if _, ok := g.nodes[to]; !ok {
							return nil, fmt.Errorf("graph: conditional edge from %q, key %q: unknown node %q", from, key, to)
						}
					}
				}
			case edgeParallel:
				for _, branch := range e.branches {
					if branch != END {
						if _, ok := g.nodes[branch]; !ok {
							return nil, fmt.Errorf("graph: parallel edge from %q: unknown branch node %q", from, branch)
						}
					}
				}
			}
		}
	}

	// Deep-copy nodes and edges for thread safety at runtime
	nodes := make(map[string]NodeFunc[S], len(g.nodes))
	for k, v := range g.nodes {
		nodes[k] = v
	}
	edgesCopy := make(map[string][]edge[S], len(g.edges))
	for k, v := range g.edges {
		cp := make([]edge[S], len(v))
		copy(cp, v)
		edgesCopy[k] = cp
	}

	return &CompiledGraph[S]{
		name:    g.name,
		nodes:   nodes,
		edges:   edgesCopy,
		reducer: g.reducer,
		cfg:     cfg,
	}, nil
}

// ---------------------------------------------------------------------------
// CompiledGraph — immutable, thread-safe, executable graph
// ---------------------------------------------------------------------------

// CompiledGraph is the compiled, executable form of a StateGraph.
// It is safe for concurrent use from multiple goroutines.
type CompiledGraph[S any] struct {
	name    string
	nodes   map[string]NodeFunc[S]
	edges   map[string][]edge[S]
	reducer StateReducer[S]
	cfg     *compiledConfig[S]
}

// ---------------------------------------------------------------------------
// Run options
// ---------------------------------------------------------------------------

// RunOption configures a single graph execution.
type RunOption[S any] func(*runConfig[S])

type runConfig[S any] struct {
	threadID                 string
	resumeAfterEdgeInterrupt bool // true when resuming from an edge interrupt
}

// WithThreadID associates a thread ID with this run for checkpointing.
// Runs with the same thread ID share checkpointed state.
func WithThreadID[S any](id string) RunOption[S] {
	return func(c *runConfig[S]) { c.threadID = id }
}

// ---------------------------------------------------------------------------
// GraphEvent — events emitted during streaming execution
// ---------------------------------------------------------------------------

// GraphEventType identifies the kind of graph event.
type GraphEventType string

const (
	GraphEventNodeStart  GraphEventType = "node_start"
	GraphEventNodeEnd    GraphEventType = "node_end"
	GraphEventCheckpoint GraphEventType = "checkpoint"
	GraphEventEnd        GraphEventType = "end"
	GraphEventError      GraphEventType = "error"
)

// GraphEvent is a single event emitted during a streaming graph run.
type GraphEvent[S any] struct {
	Type  GraphEventType
	Node  string
	State S     // state after node execution (for NodeEnd, Checkpoint, End)
	Err   error // for Error events
}

// ---------------------------------------------------------------------------
// Invoke — blocking execution
// ---------------------------------------------------------------------------

// Invoke runs the graph to completion and returns the final state.
func (c *CompiledGraph[S]) Invoke(ctx context.Context, input S, opts ...RunOption[S]) (S, error) {
	rc := &runConfig[S]{}
	for _, o := range opts {
		o(rc)
	}

	var finalState S
	for event := range c.stream(ctx, input, rc) {
		if event.Err != nil {
			return finalState, event.Err
		}
		if event.Type == GraphEventEnd {
			finalState = event.State
		}
	}
	return finalState, nil
}

// Stream runs the graph and yields GraphEvents in real time.
func (c *CompiledGraph[S]) Stream(ctx context.Context, input S, opts ...RunOption[S]) <-chan GraphEvent[S] {
	rc := &runConfig[S]{}
	for _, o := range opts {
		o(rc)
	}
	return c.stream(ctx, input, rc)
}

// ---------------------------------------------------------------------------
// Internal execution engine
// ---------------------------------------------------------------------------

// Interrupt is a sentinel error returned by a node to pause execution.
// The AgentExecutor saves state and the run can be resumed later.
type Interrupt struct {
	Message string
}

func (i *Interrupt) Error() string {
	return fmt.Sprintf("graph: interrupted: %s", i.Message)
}

// NewInterrupt creates an Interrupt error that pauses graph execution.
// Nodes return this to implement human-in-the-loop checkpoints.
func NewInterrupt(message string) *Interrupt {
	return &Interrupt{Message: message}
}

func (c *CompiledGraph[S]) stream(ctx context.Context, input S, rc *runConfig[S]) <-chan GraphEvent[S] {
	ch := make(chan GraphEvent[S], 32)

	go func() {
		defer close(ch)

		state := input

		// Queue of node names to execute next
		queue := []string{START}

		// Load checkpoint if threadID provided and checkpointer available
		if rc.threadID != "" && c.cfg.checkpointer != nil {
			cp, err := c.cfg.checkpointer.Load(ctx, rc.threadID)
			if err == nil && cp != nil {
				state = cp.State
				if cp.InterruptedNode != "" {
					queue = []string{cp.InterruptedNode}
					rc.resumeAfterEdgeInterrupt = true
				}
			}
		}

		steps := 0

		for len(queue) > 0 {
			if ctx.Err() != nil {
				ch <- GraphEvent[S]{Type: GraphEventError, Err: ctx.Err()}
				return
			}
			if steps >= c.cfg.maxSteps {
				ch <- GraphEvent[S]{
					Type: GraphEventError,
					Err:  fmt.Errorf("%w (%d)", ErrGraphMaxSteps, c.cfg.maxSteps),
				}
				return
			}

			nodeName := queue[0]
			queue = queue[1:]

			// START is virtual — don't execute, just follow edges
			if nodeName == START {
				nextNodes, err := c.resolveEdges(ctx, nodeName, state)
				if err != nil {
					ch <- GraphEvent[S]{Type: GraphEventError, Err: err}
					return
				}
				queue = append(queue, nextNodes...)
				continue
			}

			// END reached
			if nodeName == END {
				c.saveCheckpoint(ctx, rc, state, ch)
				ch <- GraphEvent[S]{Type: GraphEventEnd, Node: END, State: state}
				return
			}

			// Execute node
			fn, ok := c.nodes[nodeName]
			if !ok {
				ch <- GraphEvent[S]{Type: GraphEventError, Err: fmt.Errorf("graph: unknown node %q", nodeName)}
				return
			}

			// Inject a node-level run ID. Passing nodeCtx to fn() lets any
			// callbacks fired inside the node (e.g. LLM calls) reference this
			// node run as their parent.
			nodeCtx := callbacks.WithRunID(ctx, callbacks.NewRunID())
			if c.cfg.callbacks != nil {
				c.cfg.callbacks.OnGraphNodeStart(nodeCtx, c.name, nodeName)
			}
			ch <- GraphEvent[S]{Type: GraphEventNodeStart, Node: nodeName, State: state}

			update, err := fn(nodeCtx, state)
			steps++

			// Human-in-the-loop interrupt
			if interrupt, ok := err.(*Interrupt); ok {
				// Save state so the run can be resumed
				state = c.reducer(state, update)
				c.saveCheckpoint(ctx, rc, state, ch)
				ch <- GraphEvent[S]{Type: GraphEventError, Err: interrupt}
				return
			}

			if err != nil {
				if c.cfg.callbacks != nil {
					c.cfg.callbacks.OnError(nodeCtx, nodeName, err)
				}
				ch <- GraphEvent[S]{Type: GraphEventError, Err: fmt.Errorf("graph: node %q: %w", nodeName, err)}
				return
			}

			// Merge update into state
			state = c.reducer(state, update)

			if c.cfg.callbacks != nil {
				c.cfg.callbacks.OnGraphNodeEnd(nodeCtx, c.name, nodeName)
			}
			ch <- GraphEvent[S]{Type: GraphEventNodeEnd, Node: nodeName, State: state}

			// Checkpoint after each node if configured
			c.saveCheckpoint(ctx, rc, state, ch)

			// Resolve next nodes
			nextNodes, err := c.resolveEdges(ctx, nodeName, state)
			if err != nil {
				ch <- GraphEvent[S]{Type: GraphEventError, Err: err}
				return
			}

			// Edge-level interrupt: pause before following an interrupt edge.
			// Skip if we just resumed from this node.
			if !rc.resumeAfterEdgeInterrupt && c.hasInterruptEdge(nodeName) {
				c.saveInterruptedCheckpoint(ctx, rc, state, nodeName, ch)
				ch <- GraphEvent[S]{Type: GraphEventError, Err: NewInterrupt("graph: paused at edge from " + nodeName)}
				return
			}

			// Handle parallel branches with goroutines
			if len(nextNodes) > 1 {
				mergedState, err := c.runParallel(ctx, nextNodes, state, rc, steps)
				if err != nil {
					ch <- GraphEvent[S]{Type: GraphEventError, Err: err}
					return
				}
				state = mergedState
				// After parallel branches complete, signal end if all branches finished
				ch <- GraphEvent[S]{Type: GraphEventEnd, Node: END, State: state}
				return
			}

			queue = append(queue, nextNodes...)
		}

		// Queue exhausted without hitting END — emit end with current state
		ch <- GraphEvent[S]{Type: GraphEventEnd, State: state}
	}()

	return ch
}

// hasInterruptEdge returns true if any outgoing edge from the given node
// is marked as an interrupt point.
func (c *CompiledGraph[S]) hasInterruptEdge(from string) bool {
	for _, e := range c.edges[from] {
		if e.interrupt {
			return true
		}
	}
	return false
}

// resolveEdges computes the list of next node names from the current node.
func (c *CompiledGraph[S]) resolveEdges(ctx context.Context, from string, state S) ([]string, error) {
	edges, ok := c.edges[from]
	if !ok {
		return nil, fmt.Errorf("graph: node %q has no outgoing edges (add an edge to %q or %q)", from, END, "another node")
	}

	var next []string
	for _, e := range edges {
		switch e.kind {
		case edgeUnconditional:
			next = append(next, e.to)
		case edgeConditional:
			key := e.condition(ctx, state)
			to, ok := e.mapping[key]
			if !ok {
				// Try wildcard fallback
				to, ok = e.mapping["*"]
				if !ok {
					return nil, fmt.Errorf("graph: conditional edge from %q returned unknown key %q (add a '*' fallback or handle it)", from, key)
				}
			}
			next = append(next, to)
		case edgeParallel:
			next = append(next, e.branches...)
		}
	}
	return next, nil
}

// runParallel executes a set of node names concurrently and merges the results.
func (c *CompiledGraph[S]) runParallel(ctx context.Context, nodeNames []string, state S, rc *runConfig[S], baseSteps int) (S, error) {
	type result struct {
		state S
		err   error
	}

	results := make(chan result, len(nodeNames))
	var wg sync.WaitGroup

	for _, name := range nodeNames {
		wg.Add(1)
		go func(nodeName string) {
			defer wg.Done()
			// Each branch runs to END independently
			merged := state
			subQueue := []string{nodeName}
			steps := baseSteps

			for len(subQueue) > 0 && steps < c.cfg.maxSteps {
				n := subQueue[0]
				subQueue = subQueue[1:]

				if n == END {
					break
				}

				fn, ok := c.nodes[n]
				if !ok {
					results <- result{err: fmt.Errorf("graph: parallel branch: unknown node %q", n)}
					return
				}

				// Inject a node run ID so nested callbacks have the right parent.
				nodeCtx := callbacks.WithRunID(ctx, callbacks.NewRunID())
				if c.cfg.callbacks != nil {
					c.cfg.callbacks.OnGraphNodeStart(nodeCtx, c.name, n)
				}

				update, err := fn(nodeCtx, merged)
				steps++
				if err != nil {
					if c.cfg.callbacks != nil {
						c.cfg.callbacks.OnError(nodeCtx, n, err)
					}
					results <- result{err: fmt.Errorf("graph: parallel branch %q: %w", n, err)}
					return
				}
				merged = c.reducer(merged, update)

				if c.cfg.callbacks != nil {
					c.cfg.callbacks.OnGraphNodeEnd(nodeCtx, c.name, n)
				}

				nextNodes, err := c.resolveEdges(ctx, n, merged)
				if err != nil {
					results <- result{err: err}
					return
				}
				subQueue = append(subQueue, nextNodes...)
			}
			results <- result{state: merged}
		}(name)
	}

	go func() { wg.Wait(); close(results) }()

	merged := state
	for r := range results {
		if r.err != nil {
			return merged, r.err
		}
		merged = c.reducer(merged, r.state)
	}
	return merged, nil
}

func (c *CompiledGraph[S]) saveCheckpoint(ctx context.Context, rc *runConfig[S], state S, ch chan<- GraphEvent[S]) {
	if rc.threadID == "" || c.cfg.checkpointer == nil {
		return
	}
	cp := Checkpoint[S]{
		ThreadID:  rc.threadID,
		State:     state,
		CreatedAt: time.Now(),
	}
	if err := c.cfg.checkpointer.Save(ctx, rc.threadID, cp); err == nil {
		if c.cfg.callbacks != nil {
			c.cfg.callbacks.OnGraphCheckpoint(ctx, c.name, rc.threadID)
		}
		ch <- GraphEvent[S]{Type: GraphEventCheckpoint, State: state}
	}
}

func (c *CompiledGraph[S]) saveInterruptedCheckpoint(ctx context.Context, rc *runConfig[S], state S, interruptedNode string, ch chan<- GraphEvent[S]) {
	if rc.threadID == "" || c.cfg.checkpointer == nil {
		return
	}
	cp := Checkpoint[S]{
		ThreadID:        rc.threadID,
		State:           state,
		CreatedAt:       time.Now(),
		InterruptedNode: interruptedNode,
	}
	if err := c.cfg.checkpointer.Save(ctx, rc.threadID, cp); err == nil {
		if c.cfg.callbacks != nil {
			c.cfg.callbacks.OnGraphCheckpoint(ctx, c.name, rc.threadID)
		}
		ch <- GraphEvent[S]{Type: GraphEventCheckpoint, State: state}
	}
}

// ---------------------------------------------------------------------------
// Checkpointer interface
// ---------------------------------------------------------------------------

// Checkpoint holds a serialisable snapshot of graph state at a point in time.
type Checkpoint[S any] struct {
	ThreadID        string    `json:"thread_id"`
	State           S         `json:"state"`
	CreatedAt       time.Time `json:"created_at"`
	StepCount       int       `json:"step_count,omitempty"`
	InterruptedNode string    `json:"interrupted_node,omitempty"`
}

// Checkpointer persists and retrieves graph checkpoints.
// Implement this interface to use Redis, Postgres, S3, etc.
type Checkpointer[S any] interface {
	// Save persists a checkpoint for the given thread.
	Save(ctx context.Context, threadID string, checkpoint Checkpoint[S]) error

	// Load retrieves the most recent checkpoint for the thread.
	// Returns (nil, nil) if no checkpoint exists.
	Load(ctx context.Context, threadID string) (*Checkpoint[S], error)

	// List returns all checkpoints for the thread, ordered oldest-first.
	List(ctx context.Context, threadID string) ([]Checkpoint[S], error)

	// Delete removes all checkpoints for the thread.
	Delete(ctx context.Context, threadID string) error
}

// ---------------------------------------------------------------------------
// MemoryCheckpointer — in-memory, concurrent-safe
// ---------------------------------------------------------------------------

// MemoryCheckpointer stores checkpoints in memory. All state is lost when
// the process exits. Use it for testing or short-lived sessions.
type MemoryCheckpointer[S any] struct {
	mu      sync.RWMutex
	history map[string][]Checkpoint[S] // threadID → ordered list
}

// NewMemoryCheckpointer creates a MemoryCheckpointer.
func NewMemoryCheckpointer[S any]() *MemoryCheckpointer[S] {
	return &MemoryCheckpointer[S]{history: make(map[string][]Checkpoint[S])}
}

func (m *MemoryCheckpointer[S]) Save(_ context.Context, threadID string, cp Checkpoint[S]) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.history[threadID] = append(m.history[threadID], cp)
	return nil
}

func (m *MemoryCheckpointer[S]) Load(_ context.Context, threadID string) (*Checkpoint[S], error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	hist := m.history[threadID]
	if len(hist) == 0 {
		return nil, nil
	}
	cp := hist[len(hist)-1]
	return &cp, nil
}

func (m *MemoryCheckpointer[S]) List(_ context.Context, threadID string) ([]Checkpoint[S], error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	hist := m.history[threadID]
	cp := make([]Checkpoint[S], len(hist))
	copy(cp, hist)
	return cp, nil
}

func (m *MemoryCheckpointer[S]) Delete(_ context.Context, threadID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.history, threadID)
	return nil
}

// ---------------------------------------------------------------------------
// JSONCheckpointer — persists checkpoints as JSON to an io.ReadWriter
// (useful for simple file-based persistence)
// ---------------------------------------------------------------------------

// SerialiseCheckpoint marshals a checkpoint to JSON bytes.
func SerialiseCheckpoint[S any](cp Checkpoint[S]) ([]byte, error) {
	return json.Marshal(cp)
}

// DeserialiseCheckpoint unmarshals a checkpoint from JSON bytes.
func DeserialiseCheckpoint[S any](data []byte) (Checkpoint[S], error) {
	var cp Checkpoint[S]
	return cp, json.Unmarshal(data, &cp)
}
