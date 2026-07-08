package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/grafaelw/golangchain/tools"
)

// A2AMessage represents an A2A protocol message.
type A2AMessage struct {
	ID        string    `json:"id"`
	From      string    `json:"from"`
	To        string    `json:"to"`
	Type      string    `json:"type"`
	Content   string    `json:"content"`
	TaskID    string    `json:"task_id,omitempty"`
	Timestamp time.Time `json:"timestamp"`
}

// A2AAgentCard describes an agent in the A2A ecosystem.
type A2AAgentCard struct {
	Name         string       `json:"name"`
	Description  string       `json:"description"`
	Version      string       `json:"version"`
	Endpoint     string       `json:"endpoint"`
	Capabilities []string     `json:"capabilities"`
	Tools        []tools.Tool `json:"-"`
}

// A2ARegistry discovers and tracks A2A agents.
type A2ARegistry struct {
	mu     sync.RWMutex
	agents map[string]*A2AAgentCard
}

// NewA2ARegistry creates a new A2ARegistry.
func NewA2ARegistry() *A2ARegistry {
	return &A2ARegistry{
		agents: make(map[string]*A2AAgentCard),
	}
}

// Register adds an agent to the registry.
func (r *A2ARegistry) Register(card *A2AAgentCard) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.agents[card.Endpoint] = card
}

// Discover fetches an agent card from a URL and registers it.
func (r *A2ARegistry) Discover(ctx context.Context, url string) (*A2AAgentCard, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("a2a: discover: build request: %w", err)
	}
	req.Header.Set("Accept", "application/json")

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("a2a: discover: request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("a2a: discover: %s returned status %d", url, resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("a2a: discover: read body: %w", err)
	}

	var card A2AAgentCard
	if err := json.Unmarshal(body, &card); err != nil {
		return nil, fmt.Errorf("a2a: discover: decode: %w", err)
	}

	r.Register(&card)
	return &card, nil
}

// List returns all registered agents.
func (r *A2ARegistry) List() []*A2AAgentCard {
	r.mu.RLock()
	defer r.mu.RUnlock()
	cards := make([]*A2AAgentCard, 0, len(r.agents))
	for _, c := range r.agents {
		cards = append(cards, c)
	}
	return cards
}

// Find finds an agent by name.
func (r *A2ARegistry) Find(name string) *A2AAgentCard {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, c := range r.agents {
		if c.Name == name {
			return c
		}
	}
	return nil
}

// A2ATask represents a delegated task to another agent.
type A2ATask struct {
	ID      string    `json:"id"`
	Agent   string    `json:"agent"`
	Task    string    `json:"task"`
	Status  string    `json:"status"`
	Result  string    `json:"result,omitempty"`
	Created time.Time `json:"created"`
	Updated time.Time `json:"updated"`
}

// A2AOrchestrator delegates tasks to agents and tracks results.
type A2AOrchestrator struct {
	registry *A2ARegistry
	executor *AgentExecutor
	tasks    map[string]*A2ATask
	mu       sync.RWMutex
}

// NewA2AOrchestrator creates an A2AOrchestrator.
func NewA2AOrchestrator(registry *A2ARegistry, executor *AgentExecutor) *A2AOrchestrator {
	return &A2AOrchestrator{
		registry: registry,
		executor: executor,
		tasks:    make(map[string]*A2ATask),
	}
}

// Delegate sends a task to a named agent and returns the result.
func (o *A2AOrchestrator) Delegate(ctx context.Context, agentName, task string) (*A2ATask, error) {
	card := o.registry.Find(agentName)
	if card == nil {
		return nil, fmt.Errorf("a2a: agent %q not found in registry", agentName)
	}

	t := &A2ATask{
		ID:      fmt.Sprintf("a2a-%d", time.Now().UnixNano()),
		Agent:   agentName,
		Task:    task,
		Status:  "running",
		Created: time.Now(),
		Updated: time.Now(),
	}

	o.mu.Lock()
	o.tasks[t.ID] = t
	o.mu.Unlock()

	msg := A2AMessage{
		ID:        t.ID,
		From:      "orchestrator",
		To:        agentName,
		Type:      "task",
		Content:   task,
		TaskID:    t.ID,
		Timestamp: time.Now(),
	}

	msgBody, err := json.Marshal(msg)
	if err != nil {
		t.Status = "failed"
		t.Result = err.Error()
		t.Updated = time.Now()
		return t, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, card.Endpoint, nil)
	if err != nil {
		t.Status = "failed"
		t.Result = err.Error()
		t.Updated = time.Now()
		return t, fmt.Errorf("a2a: delegate: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Body = io.NopCloser(io.LimitReader(
		&msgBodyReader{b: msgBody}, int64(len(msgBody)),
	))
	req.ContentLength = int64(len(msgBody))

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		t.Status = "failed"
		t.Result = err.Error()
		t.Updated = time.Now()
		return t, fmt.Errorf("a2a: delegate: request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		t.Status = "failed"
		t.Result = err.Error()
		t.Updated = time.Now()
		return t, err
	}

	t.Status = "done"
	t.Result = string(body)
	t.Updated = time.Now()
	return t, nil
}

// Tasks returns all tracked tasks.
func (o *A2AOrchestrator) Tasks() []*A2ATask {
	o.mu.RLock()
	defer o.mu.RUnlock()
	tasks := make([]*A2ATask, 0, len(o.tasks))
	for _, t := range o.tasks {
		tasks = append(tasks, t)
	}
	return tasks
}

type msgBodyReader struct {
	b   []byte
	pos int
}

func (r *msgBodyReader) Read(p []byte) (int, error) {
	if r.pos >= len(r.b) {
		return 0, io.EOF
	}
	n := copy(p, r.b[r.pos:])
	r.pos += n
	return n, nil
}
