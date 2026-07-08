package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/middleware"
	"github.com/grafaelw/golangchain/tools"
)

// ---------------------------------------------------------------------------
// SubAgentMiddleware
// ---------------------------------------------------------------------------

// SubAgentMiddleware enables the main agent to spawn sub-agents that run in
// isolated contexts. When the main agent calls a subagent tool, a new agent
// executor is created with the subagent's configuration and runs to completion.
type SubAgentMiddleware struct {
	middleware.NoOpMiddleware
	SubAgents     map[string]*SubAgentConfig
	MaxConcurrent int
}

// SubAgentConfig defines a subagent that the main agent can invoke as a tool.
type SubAgentConfig struct {
	Name         string
	Description  string
	SystemPrompt string
	Model        llm.LLM
	Tools        []tools.Tool
	MaxIter      int
}

// NewSubAgentMiddleware constructs a SubAgentMiddleware from the given
// subagent configs. Each subagent is registered under its Name.
func NewSubAgentMiddleware(subAgents ...*SubAgentConfig) *SubAgentMiddleware {
	s := &SubAgentMiddleware{
		SubAgents:     make(map[string]*SubAgentConfig, len(subAgents)),
		MaxConcurrent: 4,
	}
	for _, sa := range subAgents {
		s.SubAgents[sa.Name] = sa
	}
	return s
}

func (s *SubAgentMiddleware) Name() string { return "SubAgent" }

// AsTools returns subagents as a []tools.Tool so the main agent can call them.
// Each subagent is exposed as a tool whose name matches SubAgentConfig.Name.
func (s *SubAgentMiddleware) AsTools() []tools.Tool {
	var subs []*SubAgentConfig
	for _, v := range s.SubAgents {
		subs = append(subs, v)
	}

	toolList := make([]tools.Tool, len(subs))
	sem := make(chan struct{}, s.MaxConcurrent)

	for i, sa := range subs {
		sa := sa // capture for closure
		toolList[i] = tools.NewFuncTool(
			sa.Name,
			sa.Description,
			json.RawMessage(`{"type":"object","properties":{"input":{"type":"string","description":"The task or question for the subagent."}},"required":["input"]}`),
			func(ctx context.Context, input string) (string, error) {
				sem <- struct{}{}
				defer func() { <-sem }()

				// Parse input — try JSON first, fall back to raw string.
				var args struct {
					Input string `json:"input"`
				}
				if err := json.Unmarshal([]byte(input), &args); err == nil && args.Input != "" {
					input = args.Input
				}

				maxIter := sa.MaxIter
				if maxIter <= 0 {
					maxIter = 10
				}

				subAgent := NewToolCallingAgent(sa.Model, sa.Tools, sa.SystemPrompt)
				executor := NewAgentExecutor(subAgent, sa.Tools, WithMaxIter(maxIter))

				result, err := executor.Run(ctx, input)
				if err != nil {
					return "", fmt.Errorf("subagent %q: %w", sa.Name, err)
				}
				return result, nil
			},
		)
	}

	// Ensure sem channel is kept alive — the closures reference it.
	_ = sem

	return toolList
}

// ---------------------------------------------------------------------------
// Parallel helper
// ---------------------------------------------------------------------------

// RunSubAgentsParallel executes multiple subagents concurrently and collects
// their results. The input map maps subagent name to input string.
func (s *SubAgentMiddleware) RunSubAgentsParallel(ctx context.Context, inputs map[string]string) (map[string]string, error) {
	results := make(map[string]string, len(inputs))
	var mu sync.Mutex
	var wg sync.WaitGroup
	errCh := make(chan error, len(inputs))

	sem := make(chan struct{}, s.MaxConcurrent)
	if s.MaxConcurrent <= 0 {
		sem = make(chan struct{}, 4)
	}

	for name, input := range inputs {
		sa, ok := s.SubAgents[name]
		if !ok {
			return nil, fmt.Errorf("subagents: unknown subagent %q", name)
		}

		wg.Add(1)
		go func(sa *SubAgentConfig, input string) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			maxIter := sa.MaxIter
			if maxIter <= 0 {
				maxIter = 10
			}

			subAgent := NewToolCallingAgent(sa.Model, sa.Tools, sa.SystemPrompt)
			executor := NewAgentExecutor(subAgent, sa.Tools, WithMaxIter(maxIter))

			result, err := executor.Run(ctx, input)
			mu.Lock()
			if err != nil {
				errCh <- fmt.Errorf("subagent %q: %w", sa.Name, err)
			} else {
				results[sa.Name] = result
			}
			mu.Unlock()
		}(sa, input)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		return nil, err
	}
	return results, nil
}
