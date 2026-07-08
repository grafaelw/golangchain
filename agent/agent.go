// Package agent implements ReActAgent, ToolCallingAgent, and AgentExecutor —
// the golangchain equivalent of LangChain's agent framework.
//
// Two strategies are provided:
//   - ReActAgent: parses Thought/Action/Observation text loops. Works with
//     any LLM that can follow the ReAct prompt format.
//   - ToolCallingAgent: uses the model's native function/tool-calling API.
//     Requires a model that supports tool_calls (GPT-4o, Claude 3, Gemini).
//
// Both agents are driven by AgentExecutor, which owns the run loop, memory,
// tool dispatch, and streaming of AgentEvent values.
package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/grafaelw/golangchain/callbacks"
	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/memory"
	"github.com/grafaelw/golangchain/schema"
	"github.com/grafaelw/golangchain/tools"
)

// ---------------------------------------------------------------------------
// AgentEvent — discriminated union for streaming agent output
// ---------------------------------------------------------------------------

// EventType identifies the kind of agent event.
type EventType string

const (
	EventThought     EventType = "thought"
	EventToolCall    EventType = "tool_call"
	EventToolResult  EventType = "tool_result"
	EventFinalAnswer EventType = "final_answer"
	EventError       EventType = "error"
)

// AgentEvent is a single event emitted during an agent run.
// Inspect Type to determine which fields are populated.
type AgentEvent struct {
	Type        EventType
	Thought     string             // EventThought
	Action      schema.AgentAction // EventToolCall
	Observation string             // EventToolResult
	Answer      string             // EventFinalAnswer
	Err         error              // EventError
}

// ---------------------------------------------------------------------------
// Agent interface
// ---------------------------------------------------------------------------

// Agent produces the next AgentAction (or a finish signal) given the current
// conversation and intermediate steps.
type Agent interface {
	// Plan returns either a list of actions to execute or a finish signal.
	// It must not call tools itself.
	Plan(ctx context.Context, messages []schema.Message, steps []schema.AgentStep) ([]schema.AgentAction, *schema.AgentFinish, error)

	// Name returns a human-readable identifier for this agent strategy.
	Name() string
}

// ---------------------------------------------------------------------------
// AgentExecutor — the run loop
// ---------------------------------------------------------------------------

// AgentExecutor drives the agent/tool loop to completion. It:
//  1. Calls agent.Plan to get actions.
//  2. Calls each tool.Run for each action.
//  3. Appends observations to the step history.
//  4. Repeats until a finish signal or MaxIter is reached.
type AgentExecutor struct {
	Agent     Agent
	Tools     []tools.Tool
	Memory    memory.Memory // optional; injects history into prompts
	Callbacks *callbacks.CallbackManager
	MaxIter   int  // default: 10
	Verbose   bool // if true, prints thoughts/actions to stdout
}

// NewAgentExecutor constructs an AgentExecutor.
func NewAgentExecutor(agent Agent, agentTools []tools.Tool, opts ...ExecutorOption) *AgentExecutor {
	e := &AgentExecutor{
		Agent:   agent,
		Tools:   agentTools,
		MaxIter: 10,
	}
	for _, o := range opts {
		o(e)
	}
	return e
}

// ExecutorOption configures an AgentExecutor.
type ExecutorOption func(*AgentExecutor)

func WithMemory(m memory.Memory) ExecutorOption {
	return func(e *AgentExecutor) { e.Memory = m }
}
func WithCallbackManager(cm *callbacks.CallbackManager) ExecutorOption {
	return func(e *AgentExecutor) { e.Callbacks = cm }
}
func WithMaxIter(n int) ExecutorOption {
	return func(e *AgentExecutor) { e.MaxIter = n }
}
func WithVerbose(v bool) ExecutorOption {
	return func(e *AgentExecutor) { e.Verbose = v }
}

// Run executes the agent loop and returns the final answer string.
func (e *AgentExecutor) Run(ctx context.Context, input string) (string, error) {
	var finalAnswer string
	for event := range e.streamInternal(ctx, input) {
		if event.Err != nil {
			return "", event.Err
		}
		if event.Type == EventFinalAnswer {
			finalAnswer = event.Answer
		}
	}
	return finalAnswer, nil
}

// Stream executes the agent loop and yields AgentEvents in real time.
// The channel is closed when the agent finishes or errors.
func (e *AgentExecutor) Stream(ctx context.Context, input string) <-chan AgentEvent {
	return e.streamInternal(ctx, input)
}

func (e *AgentExecutor) streamInternal(ctx context.Context, input string) <-chan AgentEvent {
	ch := make(chan AgentEvent, 16)
	go func() {
		defer close(ch)

		// Propagate callbacks into context so agent Plan() methods can fire
		// LLM-level events without holding a direct reference to the manager.
		// Also inject a root run ID for the executor so nested spans depth correctly.
		agentCtx := ctx
		if e.Callbacks != nil {
			agentCtx = callbacks.WithCallbackManager(ctx, e.Callbacks)
			agentCtx = callbacks.WithRunID(agentCtx, callbacks.NewRunID())
		}

		// Build initial messages
		messages := []schema.Message{schema.NewHumanMessage(input)}

		// Inject memory
		if e.Memory != nil {
			vars, err := e.Memory.LoadMemoryVariables(agentCtx)
			if err == nil {
				if hist, ok := vars["history"]; ok {
					if histMsgs, ok := hist.([]schema.Message); ok {
						messages = append(histMsgs, messages...)
					}
				}
			}
		}

		var steps []schema.AgentStep

		for iter := 0; iter < e.MaxIter; iter++ {
			// Ask the agent what to do next
			actions, finish, err := e.Agent.Plan(agentCtx, messages, steps)
			if err != nil {
				ch <- AgentEvent{Type: EventError, Err: fmt.Errorf("agent plan iter %d: %w", iter, err)}
				return
			}

			// Agent decided it's done
			if finish != nil {
				if e.Callbacks != nil {
					e.Callbacks.OnAgentFinish(agentCtx, *finish)
				}
				// Save to memory
				if e.Memory != nil {
					_ = e.Memory.SaveContext(agentCtx, input, finish.Output)
				}
				ch <- AgentEvent{Type: EventFinalAnswer, Answer: finish.Output}
				return
			}

			// Execute each action
			for _, action := range actions {
				if e.Callbacks != nil {
					e.Callbacks.OnAgentAction(agentCtx, action)
				}
				if action.Log != "" {
					ch <- AgentEvent{Type: EventThought, Thought: action.Log}
				}
				ch <- AgentEvent{Type: EventToolCall, Action: action}

				// Create a tool-level run ID nested under the agent run.
				toolCtx := agentCtx
				if e.Callbacks != nil {
					toolCtx = callbacks.WithRunID(agentCtx, callbacks.NewRunID())
					e.Callbacks.OnToolStart(toolCtx, action.Tool, action.ToolInput)
				}
				observation, toolErr := e.runTool(toolCtx, action)
				if toolErr != nil {
					observation = "Error: " + toolErr.Error()
				}
				if e.Callbacks != nil {
					e.Callbacks.OnToolEnd(toolCtx, action.Tool, observation)
				}
				ch <- AgentEvent{Type: EventToolResult, Observation: observation}

				steps = append(steps, schema.AgentStep{Action: action, Observation: observation})

				// Append tool result to message history for next plan call
				messages = append(messages,
					schema.Message{
						Role:    schema.RoleAI,
						Content: action.Log,
						ToolCalls: []schema.ToolCall{{
							Name:      action.Tool,
							Arguments: json.RawMessage(action.ToolInput),
						}},
					},
					schema.NewToolMessage(observation, "", action.Tool),
				)
			}
		}

		ch <- AgentEvent{
			Type: EventError,
			Err:  fmt.Errorf("agent exceeded maximum iterations (%d)", e.MaxIter),
		}
	}()
	return ch
}

func (e *AgentExecutor) runTool(ctx context.Context, action schema.AgentAction) (string, error) {
	t := tools.FindTool(e.Tools, action.Tool)
	if t == nil {
		return "", fmt.Errorf("tool %q not found", action.Tool)
	}
	return t.Run(ctx, action.ToolInput)
}

// ---------------------------------------------------------------------------
// ToolCallingAgent — uses native tool-calling APIs
// ---------------------------------------------------------------------------

// ToolCallingAgent uses the model's native function/tool-calling capability.
// Works with OpenAI GPT-4o, Anthropic Claude 3, and Google Gemini 1.5+.
type ToolCallingAgent struct {
	llm          llm.LLM
	llmOpts      []llm.Option
	systemPrompt string
	tools        []tools.Tool
}

// NewToolCallingAgent constructs a ToolCallingAgent.
func NewToolCallingAgent(model llm.LLM, agentTools []tools.Tool, systemPrompt string, opts ...llm.Option) *ToolCallingAgent {
	return &ToolCallingAgent{
		llm:          model,
		llmOpts:      opts,
		systemPrompt: systemPrompt,
		tools:        agentTools,
	}
}

func (a *ToolCallingAgent) Name() string { return "ToolCallingAgent" }

// Plan calls the LLM with tool definitions and parses the response into actions
// or a finish signal.
func (a *ToolCallingAgent) Plan(ctx context.Context, messages []schema.Message, _ []schema.AgentStep) ([]schema.AgentAction, *schema.AgentFinish, error) {
	// Build the message list including the system prompt
	var msgs []schema.Message
	if a.systemPrompt != "" {
		msgs = append(msgs, schema.NewSystemMessage(a.systemPrompt))
	}
	msgs = append(msgs, messages...)

	// Attach tool definitions
	toolDefs := tools.ToToolDefs(a.tools)
	callOpts := append(a.llmOpts, llm.WithTools(toolDefs...))

	// Fire LLM callbacks if a manager was propagated via context.
	cm := callbacks.CallbackManagerFromContext(ctx)
	llmCtx := ctx
	if cm != nil {
		llmCtx = callbacks.WithRunID(ctx, callbacks.NewRunID())
		cm.OnLLMStart(llmCtx, a.llm.ModelName(), msgs)
	}

	gen, err := a.llm.Generate(llmCtx, msgs, callOpts...)
	if err != nil {
		if cm != nil {
			cm.OnError(llmCtx, "ToolCallingAgent", err)
		}
		return nil, nil, fmt.Errorf("ToolCallingAgent: llm: %w", err)
	}
	if cm != nil {
		cm.OnLLMEnd(llmCtx, a.llm.ModelName(), gen)
	}

	// If the model returned tool calls, convert them to AgentActions
	if len(gen.Message.ToolCalls) > 0 {
		var actions []schema.AgentAction
		for _, tc := range gen.Message.ToolCalls {
			actions = append(actions, schema.AgentAction{
				Tool:      tc.Name,
				ToolInput: string(tc.Arguments),
				Log:       gen.Message.Content,
			})
		}
		return actions, nil, nil
	}

	// No tool calls → final answer
	return nil, &schema.AgentFinish{
		Output: strings.TrimSpace(gen.Text),
		Log:    gen.Text,
	}, nil
}

// ---------------------------------------------------------------------------
// ReActAgent — text-based Thought/Action/Observation loop
// ---------------------------------------------------------------------------

// ReActAgent implements the ReAct (Reason + Act) paradigm using plain text.
// It works with any LLM — no tool-calling API required.
// The LLM must follow the ReAct prompt format embedded in the system prompt.
type ReActAgent struct {
	llm     llm.LLM
	llmOpts []llm.Option
	tools   []tools.Tool
}

// NewReActAgent constructs a ReActAgent.
func NewReActAgent(model llm.LLM, agentTools []tools.Tool, opts ...llm.Option) *ReActAgent {
	return &ReActAgent{llm: model, llmOpts: opts, tools: agentTools}
}

func (a *ReActAgent) Name() string { return "ReActAgent" }

// Plan sends the full conversation (with ReAct system prompt injected) to the
// LLM and parses the structured response.
func (a *ReActAgent) Plan(ctx context.Context, messages []schema.Message, steps []schema.AgentStep) ([]schema.AgentAction, *schema.AgentFinish, error) {
	system := buildReActSystemPrompt(a.tools)
	scratchpad := buildScratchpad(steps)

	var msgs []schema.Message
	msgs = append(msgs, schema.NewSystemMessage(system))
	msgs = append(msgs, messages...)
	if scratchpad != "" {
		msgs = append(msgs, schema.NewAIMessage(scratchpad))
	}

	// Fire LLM callbacks if a manager was propagated via context.
	cm := callbacks.CallbackManagerFromContext(ctx)
	llmCtx := ctx
	if cm != nil {
		llmCtx = callbacks.WithRunID(ctx, callbacks.NewRunID())
		cm.OnLLMStart(llmCtx, a.llm.ModelName(), msgs)
	}

	gen, err := a.llm.Generate(llmCtx, msgs, a.llmOpts...)
	if err != nil {
		if cm != nil {
			cm.OnError(llmCtx, "ReActAgent", err)
		}
		return nil, nil, fmt.Errorf("ReActAgent: llm: %w", err)
	}
	if cm != nil {
		cm.OnLLMEnd(llmCtx, a.llm.ModelName(), gen)
	}

	return parseReActOutput(gen.Text)
}

// buildReActSystemPrompt constructs the ReAct-format system prompt.
func buildReActSystemPrompt(agentTools []tools.Tool) string {
	var sb strings.Builder
	sb.WriteString(`You are an AI assistant that can use tools to answer questions.

Available tools:
`)
	for _, t := range agentTools {
		fmt.Fprintf(&sb, "- %s: %s\n", t.Name(), t.Description())
	}
	sb.WriteString(`
Use the following format strictly:

Thought: [your reasoning about what to do]
Action: [tool name, must be one of: `)
	names := make([]string, len(agentTools))
	for i, t := range agentTools {
		names[i] = t.Name()
	}
	sb.WriteString(strings.Join(names, ", "))
	sb.WriteString(`]
Action Input: [input to the tool]
Observation: [result from the tool]
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer.
Final Answer: [your final answer to the user]

Begin!`)
	return sb.String()
}

func buildScratchpad(steps []schema.AgentStep) string {
	if len(steps) == 0 {
		return ""
	}
	var sb strings.Builder
	for _, step := range steps {
		sb.WriteString("Thought: " + step.Action.Log + "\n")
		sb.WriteString("Action: " + step.Action.Tool + "\n")
		sb.WriteString("Action Input: " + step.Action.ToolInput + "\n")
		sb.WriteString("Observation: " + step.Observation + "\n")
	}
	return sb.String()
}

// parseReActOutput parses the LLM text response from a ReAct agent.
func parseReActOutput(text string) ([]schema.AgentAction, *schema.AgentFinish, error) {
	text = strings.TrimSpace(text)

	// Check for Final Answer
	if idx := strings.Index(text, "Final Answer:"); idx != -1 {
		answer := strings.TrimSpace(text[idx+len("Final Answer:"):])
		return nil, &schema.AgentFinish{Output: answer, Log: text}, nil
	}

	// Parse Action / Action Input
	lines := strings.Split(text, "\n")
	var thought strings.Builder
	var actionName, actionInput string

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "Action:") {
			actionName = strings.TrimSpace(strings.TrimPrefix(line, "Action:"))
		} else if strings.HasPrefix(line, "Action Input:") {
			actionInput = strings.TrimSpace(strings.TrimPrefix(line, "Action Input:"))
		} else if strings.HasPrefix(line, "Thought:") {
			if thought.Len() > 0 {
				thought.WriteString("\n")
			}
			thought.WriteString(strings.TrimSpace(strings.TrimPrefix(line, "Thought:")))
		}
	}

	if actionName == "" {
		// Model produced text but no parseable action — treat as final answer
		return nil, &schema.AgentFinish{Output: text, Log: text}, nil
	}

	return []schema.AgentAction{{
		Tool:      actionName,
		ToolInput: actionInput,
		Log:       thought.String(),
	}}, nil, nil
}
