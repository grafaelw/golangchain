// This file adds a PlanAndExecute agent, complementing the ReAct and
// ToolCalling agents. It runs in two phases:
//
//  1. Plan: a planner LLM produces an ordered list of natural-language steps.
//  2. Execute: an inner tool-calling executor tackles each step in turn,
//     with the running answer fed forward as additional context.
//
// PlanAndExecute is a good fit for multi-step research and complex tasks
// where a single ReAct loop meanders. It composes with the same
// AgentExecutor as ReAct/ToolCalling agents.
package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/grafaelw/golangchain/callbacks"
	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/schema"
	"github.com/grafaelw/golangchain/tools"
)

// ---------------------------------------------------------------------------
// PlanAndExecuteAgent
// ---------------------------------------------------------------------------

// PlanAndExecuteAgent is a two-stage agent: a planner produces steps, then
// an inner executor performs them.
type PlanAndExecuteAgent struct {
	Planner      llm.LLM
	Executor     Agent // typically a ToolCallingAgent or ReActAgent
	Tools        []tools.Tool
	PlannerPrompt string // must contain {{ .objective }} and {{ .tools }}
	StepPrompt   string // must contain {{ .step }}, {{ .objective }}, {{ .prior }}
	LLMOptions   []llm.Option

	// Populated after Plan; also emitted as thoughts in the AgentExecutor stream.
	plannedSteps []string
	step         int
	prior        []string
	objective    string
}

// DefaultPlannerPrompt asks the planner LLM for a short ordered plan.
const DefaultPlannerPrompt = `You are a planner. Break the user's objective into a short numbered list of atomic steps. Each step must be executable using ONLY these tools:
{{ .tools }}

Return ONE step per line, prefixed by "N." with no other text.

Objective: {{ .objective }}

Plan:`

// DefaultStepPrompt is passed to the inner executor for each step.
const DefaultStepPrompt = `Objective: {{ .objective }}

Prior results (may be empty):
{{ .prior }}

Now perform ONLY this next step and report its result:
{{ .step }}`

// NewPlanAndExecuteAgent constructs a PlanAndExecuteAgent.
// executor is typically a NewToolCallingAgent that shares agentTools.
func NewPlanAndExecuteAgent(planner llm.LLM, executor Agent, agentTools []tools.Tool, opts ...llm.Option) *PlanAndExecuteAgent {
	return &PlanAndExecuteAgent{
		Planner:       planner,
		Executor:      executor,
		Tools:         agentTools,
		PlannerPrompt: DefaultPlannerPrompt,
		StepPrompt:    DefaultStepPrompt,
		LLMOptions:    opts,
	}
}

// Name identifies this agent for callbacks.
func (a *PlanAndExecuteAgent) Name() string { return "PlanAndExecuteAgent" }

// Plan returns the next action (or finish) at each AgentExecutor iteration.
// On the first call it invokes the planner; each subsequent call delegates a
// single step to the inner executor.
func (a *PlanAndExecuteAgent) Plan(ctx context.Context, messages []schema.Message, steps []schema.AgentStep) ([]schema.AgentAction, *schema.AgentFinish, error) {
	// First call: derive objective from the last human message and build the plan.
	if a.plannedSteps == nil {
		a.objective = lastHumanMessage(messages)
		if a.objective == "" {
			return nil, nil, fmt.Errorf("PlanAndExecute: no human message to plan from")
		}
		plan, err := a.buildPlan(ctx, a.objective)
		if err != nil {
			return nil, nil, err
		}
		if len(plan) == 0 {
			return nil, &schema.AgentFinish{Output: "No plan produced.", Log: "empty plan"}, nil
		}
		a.plannedSteps = plan
		a.step = 0
	}

	// Track prior observations produced by the executor.
	if n := len(steps); n > len(a.prior) {
		a.prior = append(a.prior, steps[n-1].Observation)
	}

	// If all steps done, wrap up with a summary answer.
	if a.step >= len(a.plannedSteps) {
		return nil, &schema.AgentFinish{
			Output: a.finalAnswer(),
			Log:    fmt.Sprintf("completed %d step(s)", len(a.plannedSteps)),
		}, nil
	}

	// Ask the inner executor to perform the next step.
	stepText := a.plannedSteps[a.step]
	a.step++

	stepMsg := renderStepPrompt(a.StepPrompt, a.objective, stepText, a.prior)
	subMessages := []schema.Message{schema.NewHumanMessage(stepMsg)}

	actions, finish, err := a.Executor.Plan(ctx, subMessages, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("PlanAndExecute: step %d: %w", a.step, err)
	}
	// If the inner executor finished immediately (no tool call), capture its
	// text as an observation and move on next iteration.
	if finish != nil {
		a.prior = append(a.prior, finish.Output)
		// Re-plan: recurse into ourselves to advance to next step or finish.
		return a.Plan(ctx, messages, steps)
	}
	// Prepend the step label to the log for visibility in traces.
	for i := range actions {
		actions[i].Log = fmt.Sprintf("[Step %d/%d: %s] %s",
			a.step, len(a.plannedSteps), stepText, actions[i].Log)
	}
	return actions, nil, nil
}

func (a *PlanAndExecuteAgent) buildPlan(ctx context.Context, objective string) ([]string, error) {
	var tb strings.Builder
	for _, t := range a.Tools {
		fmt.Fprintf(&tb, "- %s: %s\n", t.Name(), t.Description())
	}
	p := strings.ReplaceAll(a.PlannerPrompt, "{{ .objective }}", objective)
	p = strings.ReplaceAll(p, "{{ .tools }}", strings.TrimSpace(tb.String()))

	cm := callbacks.CallbackManagerFromContext(ctx)
	llmCtx := ctx
	if cm != nil {
		llmCtx = callbacks.WithRunID(ctx, callbacks.NewRunID())
		cm.OnLLMStart(llmCtx, a.Planner.ModelName(), []schema.Message{schema.NewHumanMessage(p)})
	}
	gen, err := a.Planner.Generate(llmCtx, []schema.Message{schema.NewHumanMessage(p)}, a.LLMOptions...)
	if err != nil {
		if cm != nil {
			cm.OnError(llmCtx, "PlanAndExecute planner", err)
		}
		return nil, fmt.Errorf("planner: %w", err)
	}
	if cm != nil {
		cm.OnLLMEnd(llmCtx, a.Planner.ModelName(), gen)
	}
	return parsePlan(gen.Text), nil
}

func (a *PlanAndExecuteAgent) finalAnswer() string {
	if len(a.prior) == 0 {
		return ""
	}
	return strings.TrimSpace(a.prior[len(a.prior)-1])
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func lastHumanMessage(msgs []schema.Message) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == schema.RoleHuman {
			return msgs[i].Content
		}
	}
	return ""
}

func parsePlan(text string) []string {
	var out []string
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// Accept "1.", "1)", "- ", "* "
		line = strings.TrimLeft(line, "-*0123456789. )")
		line = strings.TrimSpace(line)
		if line != "" {
			out = append(out, line)
		}
	}
	return out
}

func renderStepPrompt(tmpl, objective, step string, prior []string) string {
	p := strings.ReplaceAll(tmpl, "{{ .objective }}", objective)
	p = strings.ReplaceAll(p, "{{ .step }}", step)
	if len(prior) == 0 {
		p = strings.ReplaceAll(p, "{{ .prior }}", "(none)")
	} else {
		var b strings.Builder
		for i, s := range prior {
			fmt.Fprintf(&b, "%d. %s\n", i+1, s)
		}
		p = strings.ReplaceAll(p, "{{ .prior }}", strings.TrimSpace(b.String()))
	}
	return p
}
