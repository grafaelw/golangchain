package middleware

import (
	"context"
	"fmt"
)

// ---------------------------------------------------------------------------
// HumanInTheLoopMiddleware
// ---------------------------------------------------------------------------

// HumanInTheLoopMiddleware intercepts tool calls that match configured patterns
// and requires human approval before execution.
type HumanInTheLoopMiddleware struct {
	NoOpMiddleware
	ApproveFunc  func(ctx context.Context, toolName, input string) (bool, string, error)
	ToolPatterns []string
}

// HITLOption configures a HumanInTheLoopMiddleware.
type HITLOption func(*HumanInTheLoopMiddleware)

// NewHumanInTheLoopMiddleware constructs a HumanInTheLoopMiddleware.
func NewHumanInTheLoopMiddleware(approveFn func(ctx context.Context, toolName, input string) (bool, string, error), opts ...HITLOption) *HumanInTheLoopMiddleware {
	h := &HumanInTheLoopMiddleware{
		ApproveFunc: approveFn,
	}
	for _, o := range opts {
		o(h)
	}
	return h
}

func (h *HumanInTheLoopMiddleware) Name() string { return "HumanInTheLoop" }

// WithHITLToolPatterns sets the tool names that require approval.
// An empty list means all tools require approval.
func WithHITLToolPatterns(patterns ...string) HITLOption {
	return func(h *HumanInTheLoopMiddleware) { h.ToolPatterns = patterns }
}

func (h *HumanInTheLoopMiddleware) BeforeTool(ctx context.Context, toolName string, input string) (string, error) {
	if !h.requiresApproval(toolName) {
		return input, nil
	}
	approved, modified, err := h.ApproveFunc(ctx, toolName, input)
	if err != nil {
		return "", err
	}
	if !approved {
		return "", fmt.Errorf("human-in-the-loop: tool %q not approved", toolName)
	}
	return modified, nil
}

func (h *HumanInTheLoopMiddleware) requiresApproval(toolName string) bool {
	if len(h.ToolPatterns) == 0 {
		return true
	}

	for _, pattern := range h.ToolPatterns {
		if pattern == toolName {
			return true
		}
	}
	return false
}
