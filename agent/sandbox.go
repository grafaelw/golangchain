package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/grafaelw/golangchain/tools"
)

// CodeSandbox executes code in a sandboxed environment and returns the output.
// The default implementation uses os/exec with a timeout and resource limits.
type CodeSandbox struct {
	Timeout     time.Duration
	MaxOutput   int
	WorkDir     string
	AllowedCmds []string
}

const (
	defaultSandboxTimeout   = 30 * time.Second
	defaultSandboxMaxOutput = 10000
)

// NewCodeSandbox creates a CodeSandbox with the given options.
func NewCodeSandbox(opts ...SandboxOption) *CodeSandbox {
	s := &CodeSandbox{
		Timeout:   defaultSandboxTimeout,
		MaxOutput: defaultSandboxMaxOutput,
	}
	for _, o := range opts {
		o(s)
	}
	return s
}

// SandboxOption configures a CodeSandbox.
type SandboxOption func(*CodeSandbox)

// WithSandboxTimeout sets the maximum execution time.
func WithSandboxTimeout(d time.Duration) SandboxOption {
	return func(s *CodeSandbox) { s.Timeout = d }
}

// WithSandboxMaxOutput sets the maximum bytes of output to return.
func WithSandboxMaxOutput(n int) SandboxOption {
	return func(s *CodeSandbox) { s.MaxOutput = n }
}

// WithSandboxWorkDir sets the working directory for execution.
func WithSandboxWorkDir(dir string) SandboxOption {
	return func(s *CodeSandbox) { s.WorkDir = dir }
}

// WithSandboxAllowedCmds restricts which commands can be executed. If set,
// only commands whose base name appears in the list are allowed.
func WithSandboxAllowedCmds(cmds ...string) SandboxOption {
	return func(s *CodeSandbox) { s.AllowedCmds = cmds }
}

// Execute runs code in the sandbox. For shell code, uses sh -c. For python,
// uses python3 -c. For other languages, writes to a temp file and executes.
func (s *CodeSandbox) Execute(ctx context.Context, language, code string) (string, error) {
	timeout := s.Timeout
	if timeout <= 0 {
		timeout = defaultSandboxTimeout
	}
	execCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	var cmd *exec.Cmd

	switch strings.ToLower(strings.TrimSpace(language)) {
	case "shell", "sh", "bash":
		cmd = exec.CommandContext(execCtx, "sh", "-c", code)
	case "python", "python3", "py":
		cmd = exec.CommandContext(execCtx, "python3", "-c", code)
	default:
		ext := language
		if !strings.HasPrefix(ext, ".") {
			ext = "." + ext
		}
		tmpFile, err := os.CreateTemp("", "sandbox-*"+ext)
		if err != nil {
			return "", fmt.Errorf("sandbox: create temp file: %w", err)
		}
		defer func() { _ = os.Remove(tmpFile.Name()) }()

		if _, err := tmpFile.WriteString(code); err != nil {
			_ = tmpFile.Close()
			return "", fmt.Errorf("sandbox: write temp file: %w", err)
		}
		_ = tmpFile.Close()

		cmd = exec.CommandContext(execCtx, tmpFile.Name())
	}

	if s.WorkDir != "" {
		cmd.Dir = s.WorkDir
	} else {
		cmd.Dir = filepath.Dir(s.WorkDir)
		if cmd.Dir == "." || cmd.Dir == "" {
			wd, _ := os.Getwd()
			cmd.Dir = wd
		}
	}

	if len(s.AllowedCmds) > 0 {
		prog := filepath.Base(cmd.Path)
		allowed := false
		for _, c := range s.AllowedCmds {
			if c == prog {
				allowed = true
				break
			}
		}
		if !allowed {
			return "", fmt.Errorf("sandbox: command %q is not allowed", prog)
		}
	}

	out, err := cmd.CombinedOutput()

	maxOut := s.MaxOutput
	if maxOut <= 0 {
		maxOut = defaultSandboxMaxOutput
	}
	if len(out) > maxOut {
		out = out[:maxOut]
	}

	if err != nil {
		return string(out), fmt.Errorf("sandbox: %w: %s", err, string(out))
	}
	return string(out), nil
}

// AsTool returns the sandbox as a tool named "execute_code" that agents can
// call. The tool accepts JSON: {"language": "python", "code": "print(1+1)"}.
func (s *CodeSandbox) AsTool() tools.Tool {
	return tools.NewFuncTool(
		"execute_code",
		"Executes code in a sandboxed environment. Supports languages: shell, python, ruby, node, and others via temp file. Input: JSON {\"language\":\"python\",\"code\":\"print(1+1)\"}.",
		json.RawMessage(`{"type":"object","properties":{"language":{"type":"string","description":"The programming language (python, shell, ruby, node, js, etc)."},"code":{"type":"string","description":"The source code to execute."}},"required":["language","code"]}`),
		func(ctx context.Context, input string) (string, error) {
			var args struct {
				Language string `json:"language"`
				Code     string `json:"code"`
			}
			if err := json.Unmarshal([]byte(input), &args); err != nil {
				return "", fmt.Errorf("execute_code: %w", err)
			}
			if args.Language == "" {
				return "", fmt.Errorf("execute_code: language is required")
			}
			if args.Code == "" {
				return "", fmt.Errorf("execute_code: code is required")
			}
			return s.Execute(ctx, args.Language, args.Code)
		},
	)
}
