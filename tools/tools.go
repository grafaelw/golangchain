// Package tools defines the Tool interface and provides built-in tool
// implementations: Calculator, HTTPFetch, DuckDuckGoSearch, and ShellTool.
package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/grafaelw/golangchain/schema"
)

// ErrToolNotFound is returned when an agent requests a tool that is not
// registered in the executor's tool set.
var ErrToolNotFound = errors.New("tool not found")

// ---------------------------------------------------------------------------
// Tool interface
// ---------------------------------------------------------------------------

// Tool is the interface every tool must implement. Tools are called by agents
// when the LLM decides to use them.
type Tool interface {
	// Name returns the unique tool identifier (no spaces; snake_case recommended).
	Name() string
	// Description is shown to the model — be precise about input format.
	Description() string
	// Schema returns a JSON Schema object describing the tool's input.
	// Agents using tool-calling APIs pass this to the model.
	Schema() json.RawMessage
	// Run executes the tool with the given input string and returns a result string.
	Run(ctx context.Context, input string) (string, error)
}

// ToToolDef converts a Tool to the schema.ToolDef used by LLM providers.
func ToToolDef(t Tool) schema.ToolDef {
	return schema.ToolDef{
		Name:        t.Name(),
		Description: t.Description(),
		Parameters:  t.Schema(),
	}
}

// ToToolDefs converts a slice of Tools to []schema.ToolDef.
func ToToolDefs(tools []Tool) []schema.ToolDef {
	defs := make([]schema.ToolDef, len(tools))
	for i, t := range tools {
		defs[i] = ToToolDef(t)
	}
	return defs
}

// FindTool returns the Tool with the given name from a slice, or nil.
func FindTool(tools []Tool, name string) Tool {
	for _, t := range tools {
		if t.Name() == name {
			return t
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// Calculator
// ---------------------------------------------------------------------------

// Calculator evaluates simple arithmetic expressions using a hand-rolled
// parser. It supports +, -, *, /, ^ (power), and parentheses, and functions
// sqrt, abs, floor, ceil, round.
//
// Input:  "2 + 2 * 10"
// Output: "22".
type Calculator struct{}

func (Calculator) Name() string { return "calculator" }
func (Calculator) Description() string {
	return "Evaluates a mathematical expression. Input should be a valid math expression string, e.g. '2 + 2 * 10' or 'sqrt(144)'."
}
func (Calculator) Schema() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"expression": {"type": "string", "description": "The mathematical expression to evaluate."}
		},
		"required": ["expression"]
	}`)
}

func (Calculator) Run(_ context.Context, input string) (string, error) {
	// Try to parse as JSON first (tool-calling agents send JSON)
	var args struct {
		Expression string `json:"expression"`
	}
	if err := json.Unmarshal([]byte(input), &args); err == nil && args.Expression != "" {
		input = args.Expression
	}
	result, err := evalExpr(strings.TrimSpace(input))
	if err != nil {
		return "", fmt.Errorf("calculator: %w", err)
	}
	// Format: strip trailing zeros for clean output
	return strconv.FormatFloat(result, 'f', -1, 64), nil
}

// ---------------------------------------------------------------------------
// HTTPFetch — fetches a URL and returns the body as text
// ---------------------------------------------------------------------------

// HTTPFetch performs an HTTP GET and returns the response body as a string.
// HTML is returned as-is; callers can post-process with an LLM.
//
// Input:  "https://example.com" (plain URL or JSON {"url":"..."}).
type HTTPFetch struct {
	Client    *http.Client
	MaxBytes  int64
	UserAgent string
}

// NewHTTPFetch creates an HTTPFetch with sensible defaults.
func NewHTTPFetch() *HTTPFetch {
	return &HTTPFetch{
		Client:    &http.Client{Timeout: 15 * time.Second},
		MaxBytes:  1 << 20, // 1 MiB
		UserAgent: "golangchain/1.0",
	}
}

func (h *HTTPFetch) Name() string { return "http_fetch" }
func (h *HTTPFetch) Description() string {
	return "Fetches the content of a URL via HTTP GET. Input: a URL string or JSON {\"url\":\"...\"}. Returns the response body (truncated at 1 MiB)."
}
func (h *HTTPFetch) Schema() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"url": {"type": "string", "description": "The URL to fetch."}
		},
		"required": ["url"]
	}`)
}

func (h *HTTPFetch) Run(ctx context.Context, input string) (string, error) {
	rawURL := input
	var args struct {
		URL string `json:"url"`
	}
	if err := json.Unmarshal([]byte(input), &args); err == nil && args.URL != "" {
		rawURL = args.URL
	}
	rawURL = strings.TrimSpace(rawURL)
	if _, err := url.ParseRequestURI(rawURL); err != nil {
		return "", fmt.Errorf("http_fetch: invalid URL %q: %w", rawURL, err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return "", fmt.Errorf("http_fetch: build request: %w", err)
	}
	req.Header.Set("User-Agent", h.UserAgent)

	resp, err := h.Client.Do(req)
	if err != nil {
		return "", fmt.Errorf("http_fetch: request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(io.LimitReader(resp.Body, h.MaxBytes))
	if err != nil {
		return "", fmt.Errorf("http_fetch: read body: %w", err)
	}
	return fmt.Sprintf("HTTP %d\n%s", resp.StatusCode, string(body)), nil
}

// ---------------------------------------------------------------------------
// DuckDuckGoSearch — instant-answer search via DDG API
// ---------------------------------------------------------------------------

// DuckDuckGoSearch queries the DuckDuckGo Instant Answer API.
// It requires no API key and is suitable for lightweight factual queries.
//
// Input:  "capital of France" or JSON {"query":"..."}.
type DuckDuckGoSearch struct {
	Client *http.Client
}

// NewDuckDuckGoSearch creates a DuckDuckGoSearch tool.
func NewDuckDuckGoSearch() *DuckDuckGoSearch {
	return &DuckDuckGoSearch{Client: &http.Client{Timeout: 10 * time.Second}}
}

func (d *DuckDuckGoSearch) Name() string { return "duckduckgo_search" }
func (d *DuckDuckGoSearch) Description() string {
	return "Searches the web using DuckDuckGo and returns a brief answer or abstract. Best for factual questions. Input: the search query string."
}
func (d *DuckDuckGoSearch) Schema() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"query": {"type": "string", "description": "The search query."}
		},
		"required": ["query"]
	}`)
}

func (d *DuckDuckGoSearch) Run(ctx context.Context, input string) (string, error) {
	query := input
	var args struct {
		Query string `json:"query"`
	}
	if err := json.Unmarshal([]byte(input), &args); err == nil && args.Query != "" {
		query = args.Query
	}
	query = strings.TrimSpace(query)

	apiURL := "https://api.duckduckgo.com/?q=" + url.QueryEscape(query) + "&format=json&no_html=1&skip_disambig=1"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, apiURL, nil)
	if err != nil {
		return "", fmt.Errorf("duckduckgo: build request: %w", err)
	}
	req.Header.Set("User-Agent", "golangchain/1.0")

	resp, err := d.Client.Do(req)
	if err != nil {
		return "", fmt.Errorf("duckduckgo: request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	var result struct {
		Abstract       string `json:"Abstract"`
		AbstractSource string `json:"AbstractSource"`
		Answer         string `json:"Answer"`
		AnswerType     string `json:"AnswerType"`
		RelatedTopics  []struct {
			Text string `json:"Text"`
		} `json:"RelatedTopics"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("duckduckgo: decode: %w", err)
	}

	if result.Answer != "" {
		return result.Answer, nil
	}
	if result.Abstract != "" {
		return fmt.Sprintf("%s (Source: %s)", result.Abstract, result.AbstractSource), nil
	}
	// Fall back to first related topic
	var sb strings.Builder
	for i, rt := range result.RelatedTopics {
		if i >= 3 || rt.Text == "" {
			break
		}
		sb.WriteString("- ")
		sb.WriteString(rt.Text)
		sb.WriteString("\n")
	}
	if sb.Len() > 0 {
		return sb.String(), nil
	}
	return "No results found for: " + query, nil
}

// ---------------------------------------------------------------------------
// ShellTool — executes shell commands (opt-in; use with care)
// ---------------------------------------------------------------------------

// ShellTool runs shell commands on the host system. This is a powerful and
// potentially dangerous tool. Only add it to an agent when explicitly required
// and running in a sandboxed environment.
//
// Input: the shell command string or JSON {"command":"..."}.
type ShellTool struct {
	// AllowedCommands is an optional whitelist. If non-nil, only commands
	// whose executable matches one of these names are allowed.
	AllowedCommands []string
}

func (s *ShellTool) Name() string { return "shell" }
func (s *ShellTool) Description() string {
	return "Executes a shell command and returns stdout. Use only for system operations. Input: the command to run (string)."
}
func (s *ShellTool) Schema() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"command": {"type": "string", "description": "The shell command to execute."}
		},
		"required": ["command"]
	}`)
}

func (s *ShellTool) Run(ctx context.Context, input string) (string, error) {
	cmd := input
	var args struct {
		Command string `json:"command"`
	}
	if err := json.Unmarshal([]byte(input), &args); err == nil && args.Command != "" {
		cmd = args.Command
	}
	cmd = strings.TrimSpace(cmd)

	var c *exec.Cmd
	if runtime.GOOS == "windows" {
		c = exec.CommandContext(ctx, "cmd", "/C", cmd)
	} else {
		c = exec.CommandContext(ctx, "sh", "-c", cmd)
	}

	out, err := c.CombinedOutput()
	if err != nil {
		return string(out), fmt.Errorf("shell: %w", err)
	}
	return string(out), nil
}

// ---------------------------------------------------------------------------
// FuncTool — creates a Tool from a plain Go function
// ---------------------------------------------------------------------------

// FuncTool wraps an arbitrary function as a Tool. Useful for custom tools
// without boilerplate.
type FuncTool struct {
	name        string
	description string
	schema      json.RawMessage
	fn          func(ctx context.Context, input string) (string, error)
}

// NewFuncTool constructs a Tool from a function. schema may be nil.
func NewFuncTool(
	name, description string,
	schema json.RawMessage,
	fn func(ctx context.Context, input string) (string, error),
) *FuncTool {
	if schema == nil {
		schema = json.RawMessage(`{"type":"object","properties":{"input":{"type":"string"}},"required":["input"]}`)
	}
	return &FuncTool{name: name, description: description, schema: schema, fn: fn}
}

func (f *FuncTool) Name() string            { return f.name }
func (f *FuncTool) Description() string     { return f.description }
func (f *FuncTool) Schema() json.RawMessage { return f.schema }
func (f *FuncTool) Run(ctx context.Context, input string) (string, error) {
	return f.fn(ctx, input)
}

// ---------------------------------------------------------------------------
// Minimal arithmetic expression evaluator
// ---------------------------------------------------------------------------

// evalExpr is a recursive-descent parser for arithmetic expressions.
// Grammar:
//
//	expr     = term (('+' | '-') term)*
//	term     = factor (('*' | '/') factor)*
//	factor   = ('+' | '-')? primary ('^' factor)?
//	primary  = number | '(' expr ')' | ident '(' expr ')'
func evalExpr(s string) (float64, error) {
	p := &exprParser{input: []rune(s), pos: 0}
	v, err := p.parseExpr()
	if err != nil {
		return 0, err
	}
	p.skipWS()
	if p.pos < len(p.input) {
		return 0, fmt.Errorf("unexpected character %q at position %d", string(p.input[p.pos:]), p.pos)
	}
	return v, nil
}

type exprParser struct {
	input []rune
	pos   int
}

func (p *exprParser) skipWS() {
	for p.pos < len(p.input) && unicode.IsSpace(p.input[p.pos]) {
		p.pos++
	}
}

func (p *exprParser) parseExpr() (float64, error) {
	left, err := p.parseTerm()
	if err != nil {
		return 0, err
	}
	for {
		p.skipWS()
		if p.pos >= len(p.input) {
			break
		}
		op := p.input[p.pos]
		if op != '+' && op != '-' {
			break
		}
		p.pos++
		right, err := p.parseTerm()
		if err != nil {
			return 0, err
		}
		if op == '+' {
			left += right
		} else {
			left -= right
		}
	}
	return left, nil
}

func (p *exprParser) parseTerm() (float64, error) {
	left, err := p.parseFactor()
	if err != nil {
		return 0, err
	}
	for {
		p.skipWS()
		if p.pos >= len(p.input) {
			break
		}
		op := p.input[p.pos]
		if op != '*' && op != '/' {
			break
		}
		p.pos++
		right, err := p.parseFactor()
		if err != nil {
			return 0, err
		}
		if op == '*' {
			left *= right
		} else {
			if right == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			left /= right
		}
	}
	return left, nil
}

func (p *exprParser) parseFactor() (float64, error) {
	p.skipWS()
	sign := 1.0
	if p.pos < len(p.input) && (p.input[p.pos] == '+' || p.input[p.pos] == '-') {
		if p.input[p.pos] == '-' {
			sign = -1
		}
		p.pos++
	}
	base, err := p.parsePrimary()
	if err != nil {
		return 0, err
	}
	base *= sign

	p.skipWS()
	if p.pos < len(p.input) && p.input[p.pos] == '^' {
		p.pos++
		exp, err := p.parseFactor() // right-associative
		if err != nil {
			return 0, err
		}
		// Simple integer exponentiation
		result := 1.0
		if exp >= 0 {
			for i := 0; i < int(exp); i++ {
				result *= base
			}
		} else {
			for i := 0; i < int(-exp); i++ {
				result /= base
			}
		}
		return result, nil
	}
	return base, nil
}

func (p *exprParser) parsePrimary() (float64, error) {
	p.skipWS()
	if p.pos >= len(p.input) {
		return 0, fmt.Errorf("unexpected end of expression")
	}

	// Parenthesised sub-expression
	if p.input[p.pos] == '(' {
		p.pos++
		v, err := p.parseExpr()
		if err != nil {
			return 0, err
		}
		p.skipWS()
		if p.pos >= len(p.input) || p.input[p.pos] != ')' {
			return 0, fmt.Errorf("missing closing parenthesis")
		}
		p.pos++
		return v, nil
	}

	// Named function (sqrt, abs, etc.)
	if unicode.IsLetter(p.input[p.pos]) {
		start := p.pos
		for p.pos < len(p.input) && unicode.IsLetter(p.input[p.pos]) {
			p.pos++
		}
		name := strings.ToLower(string(p.input[start:p.pos]))
		p.skipWS()
		if p.pos >= len(p.input) || p.input[p.pos] != '(' {
			return 0, fmt.Errorf("unknown identifier %q", name)
		}
		p.pos++ // consume '('
		arg, err := p.parseExpr()
		if err != nil {
			return 0, err
		}
		p.skipWS()
		if p.pos >= len(p.input) || p.input[p.pos] != ')' {
			return 0, fmt.Errorf("missing ')' after function %q", name)
		}
		p.pos++
		switch name {
		case "sqrt":
			if arg < 0 {
				return 0, fmt.Errorf("sqrt of negative number")
			}
			// Integer Newton's method for sqrt
			if arg == 0 {
				return 0, nil
			}
			z := arg
			for i := 0; i < 50; i++ {
				z -= (z*z - arg) / (2 * z)
			}
			return z, nil
		case "abs":
			if arg < 0 {
				return -arg, nil
			}
			return arg, nil
		case "floor":
			return float64(int64(arg)), nil
		case "ceil":
			iv := int64(arg)
			if float64(iv) < arg {
				return float64(iv + 1), nil
			}
			return float64(iv), nil
		case "round":
			return float64(int64(arg + 0.5)), nil
		default:
			return 0, fmt.Errorf("unknown function %q", name)
		}
	}

	// Number literal
	start := p.pos
	for p.pos < len(p.input) && (unicode.IsDigit(p.input[p.pos]) || p.input[p.pos] == '.') {
		p.pos++
	}
	if start == p.pos {
		return 0, fmt.Errorf("expected number at position %d, got %q", p.pos, string(p.input[p.pos:]))
	}
	return strconv.ParseFloat(string(p.input[start:p.pos]), 64)
}
