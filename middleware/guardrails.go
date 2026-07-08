package middleware

import (
	"context"
	"fmt"
	"regexp"
	"sync"

	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// ContentFilterMiddleware
// ---------------------------------------------------------------------------

// ContentFilterMiddleware checks inputs and outputs against a filter function.
type ContentFilterMiddleware struct {
	NoOpMiddleware
	AllowFunc func(ctx context.Context, text string) (bool, error)
}

// NewContentFilterMiddleware constructs a ContentFilterMiddleware.
func NewContentFilterMiddleware(allowFn func(ctx context.Context, text string) (bool, error)) *ContentFilterMiddleware {
	return &ContentFilterMiddleware{AllowFunc: allowFn}
}

func (c *ContentFilterMiddleware) Name() string { return "ContentFilter" }

func (c *ContentFilterMiddleware) BeforeModel(ctx context.Context, messages []schema.Message, _ []schema.AgentStep) ([]schema.Message, error) {
	for _, m := range messages {
		allowed, err := c.AllowFunc(ctx, m.Content)
		if err != nil {
			return nil, fmt.Errorf("ContentFilter: %w", err)
		}
		if !allowed {
			return nil, fmt.Errorf("ContentFilter: input blocked")
		}
	}
	return messages, nil
}

func (c *ContentFilterMiddleware) AfterModel(ctx context.Context, gen *schema.Generation) (*schema.Generation, error) {
	if gen == nil {
		return gen, nil
	}
	allowed, err := c.AllowFunc(ctx, gen.Text)
	if err != nil {
		return nil, fmt.Errorf("ContentFilter: %w", err)
	}
	if !allowed {
		return nil, fmt.Errorf("ContentFilter: output blocked")
	}
	return gen, nil
}

func (c *ContentFilterMiddleware) BeforeTool(ctx context.Context, _ string, input string) (string, error) {
	allowed, err := c.AllowFunc(ctx, input)
	if err != nil {
		return "", fmt.Errorf("ContentFilter: %w", err)
	}
	if !allowed {
		return "", fmt.Errorf("ContentFilter: tool input blocked")
	}
	return input, nil
}

func (c *ContentFilterMiddleware) AfterTool(ctx context.Context, toolName string, output string) (string, error) {
	allowed, err := c.AllowFunc(ctx, output)
	if err != nil {
		return "", fmt.Errorf("ContentFilter: %w", err)
	}
	if !allowed {
		return "", fmt.Errorf("ContentFilter: tool %s output blocked", toolName)
	}
	return output, nil
}

// ---------------------------------------------------------------------------
// PIIMiddleware
// ---------------------------------------------------------------------------

// PIIMiddleware detects and masks PII in tool inputs and outputs.
type PIIMiddleware struct {
	NoOpMiddleware
	Patterns   []string
	compiled   []*regexp.Regexp
	compiledMu sync.RWMutex
	maskChar   string
}

// NewPIIMiddleware constructs a PIIMiddleware.
func NewPIIMiddleware(patterns ...string) *PIIMiddleware {
	return &PIIMiddleware{
		Patterns: patterns,
		maskChar: "*",
	}
}

func (p *PIIMiddleware) Name() string { return "PII" }

// WithPIIMaskChar sets the character used for masking.
func WithPIIMaskChar(c string) PIIOption {
	return func(p *PIIMiddleware) { p.maskChar = c }
}

// PIIOption configures a PIIMiddleware.
type PIIOption func(*PIIMiddleware)

func (p *PIIMiddleware) compile() error {
	p.compiledMu.RLock()
	if len(p.compiled) == len(p.Patterns) {
		p.compiledMu.RUnlock()
		return nil
	}
	p.compiledMu.RUnlock()

	p.compiledMu.Lock()
	defer p.compiledMu.Unlock()

	p.compiled = make([]*regexp.Regexp, 0, len(p.Patterns))
	for _, pat := range p.Patterns {
		re, err := regexp.Compile(pat)
		if err != nil {
			return fmt.Errorf("PII: invalid pattern %q: %w", pat, err)
		}
		p.compiled = append(p.compiled, re)
	}
	return nil
}

func (p *PIIMiddleware) BeforeTool(ctx context.Context, toolName string, input string) (string, error) {
	return p.mask(input)
}

func (p *PIIMiddleware) AfterTool(ctx context.Context, toolName string, output string) (string, error) {
	return p.mask(output)
}

func (p *PIIMiddleware) mask(text string) (string, error) {
	if err := p.compile(); err != nil {
		return "", err
	}
	p.compiledMu.RLock()
	defer p.compiledMu.RUnlock()

	result := text
	for _, re := range p.compiled {
		result = re.ReplaceAllStringFunc(result, func(match string) string {
			masked := ""
			for i := 0; i < len(match); i++ {
				masked += p.maskChar
			}
			return masked
		})
	}
	return result, nil
}
