// Package prompt provides PromptTemplate and ChatPromptTemplate — the
// golangchain equivalents of LangChain's prompt layer.
//
// Variables are delimited with double curly braces: {{.VarName}}
// (Go's text/template syntax), giving full access to template functions.
package prompt

import (
	"bytes"
	"fmt"
	"strings"
	"text/template"

	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// PromptTemplate — plain string template
// ---------------------------------------------------------------------------

// PromptTemplate renders a string template with named variables.
// Template syntax is Go's text/template: use {{.VarName}} for substitution.
//
//	pt := prompt.NewPromptTemplate("Translate {{.Text}} into {{.Language}}.")
//	result, err := pt.Format(map[string]any{"Text": "hello", "Language": "French"})
type PromptTemplate struct {
	template *template.Template
	raw      string
}

// NewPromptTemplate constructs a PromptTemplate from a template string.
// Returns an error if the template fails to parse.
func NewPromptTemplate(tmpl string) (*PromptTemplate, error) {
	t, err := template.New("prompt").Option("missingkey=error").Parse(tmpl)
	if err != nil {
		return nil, fmt.Errorf("prompt: parse template: %w", err)
	}
	return &PromptTemplate{template: t, raw: tmpl}, nil
}

// MustNewPromptTemplate is like NewPromptTemplate but panics on error.
// Use only in package-level var blocks or tests.
func MustNewPromptTemplate(tmpl string) *PromptTemplate {
	pt, err := NewPromptTemplate(tmpl)
	if err != nil {
		panic(err)
	}
	return pt
}

// Format renders the template with the supplied variables.
func (p *PromptTemplate) Format(vars map[string]any) (string, error) {
	var buf bytes.Buffer
	if err := p.template.Execute(&buf, vars); err != nil {
		return "", fmt.Errorf("prompt: format: %w", err)
	}
	return buf.String(), nil
}

// Template returns the raw template string.
func (p *PromptTemplate) Template() string { return p.raw }

// ---------------------------------------------------------------------------
// MessageTemplate — one slot in a ChatPromptTemplate
// ---------------------------------------------------------------------------

// MessageTemplate pairs a Role with a PromptTemplate.
type MessageTemplate struct {
	Role     schema.Role
	Template *PromptTemplate
}

// NewSystemMessageTemplate creates a system message slot.
func NewSystemMessageTemplate(tmpl string) (MessageTemplate, error) {
	pt, err := NewPromptTemplate(tmpl)
	if err != nil {
		return MessageTemplate{}, err
	}
	return MessageTemplate{Role: schema.RoleSystem, Template: pt}, nil
}

// NewHumanMessageTemplate creates a human message slot.
func NewHumanMessageTemplate(tmpl string) (MessageTemplate, error) {
	pt, err := NewPromptTemplate(tmpl)
	if err != nil {
		return MessageTemplate{}, err
	}
	return MessageTemplate{Role: schema.RoleHuman, Template: pt}, nil
}

// NewAIMessageTemplate creates an AI message slot.
func NewAIMessageTemplate(tmpl string) (MessageTemplate, error) {
	pt, err := NewPromptTemplate(tmpl)
	if err != nil {
		return MessageTemplate{}, err
	}
	return MessageTemplate{Role: schema.RoleAI, Template: pt}, nil
}

// MustSystem, MustHuman, MustAI — panic-on-error variants for init code.

func MustSystem(tmpl string) MessageTemplate {
	mt, err := NewSystemMessageTemplate(tmpl)
	if err != nil {
		panic(err)
	}
	return mt
}

func MustHuman(tmpl string) MessageTemplate {
	mt, err := NewHumanMessageTemplate(tmpl)
	if err != nil {
		panic(err)
	}
	return mt
}

func MustAI(tmpl string) MessageTemplate {
	mt, err := NewAIMessageTemplate(tmpl)
	if err != nil {
		panic(err)
	}
	return mt
}

// ---------------------------------------------------------------------------
// MessagePlaceholder — injects a slice of messages from vars
// ---------------------------------------------------------------------------

// MessagePlaceholder marks a position in a ChatPromptTemplate where a slice
// of pre-built messages (e.g. conversation history) will be inserted.
//
//	chat := prompt.NewChatPromptTemplate(
//	    prompt.MustSystem("You are a helpful assistant."),
//	    prompt.NewMessagePlaceholder("history"),
//	    prompt.MustHuman("{{.Question}}"),
//	)
type MessagePlaceholder struct {
	VariableName string
}

// NewMessagePlaceholder constructs a MessagePlaceholder with the given variable name.
func NewMessagePlaceholder(variableName string) MessagePlaceholder {
	return MessagePlaceholder{VariableName: variableName}
}

// ---------------------------------------------------------------------------
// ChatPromptTemplate — ordered list of MessageTemplate / MessagePlaceholder
// ---------------------------------------------------------------------------

// slot is either a MessageTemplate or a MessagePlaceholder.
type slot struct {
	msg         *MessageTemplate
	placeholder *MessagePlaceholder
}

// ChatPromptTemplate builds a []schema.Message from an ordered list of
// MessageTemplates and MessagePlaceholders.
//
//	chat, err := prompt.NewChatPromptTemplate(
//	    prompt.MustSystem("You are {{.Persona}}."),
//	    prompt.NewMessagePlaceholder("history"),
//	    prompt.MustHuman("{{.Question}}"),
//	)
//	msgs, err := chat.FormatMessages(map[string]any{
//	    "Persona":  "an expert Go programmer",
//	    "Question": "Explain interfaces.",
//	    "history":  []schema.Message{...},
//	})
type ChatPromptTemplate struct {
	slots []slot
}

// NewChatPromptTemplate accepts any mix of MessageTemplate and MessagePlaceholder
// values (in order). Other types cause an error at construction time.
func NewChatPromptTemplate(parts ...any) (*ChatPromptTemplate, error) {
	cpt := &ChatPromptTemplate{}
	for i, p := range parts {
		switch v := p.(type) {
		case MessageTemplate:
			mt := v
			cpt.slots = append(cpt.slots, slot{msg: &mt})
		case MessagePlaceholder:
			mp := v
			cpt.slots = append(cpt.slots, slot{placeholder: &mp})
		default:
			return nil, fmt.Errorf("prompt: ChatPromptTemplate slot %d has unsupported type %T", i, p)
		}
	}
	return cpt, nil
}

// MustNewChatPromptTemplate is like NewChatPromptTemplate but panics on error.
func MustNewChatPromptTemplate(parts ...any) *ChatPromptTemplate {
	cpt, err := NewChatPromptTemplate(parts...)
	if err != nil {
		panic(err)
	}
	return cpt
}

// FormatMessages renders all slots and returns the resulting message slice.
// vars["history"] must be []schema.Message when a MessagePlaceholder is present.
func (c *ChatPromptTemplate) FormatMessages(vars map[string]any) ([]schema.Message, error) {
	var out []schema.Message

	for _, s := range c.slots {
		if s.placeholder != nil {
			raw, ok := vars[s.placeholder.VariableName]
			if !ok {
				// Missing placeholder → skip (allows optional history).
				continue
			}
			msgs, ok := raw.([]schema.Message)
			if !ok {
				return nil, fmt.Errorf("prompt: placeholder %q: expected []schema.Message, got %T",
					s.placeholder.VariableName, raw)
			}
			out = append(out, msgs...)
			continue
		}

		text, err := s.msg.Template.Format(vars)
		if err != nil {
			return nil, fmt.Errorf("prompt: format message (role=%s): %w", s.msg.Role, err)
		}
		out = append(out, schema.Message{Role: s.msg.Role, Content: text})
	}

	return out, nil
}

// ---------------------------------------------------------------------------
// FewShotPromptTemplate — builds a prompt from examples + a suffix template
// ---------------------------------------------------------------------------

// Example is a single input/output pair used in few-shot prompting.
type Example struct {
	Input  string
	Output string
}

// FewShotPromptTemplate formats a set of examples followed by a query.
type FewShotPromptTemplate struct {
	Prefix         string
	ExamplePrefix  string // e.g. "Input:"
	ExampleSuffix  string // e.g. "Output:"
	Examples       []Example
	SuffixTemplate *PromptTemplate
}

// Format renders the full few-shot prompt.
func (f *FewShotPromptTemplate) Format(vars map[string]any) (string, error) {
	var sb strings.Builder
	if f.Prefix != "" {
		sb.WriteString(f.Prefix)
		sb.WriteString("\n\n")
	}
	for _, ex := range f.Examples {
		sb.WriteString(f.ExamplePrefix)
		sb.WriteString(" ")
		sb.WriteString(ex.Input)
		sb.WriteString("\n")
		sb.WriteString(f.ExampleSuffix)
		sb.WriteString(" ")
		sb.WriteString(ex.Output)
		sb.WriteString("\n\n")
	}
	suffix, err := f.SuffixTemplate.Format(vars)
	if err != nil {
		return "", fmt.Errorf("prompt: few-shot suffix: %w", err)
	}
	sb.WriteString(suffix)
	return sb.String(), nil
}
