package retriever

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// SelfQueryRetriever
// ---------------------------------------------------------------------------

// SelfQueryRetriever uses an LLM to extract metadata filters from a natural
// language query, then applies those filters to a vector store search.
type SelfQueryRetriever struct {
	Retriever      Retriever
	LLM            llm.LLM
	MetadataFields []MetadataFieldInfo
}

// MetadataFieldInfo describes a metadata field the LLM can filter on.
type MetadataFieldInfo struct {
	Name        string
	Description string
	Type        string
	Values      []string
}

// NewSelfQueryRetriever creates a self-query retriever.
func NewSelfQueryRetriever(base Retriever, model llm.LLM, fields []MetadataFieldInfo) *SelfQueryRetriever {
	return &SelfQueryRetriever{
		Retriever:      base,
		LLM:            model,
		MetadataFields: fields,
	}
}

const selfQueryPrompt = `Given the user query, extract metadata filters from the available fields.
Available fields:
{{ .fields }}

Query: {{ .query }}

Return a JSON object with a "filter" key containing an object with field->condition mappings.
Example: {"filter": {"year": {"$gte": 2020}, "author": {"$eq": "Smith"}}}`

// GetRelevantDocuments extracts metadata filters from the query using the LLM,
// applies them to the base retriever's results, and returns filtered documents.
func (s *SelfQueryRetriever) GetRelevantDocuments(ctx context.Context, query string) ([]schema.Document, error) {
	prompt := buildSelfQueryPrompt(s.MetadataFields, query)

	gen, err := s.LLM.Generate(ctx, []schema.Message{schema.NewHumanMessage(prompt)})
	if err != nil {
		return nil, fmt.Errorf("retriever: selfquery: llm: %w", err)
	}

	filter, err := parseFilter(gen.Text)
	if err != nil {
		return nil, fmt.Errorf("retriever: selfquery: parse filter: %w", err)
	}

	docs, err := s.Retriever.GetRelevantDocuments(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("retriever: selfquery: base: %w", err)
	}

	if len(filter) == 0 {
		return docs, nil
	}

	out := docs[:0:0]
	for _, d := range docs {
		if matchMetadata(d.Metadata, filter) {
			out = append(out, d)
		}
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// prompt building
// ---------------------------------------------------------------------------

func buildSelfQueryPrompt(fields []MetadataFieldInfo, query string) string {
	var b strings.Builder
	for _, f := range fields {
		fmt.Fprintf(&b, "- %s (%s): %s", f.Name, f.Type, f.Description)
		if len(f.Values) > 0 {
			fmt.Fprintf(&b, " [values: %s]", strings.Join(f.Values, ", "))
		}
		b.WriteByte('\n')
	}
	p := strings.ReplaceAll(selfQueryPrompt, "{{ .fields }}", b.String())
	p = strings.ReplaceAll(p, "{{ .query }}", query)
	return p
}

// ---------------------------------------------------------------------------
// filter parsing
// ---------------------------------------------------------------------------

// filter is a map of field name to operator->value pairs.
type filter map[string]map[string]any

func parseFilter(text string) (filter, error) {
	js := extractJSON(text)
	if js == "" {
		// LLM returned no JSON; treat as empty filter.
		return nil, nil
	}
	var wrapper struct {
		Filter filter `json:"filter"`
	}
	if err := json.Unmarshal([]byte(js), &wrapper); err != nil {
		return nil, err
	}
	return wrapper.Filter, nil
}

func extractJSON(text string) string {
	text = strings.TrimSpace(text)
	if idx := strings.Index(text, "```json"); idx >= 0 {
		rest := text[idx+7:]
		if end := strings.Index(rest, "```"); end >= 0 {
			return strings.TrimSpace(rest[:end])
		}
	}
	if idx := strings.Index(text, "```"); idx >= 0 {
		rest := text[idx+3:]
		if end := strings.Index(rest, "```"); end >= 0 {
			return strings.TrimSpace(rest[:end])
		}
	}
	if idx := strings.Index(text, "{"); idx >= 0 {
		return text[idx:]
	}
	return ""
}

// ---------------------------------------------------------------------------
// metadata matching
// ---------------------------------------------------------------------------

func matchMetadata(meta map[string]any, f filter) bool {
	for field, conditions := range f {
		val, ok := meta[field]
		if !ok {
			return false
		}
		for op, target := range conditions {
			if !matchOp(val, op, target) {
				return false
			}
		}
	}
	return true
}

func matchOp(val any, op string, target any) bool {
	switch op {
	case "$eq":
		return cmpEqual(val, target)
	case "$contains":
		return cmpContains(val, target)
	case "$gt":
		return cmpNum(val, target) > 0
	case "$lt":
		return cmpNum(val, target) < 0
	case "$gte":
		return cmpNum(val, target) >= 0
	case "$lte":
		return cmpNum(val, target) <= 0
	default:
		return false
	}
}

func cmpEqual(a, b any) bool {
	if as, ok := a.(string); ok {
		if bs, ok := b.(string); ok {
			return as == bs
		}
		return false
	}
	af, aok := toFloat(a)
	bf, bok := toFloat(b)
	if aok && bok {
		return af == bf
	}
	return fmt.Sprint(a) == fmt.Sprint(b)
}

func cmpContains(a, b any) bool {
	as := fmt.Sprint(a)
	bs := fmt.Sprint(b)
	return strings.Contains(as, bs)
}

func cmpNum(a, b any) int {
	af, aok := toFloat(a)
	bf, bok := toFloat(b)
	if !aok || !bok {
		return 0
	}
	if af > bf {
		return 1
	}
	if af < bf {
		return -1
	}
	return 0
}

func toFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case int32:
		return float64(n), true
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	default:
		return 0, false
	}
}
