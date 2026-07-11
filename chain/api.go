package chain

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/schema"
)

// APIDescription describes a REST API endpoint for the APIChain to call.
type APIDescription struct {
	// Name is the human-readable name of this API (e.g. "weather").
	Name string
	// URL is the endpoint URL template, e.g. "https://api.weather.com/v1/current?city={{ .city }}".
	URL string
	// Method is the HTTP method (GET, POST, etc.). Defaults to GET.
	Method string
	// Headers are additional headers to include (e.g. authorization).
	Headers map[string]string
	// BodyTemplate is an optional JSON body template for POST requests.
	BodyTemplate string
}

// APIChain converts natural language questions into REST API calls.
// The LLM extracts parameters from the query, fills the URL template,
// executes the HTTP call, and then summarises the response in one step.
//
//	apiDesc := chain.APIDescription{
//	    Name: "weather",
//	    URL:  "https://api.weather.com/v1/current?city={{ .city }}",
//	}
//	chain := chain.NewAPIChain(model, apiDesc)
//	ans, _ := chain.Invoke(ctx, "What's the weather in Berlin?")
type APIChain struct {
	LLM        llm.LLM
	LLMOptions []llm.Option
	API        APIDescription
	HTTPClient *http.Client
	Name       string
}

// NewAPIChain creates an API chain for the given endpoint.
func NewAPIChain(model llm.LLM, api APIDescription, opts ...llm.Option) *APIChain {
	if api.Method == "" {
		api.Method = "GET"
	}
	return &APIChain{
		LLM:        model,
		LLMOptions: opts,
		API:        api,
		HTTPClient: &http.Client{Timeout: 30 * time.Second},
		Name:       "APIChain:" + api.Name,
	}
}

func (c *APIChain) Invoke(ctx context.Context, input any) (any, error) {
	question := fmt.Sprint(input)

	extracted, err := c.extractParams(ctx, question)
	if err != nil {
		return nil, err
	}

	filledURL := fillTemplate(c.API.URL, extracted)

	body := ""
	if c.API.BodyTemplate != "" {
		body = fillTemplate(c.API.BodyTemplate, extracted)
	}

	apiResp, err := c.callAPI(ctx, filledURL, c.API.Method, c.API.Headers, body)
	if err != nil {
		return nil, fmt.Errorf("%s: API call: %w", c.Name, err)
	}

	summary, err := c.summarise(ctx, question, apiResp)
	if err != nil {
		return nil, err
	}

	return map[string]any{
		"question":     question,
		"api_url":      filledURL,
		"api_method":   c.API.Method,
		"api_response": truncateStr(apiResp, 2000),
		"answer":       summary,
	}, nil
}

func (c *APIChain) extractParams(ctx context.Context, question string) (map[string]string, error) {
	prompt := fmt.Sprintf(`Extract key-value parameters from the question below that are needed to call the "%s" API. The API endpoint is: %s

Output ONLY valid JSON as a flat object of string keys and string values. If no parameters are needed, output {}.

Question: %s

JSON:`, c.API.Name, c.API.URL, question)

	gen, err := c.LLM.Generate(ctx, []schema.Message{schema.NewHumanMessage(prompt)}, c.LLMOptions...)
	if err != nil {
		return nil, fmt.Errorf("%s: extract: %w", c.Name, err)
	}

	raw := stripCodeFence(gen.Text)
	params := make(map[string]string)
	// Manual simple JSON parser for robustness
	c.parseSimpleJSON(raw, params)
	return params, nil
}

func (c *APIChain) callAPI(ctx context.Context, url, method string, headers map[string]string, body string) (string, error) {
	var bodyReader io.Reader
	if body != "" {
		bodyReader = strings.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		return "", err
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	if body != "" && req.Header.Get("Content-Type") == "" {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return "", err
	}
	defer func() { _ = resp.Body.Close() }()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 100*1024))
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func (c *APIChain) summarise(ctx context.Context, question, apiResp string) (string, error) {
	prompt := fmt.Sprintf(`Given the question and the API response below, provide a concise natural language answer.

Question: %s

API response: %s

Answer:`, question, truncateStr(apiResp, 3000))

	gen, err := c.LLM.Generate(ctx, []schema.Message{schema.NewHumanMessage(prompt)}, c.LLMOptions...)
	if err != nil {
		return "", fmt.Errorf("%s: summarise: %w", c.Name, err)
	}
	return strings.TrimSpace(gen.Text), nil
}

func (c *APIChain) parseSimpleJSON(raw string, dest map[string]string) {
	inString := false
	escaped := false
	var currentKey, currentValue string
	var sb strings.Builder
	var isKey bool

	for i := 0; i < len(raw); i++ {
		ch := raw[i]
		if escaped {
			sb.WriteByte(ch)
			escaped = false
			continue
		}
		if ch == '\\' {
			sb.WriteByte(ch)
			escaped = true
			continue
		}
		if ch == '"' {
			if !inString {
				inString = true
				sb.Reset()
				continue
			}
			inString = false
			if !isKey {
				// Just finished a key
				currentKey = sb.String()
				isKey = true
				sb.Reset()
			} else {
				// Just finished a value
				currentValue = sb.String()
				if currentKey != "" {
					dest[currentKey] = currentValue
					currentKey = ""
				}
				isKey = false
				sb.Reset()
			}
			continue
		}
		if !inString {
			if ch == ':' {
				continue // separator
			}
			if ch == ',' || ch == '}' {
				continue
			}
			continue
		}
		sb.WriteByte(ch)
	}
}

func (c *APIChain) Stream(ctx context.Context, input any) (<-chan schema.StreamChunk, error) {
	out, err := c.Invoke(ctx, input)
	ch := make(chan schema.StreamChunk, 1)
	if err != nil {
		ch <- schema.StreamChunk{Err: err}
		close(ch)
		return ch, nil
	}
	ch <- schema.StreamChunk{Value: out, Done: true}
	close(ch)
	return ch, nil
}

func (c *APIChain) Pipe(next Runnable) Runnable { return &pipeRunnable{first: c, second: next} }
func (c *APIChain) Batch(ctx context.Context, inputs []any) ([]any, error) {
	return RunBatch(ctx, c, inputs)
}

func fillTemplate(tmpl string, vars map[string]string) string {
	result := tmpl
	for k, v := range vars {
		result = strings.ReplaceAll(result, "{{ ."+k+" }}", v)
	}
	return result
}

func truncateStr(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func stripCodeFence(s string) string {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "```json")
	s = strings.TrimPrefix(s, "```")
	s = strings.TrimSuffix(s, "```")
	return strings.TrimSpace(s)
}
