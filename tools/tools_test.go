package tools_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/grafaelw/golangchain/tools"
)

// ---------------------------------------------------------------------------
// Calculator
// ---------------------------------------------------------------------------

func TestCalculator_Name(t *testing.T) {
	c := tools.Calculator{}
	if c.Name() != "calculator" {
		t.Errorf("Name: want %q got %q", "calculator", c.Name())
	}
}

func TestCalculator_Description(t *testing.T) {
	c := tools.Calculator{}
	if c.Description() == "" {
		t.Error("Description should not be empty")
	}
}

func TestCalculator_Schema(t *testing.T) {
	c := tools.Calculator{}
	var schema map[string]any
	if err := json.Unmarshal(c.Schema(), &schema); err != nil {
		t.Fatalf("Schema is not valid JSON: %v", err)
	}
}

func TestCalculator_BasicArithmetic(t *testing.T) {
	ctx := context.Background()
	c := tools.Calculator{}

	tests := []struct {
		input string
		want  string
	}{
		{"2 + 2", "4"},
		{"10 - 3", "7"},
		{"3 * 4", "12"},
		{"10 / 4", "2.5"},
		{"2 ^ 10", "1024"},
		{"(2 + 3) * 4", "20"},
		{"sqrt(144)", "12"},
		{"abs(-7)", "7"},
		{"floor(3.9)", "3"},
		{"ceil(3.1)", "4"},
		{"round(3.5)", "4"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got, err := c.Run(ctx, tt.input)
			if err != nil {
				t.Fatalf("Run(%q): %v", tt.input, err)
			}
			if got != tt.want {
				t.Errorf("Run(%q): want %q got %q", tt.input, tt.want, got)
			}
		})
	}
}

func TestCalculator_JSONInput(t *testing.T) {
	ctx := context.Background()
	c := tools.Calculator{}
	got, err := c.Run(ctx, `{"expression":"3 + 3"}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "6" {
		t.Errorf("want %q got %q", "6", got)
	}
}

func TestCalculator_DivisionByZero(t *testing.T) {
	c := tools.Calculator{}
	_, err := c.Run(context.Background(), "1 / 0")
	if err == nil {
		t.Error("expected division-by-zero error")
	}
}

func TestCalculator_SqrtNegative(t *testing.T) {
	c := tools.Calculator{}
	_, err := c.Run(context.Background(), "sqrt(-4)")
	if err == nil {
		t.Error("expected error for sqrt of negative number")
	}
}

func TestCalculator_InvalidExpression(t *testing.T) {
	c := tools.Calculator{}
	_, err := c.Run(context.Background(), "2 +")
	if err == nil {
		t.Error("expected error for incomplete expression")
	}
}

func TestCalculator_UnknownFunction(t *testing.T) {
	c := tools.Calculator{}
	_, err := c.Run(context.Background(), "foo(3)")
	if err == nil {
		t.Error("expected error for unknown function")
	}
}

func TestCalculator_NegativeNumber(t *testing.T) {
	c := tools.Calculator{}
	got, err := c.Run(context.Background(), "-5 + 10")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "5" {
		t.Errorf("want %q got %q", "5", got)
	}
}

func TestCalculator_FloatResult(t *testing.T) {
	c := tools.Calculator{}
	got, err := c.Run(context.Background(), "1 / 3")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Should produce a decimal
	if !strings.Contains(got, ".") {
		t.Errorf("expected decimal result, got %q", got)
	}
}

func TestCalculator_NestedParens(t *testing.T) {
	c := tools.Calculator{}
	got, err := c.Run(context.Background(), "((2 + 3) * (4 - 1)) / 5")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "3" {
		t.Errorf("want %q got %q", "3", got)
	}
}

// ---------------------------------------------------------------------------
// HTTPFetch
// ---------------------------------------------------------------------------

func TestHTTPFetch_Name(t *testing.T) {
	h := tools.NewHTTPFetch()
	if h.Name() != "http_fetch" {
		t.Errorf("Name: want %q got %q", "http_fetch", h.Name())
	}
}

func TestHTTPFetch_Success(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte("hello from test server"))
	}))
	defer ts.Close()

	h := tools.NewHTTPFetch()
	got, err := h.Run(context.Background(), ts.URL)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(got, "hello from test server") {
		t.Errorf("response missing expected body: %q", got)
	}
	if !strings.Contains(got, "HTTP 200") {
		t.Errorf("response missing status line: %q", got)
	}
}

func TestHTTPFetch_JSONInput(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("json test"))
	}))
	defer ts.Close()

	h := tools.NewHTTPFetch()
	input, _ := json.Marshal(map[string]string{"url": ts.URL})
	got, err := h.Run(context.Background(), string(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(got, "json test") {
		t.Errorf("body missing: %q", got)
	}
}

func TestHTTPFetch_InvalidURL(t *testing.T) {
	h := tools.NewHTTPFetch()
	_, err := h.Run(context.Background(), "not-a-url")
	if err == nil {
		t.Error("expected error for invalid URL")
	}
}

func TestHTTPFetch_Non200(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(404)
		w.Write([]byte("not found"))
	}))
	defer ts.Close()

	h := tools.NewHTTPFetch()
	got, err := h.Run(context.Background(), ts.URL)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(got, "HTTP 404") {
		t.Errorf("want HTTP 404 in response, got: %q", got)
	}
}

// ---------------------------------------------------------------------------
// FuncTool
// ---------------------------------------------------------------------------

func TestFuncTool(t *testing.T) {
	ft := tools.NewFuncTool(
		"echo",
		"Echoes input back",
		nil,
		func(ctx context.Context, input string) (string, error) {
			return "echo: " + input, nil
		},
	)

	if ft.Name() != "echo" {
		t.Errorf("Name mismatch")
	}
	if ft.Description() != "Echoes input back" {
		t.Errorf("Description mismatch")
	}
	if ft.Schema() == nil {
		t.Error("Schema should not be nil (default schema applied)")
	}

	got, err := ft.Run(context.Background(), "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "echo: hello" {
		t.Errorf("want %q got %q", "echo: hello", got)
	}
}

func TestFuncTool_WithCustomSchema(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"msg":{"type":"string"}}}`)
	ft := tools.NewFuncTool("custom", "desc", schema, func(_ context.Context, input string) (string, error) {
		return input, nil
	})
	if string(ft.Schema()) != string(schema) {
		t.Errorf("custom schema not preserved")
	}
}

// ---------------------------------------------------------------------------
// ToToolDef / ToToolDefs / FindTool
// ---------------------------------------------------------------------------

func TestToToolDef(t *testing.T) {
	c := tools.Calculator{}
	def := tools.ToToolDef(c)
	if def.Name != "calculator" {
		t.Errorf("Name mismatch: %q", def.Name)
	}
	if def.Description == "" {
		t.Error("Description should not be empty")
	}
}

func TestToToolDefs(t *testing.T) {
	ts := []tools.Tool{tools.Calculator{}, tools.NewHTTPFetch()}
	defs := tools.ToToolDefs(ts)
	if len(defs) != 2 {
		t.Fatalf("want 2 defs, got %d", len(defs))
	}
}

func TestFindTool_Found(t *testing.T) {
	ts := []tools.Tool{tools.Calculator{}, tools.NewHTTPFetch()}
	found := tools.FindTool(ts, "calculator")
	if found == nil {
		t.Error("expected to find calculator")
	}
	if found.Name() != "calculator" {
		t.Errorf("Name mismatch")
	}
}

func TestFindTool_NotFound(t *testing.T) {
	ts := []tools.Tool{tools.Calculator{}}
	found := tools.FindTool(ts, "nonexistent")
	if found != nil {
		t.Error("expected nil for nonexistent tool")
	}
}
