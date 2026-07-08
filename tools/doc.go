// Package tools defines the Tool interface and provides built-in tool
// implementations for use with golangchain agents.
//
// # Tool interface
//
//	type Tool interface {
//	    Name()        string
//	    Description() string
//	    Schema()      json.RawMessage  // JSON Schema of the input object
//	    Run(ctx context.Context, input string) (string, error)
//	}
//
// # Built-in tools
//
//   - [Calculator]       — evaluates arithmetic expressions using a recursive-descent parser;
//     supports +, -, *, /, ^ and functions sqrt, abs, floor, ceil, round
//   - [HTTPFetch]        — HTTP GET a URL, returns the response body as a string
//   - [DuckDuckGoSearch] — queries the DuckDuckGo Instant Answer API (no API key required)
//   - [ShellTool]        — executes shell commands (use in sandboxed environments only)
//   - [FuncTool]         — wraps any Go function as a Tool with zero boilerplate
//
// # Helper functions
//
//   - [ToToolDef]  — converts a Tool to schema.ToolDef for passing to LLM providers
//   - [ToToolDefs] — batch conversion of a []Tool slice
//   - [FindTool]   — looks up a tool by name in a slice
//
// # Example: custom tool
//
//	myTool := tools.NewFuncTool(
//	    "word_count",
//	    "Counts the number of words in the input.",
//	    nil, // schema is optional
//	    func(ctx context.Context, input string) (string, error) {
//	        return fmt.Sprintf("%d", len(strings.Fields(input))), nil
//	    },
//	)
package tools
