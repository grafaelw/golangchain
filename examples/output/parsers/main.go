// Example: output_parsers
//
// Demonstrates the advanced output parsers added to the output package:
//   - RetryWithErrorOutputParser  — auto-corrects parse failures via LLM
//   - XMLOutputParser             — parses LLM output as XML
//
// Run this example with:
//
//	go run ./examples/output/parsers
package main

import (
	"context"
	"fmt"

	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/output"
	"github.com/grafaelw/golangchain/schema"
)

func main() {
	ctx := context.Background()

	// -------------------------------------------------------------------------
	// 1. RetryWithErrorOutputParser — retries with LLM correction
	// -------------------------------------------------------------------------
	fmt.Println("--- 1. RetryWithErrorOutputParser ---")

	// A mock LLM that initially produces bad JSON, then corrects it.
	mockLLM := &mockCorrector{
		responses: []string{
			`{"answer": 42, missing closing brace`, // bad
			`{"answer": 42}`,                       // corrected
		},
	}

	jsonParser := output.JSONOutputParser{}
	retryParser := output.NewRetryWithErrorOutputParser(
		jsonParser,
		mockLLM,
		2, // up to 2 retries
	)

	result, err := retryParser.ParseContext(ctx, `{"answer": 42, missing closing brace`, "Output valid JSON.")
	if err != nil {
		panic(err)
	}
	fmt.Printf("  Corrected result: %v\n", result)

	// -------------------------------------------------------------------------
	// 2. Direct parse (no retry) — falls back to inner parser
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 2. Direct parse (no retry) ---")
	val, err := retryParser.Parse(`{"answer": 42}`)
	if err != nil {
		panic(err)
	}
	fmt.Printf("  Direct parse: %v\n", val)

	// -------------------------------------------------------------------------
	// 3. XMLOutputParser — parse LLM XML output
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 3. XMLOutputParser ---")
	xmlParser := output.XMLOutputParser{}

	xmlDoc, err := xmlParser.Parse(`
<response>
  <answer>Tokyo</answer>
  <country>Japan</country>
  <confidence>0.95</confidence>
</response>`)
	if err != nil {
		panic(err)
	}
	fmt.Printf("  XML result: %v\n", xmlDoc.Data)

	// -------------------------------------------------------------------------
	// 4. XML with nested structure
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 4. XML with nested structure ---")
	nested, err := xmlParser.Parse(`
<report>
  <title>Weather</title>
  <location>
    <city>Paris</city>
    <temp>18</temp>
  </location>
</report>`)
	if err != nil {
		panic(err)
	}
	report := nested.Data["report"].(map[string]any)
	fmt.Printf("  Report title: %v\n", report["title"])
	loc := report["location"].(map[string]any)
	fmt.Printf("  City: %v, Temp: %v°C\n", loc["city"], loc["temp"])

	fmt.Println("\n✅ Output parsers complete.")
}

// mockCorrector implements output.OutputLLM for demo purposes.
type mockCorrector struct {
	responses []string
	idx       int
}

func (m *mockCorrector) Generate(_ context.Context, _ []schema.Message, _ ...llm.Option) (*schema.Generation, error) {
	resp := m.responses[m.idx%len(m.responses)]
	m.idx++
	return &schema.Generation{Text: resp, Message: schema.NewAIMessage(resp)}, nil
}

var _ output.OutputLLM = (*mockCorrector)(nil)
