// Example: cost_tracking
//
// Demonstrates the cost tracking and estimation features added to
// schema.TokenUsage and schema.Generate, plus the built-in ModelPricing
// lookup table covering 20+ models from OpenAI, Anthropic, and Google.
//
// Highlights:
//
//   - TokenUsage.EstimateCost(model) returns the USD cost.
//
//   - ModelPricing is a map of known model → CostPerToken (prompt + completion).
//
//   - Generation.EstimatedCost is auto-populated by LLM providers.
//
//   - CostPerToken.Multiply(prompt, completion) does the math.
//
//   - Custom models can be added to the pricing map at runtime.
//
//     Run this example with:
//     go run ./examples/cost_tracking
package main

import (
	"fmt"

	"github.com/grafaelw/golangchain/schema"
)

func main() {
	// -------------------------------------------------------------------------
	// 1. Basic cost estimation
	// -------------------------------------------------------------------------
	fmt.Println("--- 1. Cost estimation ---")
	usage := schema.TokenUsage{
		PromptTokens:     1500,
		CompletionTokens: 500,
		TotalTokens:      2000,
	}
	fmt.Printf("  %s\n", usage.String())
	fmt.Printf("  GPT-4o cost:    $%.6f\n", usage.EstimateCost("gpt-4o"))
	fmt.Printf("  GPT-4o-mini:    $%.6f\n", usage.EstimateCost("gpt-4o-mini"))
	fmt.Printf("  GPT-4:          $%.6f\n", usage.EstimateCost("gpt-4"))
	fmt.Printf("  Unknown model:  $%.4f (no pricing data)\n", usage.EstimateCost("my-custom-model"))

	// -------------------------------------------------------------------------
	// 2. Realistic usage comparison: same tokens, different models
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 2. Cross-model cost comparison ---")
	realUsage := schema.TokenUsage{
		PromptTokens:     2000,
		CompletionTokens: 800,
		TotalTokens:      2800,
	}
	models := []struct {
		name string
		desc string
	}{
		{"gpt-4o", "OpenAI GPT-4o"},
		{"gpt-4o-mini", "OpenAI GPT-4o Mini"},
		{"claude-3-5-sonnet", "Anthropic Claude 3.5 Sonnet"},
		{"claude-3-haiku", "Anthropic Claude 3 Haiku"},
		{"gemini-2.5-flash", "Google Gemini 2.5 Flash"},
		{"gemini-1.5-pro", "Google Gemini 1.5 Pro"},
	}
	fmt.Printf("  %s\n", realUsage.String())
	for _, m := range models {
		cost := realUsage.EstimateCost(m.name)
		fmt.Printf("  %-25s $%9.6f\n", m.desc, cost)
	}

	// -------------------------------------------------------------------------
	// 3. Add custom model pricing at runtime
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 3. Custom model pricing ---")
	schema.ModelPricing["my-llama-model"] = schema.CostPerToken{
		PromptPrice:     0.0001,
		CompletionPrice: 0.0002,
	}
	cost := usage.EstimateCost("my-llama-model")
	fmt.Printf("  Custom model cost: $%.6f\n", cost)

	// -------------------------------------------------------------------------
	// 4. Simulation: Agent run cost over multiple LLM calls
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 4. Simulated agent run cost ---")
	type call struct {
		model string
		usage schema.TokenUsage
	}
	agentCalls := []call{
		{"gpt-4o", schema.TokenUsage{PromptTokens: 1500, CompletionTokens: 200, TotalTokens: 1700}},
		{"gpt-4o", schema.TokenUsage{PromptTokens: 2000, CompletionTokens: 400, TotalTokens: 2400}},
		{"gpt-4o-mini", schema.TokenUsage{PromptTokens: 800, CompletionTokens: 100, TotalTokens: 900}},
		{"gpt-4o", schema.TokenUsage{PromptTokens: 1200, CompletionTokens: 300, TotalTokens: 1500}},
	}
	var totalCost float64
	var totalPrompt, totalComplete int
	for i, c := range agentCalls {
		cost := c.usage.EstimateCost(c.model)
		totalCost += cost
		totalPrompt += c.usage.PromptTokens
		totalComplete += c.usage.CompletionTokens
		fmt.Printf("  Call %d (%s): %s → $%.6f\n", i+1, c.model, c.usage.String(), cost)
	}
	fmt.Printf("  ─────────────────────────────────────\n")
	fmt.Printf("  Total: %d calls, ↑%d↓%d tok, $%.6f\n",
		4, totalPrompt, totalComplete, totalCost)

	// -------------------------------------------------------------------------
	// 5. Generation.EstimatedCost auto-populated by providers
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 5. Generation with EstimatedCost ---")
	gen := &schema.Generation{
		Text: "The capital of Japan is Tokyo.",
		Usage: schema.TokenUsage{
			PromptTokens:     100,
			CompletionTokens: 30,
			TotalTokens:      130,
		},
		EstimatedCost: 0.00055, // set automatically by the OpenAI provider
	}
	fmt.Printf("  %s\n", gen.String())

	fmt.Println("\n✅ Cost tracking examples complete.")
}
