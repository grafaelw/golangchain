// Package anthropic provides an Anthropic Claude LLM implementation for golangchain.
//
// # Usage
//
//	model, err := anthropic.New(
//	    anthropic.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")),
//	    anthropic.WithModel("claude-sonnet-4-5"),
//	)
//
// Uses the official Anthropic Go SDK (github.com/anthropics/anthropic-sdk-go).
package anthropic
