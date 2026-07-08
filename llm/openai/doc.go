// Package openai provides an OpenAI Chat Completions LLM implementation for golangchain.
//
// # Usage
//
//	model, err := openai.New(
//	    openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
//	    openai.WithModel("gpt-4o-mini"),
//	)
//
// The returned *LLM implements [llm.LLM] and can be used with any golangchain
// chain, agent, or memory component.
package openai
