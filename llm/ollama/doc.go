// Package ollama provides a local Ollama server LLM implementation for golangchain.
//
// Ollama (https://ollama.com) runs open-weight models locally. This package
// communicates with it via its OpenAI-compatible /v1 endpoint.
//
// # Usage
//
//	model, err := ollama.New(
//	    ollama.WithModel("llama3.2"),
//	    ollama.WithBaseURL("http://localhost:11434"), // default
//	)
//
// No API key is required. Start Ollama with: ollama serve
package ollama
