// Package openaicompat provides an LLM implementation for any server that
// speaks the OpenAI Chat Completions API schema.
//
// Compatible servers include vLLM, LM Studio, LocalAI, llama.cpp server,
// Jan, and many others.
//
// # Usage
//
//	model, err := openaicompat.New(
//	    openaicompat.WithBaseURL("http://localhost:1234/v1"),
//	    openaicompat.WithModel("mistral-7b-instruct"),
//	    openaicompat.WithAPIKey("not-needed"), // some servers require a non-empty value
//	)
package openaicompat
