// Package gemini provides a Google Gemini LLM implementation for golangchain.
//
// # Usage
//
//	model, err := gemini.New(ctx,
//	    gemini.WithAPIKey(os.Getenv("GEMINI_API_KEY")),
//	    gemini.WithModel("gemini-2.0-flash"),
//	)
//
// Note: unlike other providers, New requires a context.Context as its first
// argument because the Gemini SDK establishes a connection at construction time.
//
// Uses the official Google Gen AI Go SDK (google.golang.org/genai).
package gemini
