// Package golangchain is a production-grade LangChain + LangGraph equivalent
// library for Go.
//
// # Architecture
//
// The library is organised into focused, composable packages:
//
//	schema        — shared types (Message, Document, ToolCall, Generation …)
//	llm           — LLM interface + functional options
//	  ├─ openai   — OpenAI Chat Completions
//	  ├─ azure    — Azure OpenAI Service
//	  ├─ anthropic— Anthropic Claude
//	  ├─ gemini   — Google Gemini
//	  ├─ ollama   — Local Ollama server
//	  └─ openaicompat — any OpenAI-schema server (vLLM, LM Studio …)
//	prompt        — PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
//	output        — typed output parsers (Str, JSON, Struct, List, Bool)
//	chain         — Runnable / Pipe composition, LLMChain, SequentialChain …
//	memory        — Buffer, Window, and Summary conversation memories
//	tools         — Tool interface + Calculator, HTTPFetch, DuckDuckGo, Shell, FuncTool
//	agent         — ReActAgent, ToolCallingAgent, AgentExecutor
//	embeddings    — Embedder interface, OpenAI and Azure embedders
//	vectorstore   — VectorStore interface, InMemoryVectorStore (cosine similarity)
//	callbacks     — Handler interface, CallbackManager fan-out, LoggingHandler
//	graph         — StateGraph[S], LangGraph-equivalent engine with checkpointing
//
// # Quick start
//
//	import (
//	    "github.com/grafaelw/golangchain/chain"
//	    "github.com/grafaelw/golangchain/llm/openai"
//	    "github.com/grafaelw/golangchain/output"
//	    "github.com/grafaelw/golangchain/prompt"
//	)
//
//	model, _ := openai.New(openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")))
//	tmpl := prompt.MustNewChatPromptTemplate(
//	    prompt.MustSystem("You are a helpful assistant."),
//	    prompt.MustHuman("{{.question}}"),
//	)
//	c := chain.NewLLMChain(tmpl, model, output.AsAny(output.StrOutputParser{}))
//	answer, _ := c.Invoke(ctx, map[string]any{"question": "What is Go?"})
//
// See the examples/ directory for complete runnable programs.
package golangchain
