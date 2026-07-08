// Package golangchain is a production-grade LangChain + LangGraph equivalent
// library for Go.
//
// # Architecture
//
// The library is organised into focused, composable packages:
//
//	schema         — shared types (Message, Document, ToolCall, Generation …)
//	llm            — LLM interface + functional options
//	  ├─ openai    — OpenAI Chat Completions
//	  ├─ azure     — Azure OpenAI Service
//	  ├─ anthropic — Anthropic Claude
//	  ├─ gemini    — Google Gemini
//	  ├─ ollama    — Local Ollama server
//	  └─ openaicompat — any OpenAI-schema server (vLLM, LM Studio …)
//	llmutil        — LLM wrappers: caching, retry with backoff, rate limiting
//	prompt         — PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
//	output         — typed output parsers (Str, JSON, Struct, List, Bool)
//	chain          — Runnable / Pipe composition; LLMChain, Sequential, Map,
//	                 Router, RetrievalQA, MapReduce/Refine summarizers
//	memory         — Buffer, Window, Summary; FileChatHistory (persistent);
//	                 VectorStoreMemory (semantic long-term memory)
//	tools          — Tool interface + Calculator, HTTPFetch, DuckDuckGo,
//	                 Shell, FuncTool, RetrieverTool
//	agent          — ReActAgent, ToolCallingAgent, PlanAndExecuteAgent,
//	                 AgentExecutor
//	embeddings     — Embedder interface, OpenAI and Azure embedders
//	textsplitter   — CharacterSplitter, RecursiveCharacterSplitter,
//	                 MarkdownSplitter
//	documentloader — Text, Markdown, CSV, HTML, HTTP, and Directory loaders
//	vectorstore    — VectorStore interface; InMemoryVectorStore and
//	                 FileVectorStore (persistent JSON)
//	retriever      — Retriever interface; VectorStore, BM25, Ensemble (RRF),
//	                 MultiQuery, and ContextualCompression retrievers
//	callbacks      — Handler interface, CallbackManager fan-out, LoggingHandler
//	tracing        — In-memory tracer, PrettyHandler, JSONLinesExporter,
//	                 FeedbackStore
//	eval           — Dataset (JSONL), Runner, and evaluators (ExactMatch,
//	                 Contains, Regex, JSONEqual, LLMAsJudge)
//	graph          — StateGraph[S], LangGraph-equivalent engine with
//	                 MemoryCheckpointer and FileCheckpointer
//	serve          — LangServe-style HTTP endpoints (invoke/stream/health)
//	                 for any Runnable, AgentExecutor, or CompiledGraph
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
