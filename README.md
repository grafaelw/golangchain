# golangchain

A production-grade **LangChain + LangGraph equivalent library for Go** —
chains, agents, memory, vector stores, RAG, state graphs, tracing, and
evaluation — all in idiomatic Go with no code generation or reflection.

## Getting started

### Prerequisites

- **Go 1.21+** (generics)
- An **OpenAI API key** or **Azure AI Foundry** account

### Install

```bash
go get github.com/grafaelw/golangchain
```

### Minimal example (OpenAI)

Create `main.go`:

```go
package main

import (
    "context"
    "fmt"
    "os"

    "github.com/grafaelw/golangchain/chain"
    "github.com/grafaelw/golangchain/llm/openai"
    "github.com/grafaelw/golangchain/output"
    "github.com/grafaelw/golangchain/prompt"
)

func main() {
    ctx := context.Background()

    model, _ := openai.New(
        openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
        openai.WithModel("gpt-4o-mini"),
    )

    tmpl := prompt.MustNewChatPromptTemplate(
        prompt.MustSystem("You are a helpful assistant."),
        prompt.MustHuman("{{.question}}"),
    )

    c := chain.NewLLMChain(tmpl, model, output.AsAny(output.StrOutputParser{}))

    answer, _ := c.Invoke(ctx, map[string]any{"question": "What is Go?"})
    fmt.Println(answer)
}
```

```bash
OPENAI_API_KEY=sk-... go run main.go
```

### Azure AI Foundry

Same code — just swap the model setup:

```go
model, _ := openai.New(
    openai.WithAPIKey(os.Getenv("AZURE_OPENAI_API_KEY")),
    openai.WithModel("gpt-4o-mini"),
    openai.WithBaseURL("https://<resource>.services.ai.azure.com/openai/v1/"),
)
```

### What's included

| Area | Packages |
|---|---|
| Core | `schema`, `prompt`, `output`, `chain`, `callbacks`, `tracing` |
| LLM providers | `llm/{openai,azure,anthropic,gemini,ollama,openaicompat}` — all with native tool calling |
| LLM production wrappers | `llmutil` — cache, retry with backoff, rate limiter |
| Agents | `agent` — ReAct, ToolCalling, PlanAndExecute, A2A, MCP |
| RAG stack | `documentloader` (+ pdf, docx), `textsplitter`, `embeddings` (+ ollama), `vectorstore` (in-memory, file, qdrant, chroma, pinecone, pgvector, azure aisearch), `retriever` (BM25, ensemble/RRF, multi-query, contextual compression) |
| Chains | LLM, Sequential, Map, Router, RetrievalQA, MapReduce/Refine summarizers, Batch |
| Memory | Buffer, Window, Summary, File (persistent), VectorStore (semantic) |
| Graph | Generic `StateGraph[S]` with cycles, conditional/parallel edges, interrupts, MemoryCheckpointer and FileCheckpointer |
| Evaluation | `eval` — Dataset (JSONL), Runner, evaluators: ExactMatch, Contains, Regex, JSONEqual, LLMAsJudge |
| Serving | `serve` — LangServe-style HTTP handlers for any Runnable, AgentExecutor, or CompiledGraph (JSON + SSE) |
| Observability | `tracing` — in-memory tree, pretty terminal, JSON-lines exporter, FeedbackStore, OpenTelemetry adapter |

## Packages

| Package | Purpose |
|---|---|
| `schema` | Shared types: `Message`, `Document`, `ToolCall`, `Generation`, `StreamChunk` |
| `llm` | `LLM` interface + functional call options |
| `llm/openai` | OpenAI Chat Completions — also works with Azure AI Foundry via `WithBaseURL` |
| `llm/azure` | Azure OpenAI Service (Azure SDK, separate from `llm/openai`) |
| `llm/anthropic` | Anthropic Claude — **native tool_use support** |
| `llm/gemini` | Google Gemini — **native function-calling** |
| `llm/ollama` | Local LLMs via Ollama — OpenAI-compatible tool calling |
| `llm/openaicompat` | OpenAI-compatible servers (vLLM, LM Studio, llama.cpp) |
| `llm/cohere` | Cohere chat models |
| `llm/huggingface` | Hugging Face inference endpoints |
| `llmutil` | `CachingLLM`, `RetryingLLM`, `RateLimitedLLM` |
| `prompt` | `PromptTemplate`, `ChatPromptTemplate`, `FewShotPromptTemplate`, `MessagePlaceholder` |
| `output` | Parsers: `StrOutputParser`, `JSONOutputParser`, `StructOutputParser[T]`, `ListOutputParser`, `XMLOutputParser`, `RetryWithErrorOutputParser` |
| `chain` | `Runnable`, `LLMChain`, `SequentialChain`, `MapChain`, `RouterChain`, `RetrievalQAChain`, `MapReduceSummarizer`, `RefineSummarizer`, `Batch` |
| `memory` | `ConversationBufferMemory`, `ConversationWindowMemory`, `ConversationSummaryMemory`, `TokenBufferMemory`, `ConversationEntityMemory`, `ConversationKGMemory`, `CombinedMemory` |
| `tools` | `Calculator`, `HTTPFetch`, `DuckDuckGoSearch`, `ShellTool`, `FuncTool` |
| `agent` | `ReActAgent`, `ToolCallingAgent`, `PlanAndExecuteAgent`, `AgentExecutor`, `A2A`, `MCP`, `Sandbox` |
| `graph` | `StateGraph[S]` — nodes, edges, conditional routing, interrupts, checkpoints |
| `callbacks` | `Handler` interface (14 lifecycle methods), `CallbackManager`, `LoggingHandler` |
| `tracing` | In-memory `Tracer`, `PrettyHandler`, `JSONLinesExporter`, `FeedbackStore` |
| `tracing/otel` | OpenTelemetry adapter — maps lifecycle events to OTel spans |
| `embeddings` | `OpenAIEmbedder`, `AzureEmbedder` |
| `embeddings/ollama` | Local embeddings via Ollama |
| `textsplitter` | `CharacterSplitter`, `RecursiveCharacterSplitter`, `MarkdownSplitter` |
| `documentloader` | `TextLoader`, `MarkdownLoader`, `CSVLoader`, `HTMLLoader`, `HTTPLoader`, `DirectoryLoader` (+ pdf, docx sub-packages) |
| `vectorstore` | `VectorStore` interface + `InMemoryVectorStore`, `FileVectorStore` (+ qdrant, chroma, pinecone, pgvector, azureaisearch backends) |
| `retriever` | `VectorStoreRetriever`, `BM25Retriever`, `EnsembleRetriever`, `MultiQueryRetriever`, `ContextualCompressionRetriever` |
| `eval` | `Dataset`, `Runner`, evaluators: `ExactMatch`, `Contains`, `Regex`, `JSONEqual`, `LLMAsJudge` |
| `serve` | LangServe-style HTTP handlers for Runnables, Agents, Graphs (JSON + SSE) |
| `middleware` | Agent middleware: `Retry`, `Guardrails`, `HITL`, `Summarization`, `ToDoList` |

## Examples

| Group | Example | Description |
|---|---|---|
| **Basics** | [`basics/simple_chain`](examples/basics/simple_chain/main.go) | LCEL pipeline + memory + streaming |
| | [`basics/streaming`](examples/basics/streaming/main.go) | Unified `StreamChunk` and `Runnable.Stream()` piping |
| | [`basics/batch`](examples/basics/batch/main.go) | `Runnable.Batch()` — concurrent translation + comparison |
| **Chains** | [`chains/composition`](examples/chains/composition/main.go) | FuncRunnable, Sequential/Map/Router chains + output parsers |
| | [`chains/sql`](examples/chains/sql/main.go) | Natural language → SQL → answer via SQLDatabaseChain |
| **RAG** | [`rag/inmemory`](examples/rag/inmemory/main.go) | Embeddings + InMemoryVectorStore + RetrieverTool |
| | [`rag/document_loaders`](examples/rag/document_loaders/main.go) | PDF, DOCX, Text, Directory → splitter → RAG QA |
| | [`rag/code_splitters`](examples/rag/code_splitters/main.go) | Language-aware code splitters for Python/JS/Go |
| | [`rag/qdrant`](examples/rag/qdrant/main.go) | Qdrant vector DB: collection, indexing, search, delete |
| | [`rag/azure_ai_search`](examples/rag/azure_ai_search/main.go) | Azure AI Search: managed vector store at scale |
| | [`rag/ollama`](examples/rag/ollama/main.go) | Fully-local RAG: Ollama LLM + embeddings + vector store |
| | [`rag/full`](examples/rag/full/main.go) | End-to-end: loader → splitter → hybrid retriever → QA + eval |
| **Agents** | [`agents/react`](examples/agents/react/main.go) | Tool-calling agent with streaming events |
| | [`agents/anthropic`](examples/agents/anthropic/main.go) | ToolCallingAgent with Anthropic native `tool_use` |
| | [`agents/gemini`](examples/agents/gemini/main.go) | ToolCallingAgent with Gemini native function-calling |
| | [`agents/tools_and_memory`](examples/agents/tools_and_memory/main.go) | Memory types + Calculator/FuncTool |
| **Graph** | [`graph/state_graph`](examples/graph/state_graph/main.go) | StateGraph with checkpointing + human-in-the-loop |
| **Observability** | [`observability/tracing`](examples/observability/tracing/main.go) | LangSmith-style tracing with PrettyHandler |
| | [`observability/otel`](examples/observability/otel/main.go) | OpenTelemetry adapter (span hierarchy, attributes) |
| | [`observability/cost`](examples/observability/cost/main.go) | Token cost estimation + ModelPricing lookup |
| | [`observability/cleanup`](examples/observability/cleanup/main.go) | `io.Closer` patterns for resource-holding types |
| | [`observability/errors`](examples/observability/errors/main.go) | Sentinel errors with `errors.Is()` recovery |
| **Output** | [`output/parsers`](examples/output/parsers/main.go) | RetryWithErrorOutputParser + XMLOutputParser |
| **Memory** | [`memory/persistence`](examples/memory/persistence/main.go) | MessagesMemory interface for tool-call persistence |
| **Multimodal** | [`multimodal/vision`](examples/multimodal/vision/main.go) | ContentPart for images + text with GPT-4V / Claude 3 |

All examples use **Azure AI Foundry** by default. Every file contains a
clearly marked comment block showing the two-line swap for the **OpenAI API**.

**Azure AI Foundry** (most examples):

```bash
# .env
AZURE_OPENAI_API_KEY=<your-key>

go run ./examples/basics/simple_chain
go run ./examples/chains/composition
go run ./examples/agents/tools_and_memory
go run ./examples/agents/react
go run ./examples/graph/state_graph
go run ./examples/observability/tracing
```

**OpenAI API** — same commands; just change the model block in the file:

```go
// Replace the "2. Create the LLM" block in any example with:
model, err := openai.New(
    openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
    openai.WithModel("gpt-4o-mini"),
)
// and set OPENAI_API_KEY=sk-... in your .env
```

**Embeddings examples** (rag/*) need additional env vars:

```bash
# Azure embeddings (.env)
AZURE_OPENAI_API_KEY=<azure-api-key>
AZURE_OPENAI_ENDPOINT=https://<resource>.cognitiveservices.azure.com
OPENAI_EMBEDDING_DEPLOYMENT=<deployment-name>
OPENAI_API_VERSION=2024-02-01

# OpenAI embeddings alternative — replace the embedder block with:
# embedder, err := embeddings.NewOpenAIEmbedder(os.Getenv("OPENAI_API_KEY"), "text-embedding-3-small")
```

## License

MIT
