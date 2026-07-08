# golangchain

A production-grade **LangChain + LangGraph equivalent library for Go**.

Composable LLM chains, tool-using agents, conversation memory, vector stores,
retrievers, document loaders, text splitters, an evaluation harness, LangServe-
like HTTP serving, JSON-lines tracing, and a fully generic StateGraph engine —
all idiomatic Go, no code generation, no reflection magic.

### What's included

| Area | Packages |
|---|---|
| Core | `schema`, `prompt`, `output`, `chain`, `callbacks`, `tracing` |
| LLM providers | `llm/{openai,azure,anthropic,gemini,ollama,openaicompat}` |
| LLM production wrappers | `llmutil` — cache (memory + file), retry with backoff, rate limiter |
| Agents | `agent` — ReAct, ToolCalling, PlanAndExecute |
| RAG stack | `documentloader`, `textsplitter`, `embeddings`, `vectorstore` (in-memory + file), `retriever` (BM25, ensemble/RRF, multi-query, contextual compression) |
| Chains | LLM, Sequential, Map, Router, **RetrievalQA**, **MapReduce/Refine summarizers** |
| Memory | Buffer, Window, Summary, **File (persistent)**, **VectorStore (semantic)** |
| Graph | Generic `StateGraph[S]` with cycles, conditional/parallel edges, interrupts, and both `MemoryCheckpointer` and **`FileCheckpointer`** |
| Evaluation | `eval` — Dataset (JSONL), Runner, evaluators: ExactMatch, Contains, Regex, JSONEqual, LLMAsJudge |
| Serving | `serve` — LangServe-style HTTP handlers for any Runnable, AgentExecutor, or CompiledGraph (JSON + SSE) |
| Observability | `tracing` — in-memory tree, pretty terminal handler, **JSON-lines exporter** + **FeedbackStore** |

## Install

```bash
go get github.com/grafaelw/golangchain
```

Requires **Go 1.21+** (generics).

## Quick start

The examples in this repo run against **Azure AI Foundry** (Azure OpenAI
Service) using the `llm/openai` package with a custom `WithBaseURL`. If you
have a plain **OpenAI API key** instead, the only change is two lines — see the
comments in each example.

**Azure AI Foundry** — create a `.env` file:

```
AZURE_OPENAI_API_KEY=<your-key>
```

```go
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

    // Azure AI Foundry endpoint — openai package with a custom base URL.
    model, _ := openai.New(
        openai.WithAPIKey(os.Getenv("AZURE_OPENAI_API_KEY")),
        openai.WithModel("gpt-4o-mini"),
        openai.WithBaseURL("https://<resource>.services.ai.azure.com/openai/v1/"),
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

**OpenAI API** — swap the model initialisation; everything else stays the same:

```go
// Create a .env with: OPENAI_API_KEY=sk-...
model, _ := openai.New(
    openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
    openai.WithModel("gpt-4o-mini"),
    // no WithBaseURL — defaults to api.openai.com
)
```

## Packages

| Package | Purpose |
|---|---|
| `schema` | Shared types: `Message`, `Document`, `ToolCall`, `Generation`, `StreamChunk` |
| `llm` | `LLM` interface + functional call options |
| `llm/openai` | OpenAI Chat Completions — also works with Azure AI Foundry via `WithBaseURL` |
| `llm/azure` | Azure OpenAI Service (Azure SDK, separate from `llm/openai`) |
| `llm/anthropic` | Anthropic Claude |
| `llm/gemini` | Google Gemini |
| `llm/ollama` | Local Ollama server |
| `llm/openaicompat` | Any OpenAI-schema server (vLLM, LM Studio, llama.cpp …) |
| `llmutil` | LLM wrappers: `CachingLLM`, `RetryingLLM`, `RateLimitedLLM`; `MemoryCache`, `FileCache` |
| `prompt` | `PromptTemplate`, `ChatPromptTemplate`, `FewShotPromptTemplate` |
| `output` | Typed parsers: `Str`, `JSON`, `Struct[T]`, `List`, `Bool` |
| `chain` | `Runnable` / `Pipe`, `LLMChain`, `SequentialChain`, `MapChain`, `RouterChain`, `RetrievalQAChain`, `MapReduceSummarizer`, `RefineSummarizer` |
| `memory` | `ConversationBufferMemory`, `ConversationWindowMemory`, `ConversationSummaryMemory`, `FileChatHistory`, `VectorStoreMemory` |
| `tools` | `Tool` interface, `Calculator`, `HTTPFetch`, `DuckDuckGoSearch`, `ShellTool`, `FuncTool` |
| `agent` | `ReActAgent`, `ToolCallingAgent`, `PlanAndExecuteAgent`, `AgentExecutor`, streaming `AgentEvent` |
| `embeddings` | `Embedder` interface, `OpenAIEmbedder`, `AzureEmbedder` |
| `textsplitter` | `CharacterSplitter`, `RecursiveCharacterSplitter`, `MarkdownSplitter` |
| `documentloader` | `TextLoader`, `MarkdownLoader`, `CSVLoader`, `HTMLLoader`, `HTTPLoader`, `DirectoryLoader` |
| `vectorstore` | `VectorStore` interface, `InMemoryVectorStore`, `FileVectorStore`, `RetrieverTool` |
| `retriever` | `Retriever` interface, `VectorStoreRetriever`, `BM25Retriever`, `EnsembleRetriever` (RRF), `MultiQueryRetriever`, `ContextualCompressionRetriever` |
| `callbacks` | `Handler` interface, `CallbackManager` fan-out, `LoggingHandler` |
| `tracing` | In-memory `Tracer`, `PrettyHandler`, `JSONLinesExporter`, `FeedbackStore` |
| `eval` | `Dataset` (JSONL), `Run`, evaluators: `ExactMatch`, `Contains`, `Regex`, `JSONEqual`, `LLMAsJudge` |
| `graph` | `StateGraph[S]`, `CompiledGraph[S]`, `Interrupt`, `MemoryCheckpointer[S]`, `FileCheckpointer[S]` |
| `serve` | LangServe-style HTTP handlers for any `Runnable`, `AgentExecutor`, or `CompiledGraph` |

## LLM providers

All examples use **Azure AI Foundry** via the `llm/openai` package with
`WithBaseURL`. To switch to a different provider, only the model initialisation
changes — all chains, agents, and graphs are provider-agnostic.

```go
// Azure AI Foundry (used in the examples — env: AZURE_OPENAI_API_KEY)
model, _ := openai.New(
    openai.WithAPIKey(os.Getenv("AZURE_OPENAI_API_KEY")),
    openai.WithModel("gpt-4o-mini"),
    openai.WithBaseURL("https://<resource>.services.ai.azure.com/openai/v1/"),
)

// OpenAI API (env: OPENAI_API_KEY)
model, _ := openai.New(
    openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
    openai.WithModel("gpt-4o-mini"),
    // no WithBaseURL — defaults to api.openai.com
)

// Azure OpenAI Service (dedicated Azure SDK, env: AZURE_OPENAI_API_KEY)
model, _ := azure.New(
    azure.WithAPIKey(os.Getenv("AZURE_OPENAI_API_KEY")),
    azure.WithEndpoint(os.Getenv("AZURE_OPENAI_ENDPOINT")),
    azure.WithDeployment("gpt-4o"),
)

// Anthropic Claude
model, _ := anthropic.New(anthropic.WithAPIKey(key), anthropic.WithModel("claude-sonnet-4-5"))

// Google Gemini
model, _ := gemini.New(ctx, gemini.WithAPIKey(key), gemini.WithModel("gemini-2.0-flash"))

// Local Ollama
model, _ := ollama.New(ollama.WithModel("llama3.2"))

// Any OpenAI-compatible server (vLLM, LM Studio, llama.cpp …)
model, _ := openaicompat.New(openaicompat.WithBaseURL("http://localhost:1234/v1"), openaicompat.WithModel("mistral"))
```

All providers satisfy the same `llm.LLM` interface. Swap providers without changing any other code.

## Chains (LCEL-style pipelines)

```go
// Compose with Pipe
pipeline := chain.NewFuncRunnable("upper", func(_ context.Context, in any) (any, error) {
    return strings.ToUpper(in.(string)), nil
}).Pipe(
    chain.NewFuncRunnable("exclaim", func(_ context.Context, in any) (any, error) {
        return in.(string) + "!", nil
    }),
)
result, _ := pipeline.Invoke(ctx, "hello") // "HELLO!"

// Full LLM chain with streaming
c := chain.NewLLMChain(chatPrompt, model, output.AsAny(output.StrOutputParser{}))
streamCh, _ := c.Stream(ctx, map[string]any{"question": "..."})
for chunk := range streamCh {
    fmt.Print(chunk.Value)
}

// Run branches in parallel
mc := chain.NewMapChain("Analyser", map[string]chain.Runnable{
    "summary":  summaryChain,
    "keywords": keywordsChain,
})
results, _ := mc.Invoke(ctx, text) // map[string]any{"summary": ..., "keywords": ...}
```

## Agents

```go
agentTools := []tools.Tool{
    tools.Calculator{},
    tools.NewDuckDuckGoSearch(),
    tools.NewHTTPFetch(),
}

// ToolCallingAgent — uses native function-calling API (GPT-4o, Claude 3+, Gemini 1.5+)
executor := agent.NewAgentExecutor(
    agent.NewToolCallingAgent(model, agentTools, "You are a research assistant."),
    agentTools,
    agent.WithMaxIter(10),
)

// Blocking
answer, _ := executor.Run(ctx, "What is 1337 * 42?")

// Streaming events
for event := range executor.Stream(ctx, "What is the population of Amsterdam?") {
    switch event.Type {
    case agent.EventToolCall:    fmt.Println("Calling:", event.Action.Tool)
    case agent.EventFinalAnswer: fmt.Println("Answer:", event.Answer)
    }
}

// Custom tool
myTool := tools.NewFuncTool("reverse", "Reverses a string.", nil,
    func(_ context.Context, input string) (string, error) {
        r := []rune(input)
        for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 { r[i], r[j] = r[j], r[i] }
        return string(r), nil
    },
)
```

## StateGraph (LangGraph equivalent)

```go
type State struct {
    Messages []schema.Message
    Next     string
}

reducer := func(cur, upd State) State {
    cur.Messages = append(cur.Messages, upd.Messages...)
    if upd.Next != "" { cur.Next = upd.Next }
    return cur
}

g := graph.NewStateGraph(reducer)
g.MustAddNode("agent", agentNode)
g.MustAddNode("tools", toolsNode)
g.MustAddEdge(graph.START, "agent")
g.MustAddConditionalEdges("agent", routerFn, map[string]string{
    "tools": "tools",
    "end":   graph.END,
})
g.MustAddEdge("tools", "agent") // cycle: agent can call tools multiple times

checkpointer := graph.NewMemoryCheckpointer[State]()
compiled, _ := g.Compile(
    graph.WithCheckpointer[State](checkpointer),
    graph.WithMaxSteps[State](50),
)

// Stream node events in real time
for event := range compiled.Stream(ctx, initialState, graph.WithThreadID[State]("thread-1")) {
    fmt.Println(event.Type, event.Node)
}
```

### Human-in-the-loop

```go
// A node pauses execution by returning an Interrupt error
myNode := func(ctx context.Context, s State) (State, error) {
    return s, graph.NewInterrupt("Waiting for human approval")
}

// First run — pauses
_, err := compiled.Invoke(ctx, state, graph.WithThreadID[State]("t1"))

// Human reviews, then resume from saved checkpoint
saved, _ := checkpointer.Load(ctx, "t1")
finalState, _ := compiled.Invoke(ctx, saved.State, graph.WithThreadID[State]("t1"))
```

## Memory

```go
// Buffer — full history
mem := memory.NewConversationBufferMemory()

// Window — keep last k turns
mem := memory.NewConversationWindowMemory(5)

// Summary — compress old turns via LLM
mem := memory.NewConversationSummaryMemory(model)

// Use with LLMChain
vars, _ := mem.LoadMemoryVariables(ctx)
vars["question"] = userInput
answer, _ := c.Invoke(ctx, vars)
_ = mem.SaveContext(ctx, userInput, answer.(string))
```

## Output parsers

```go
output.StrOutputParser{}.Parse("  hello  ")           // "hello"
output.JSONOutputParser{}.Parse("```json\n{...}\n```") // map[string]any
output.NewStructOutputParser[MyStruct]().Parse("{...}") // MyStruct
output.NewListOutputParser(output.SepComma).Parse("a,b,c") // ["a","b","c"]
output.BoolOutputParser{}.Parse("yes")                // true

// Bridge typed Parser[T] to LLMChain's untyped interface
chain.NewLLMChain(tmpl, model, output.AsAny(output.StrOutputParser{}))
chain.NewLLMChain(tmpl, model, output.AsAny(output.NewStructOutputParser[Reply]()))
```

## Callbacks (observability)

```go
type MyTracer struct{ callbacks.NoOpHandler }

func (t *MyTracer) OnLLMEnd(_ context.Context, model string, gen *schema.Generation) {
    fmt.Printf("model=%s tokens=%d\n", model, gen.Usage.TotalTokens)
}
func (t *MyTracer) OnToolStart(_ context.Context, tool, input string) {
    fmt.Printf("calling tool=%s\n", tool)
}

cm := callbacks.NewCallbackManager(
    &MyTracer{},
    callbacks.NewLoggingHandler(log.Printf),
)

// Attach to any component
chain.NewLLMChain(tmpl, model, parser, chain.WithChainCallbacks(cm))
agent.NewAgentExecutor(myAgent, tools, agent.WithCallbackManager(cm))
```

## Vector store (RAG)

```go
// Azure embeddings (used in the examples)
embedder, _ := embeddings.NewAzureEmbedder(
    os.Getenv("OPENAI_KEY"),
    "https://<resource>.cognitiveservices.azure.com",
    os.Getenv("OPENAI_EMBEDDING_DEPLOYMENT"),
    os.Getenv("OPENAI_API_VERSION"),
)

// OpenAI embeddings alternative
embedder, _ := embeddings.NewOpenAIEmbedder(os.Getenv("OPENAI_API_KEY"), "text-embedding-3-small")

store := vectorstore.NewInMemoryVectorStore(embedder)

_ = store.AddDocuments(ctx, []schema.Document{
    {PageContent: "Go was created at Google in 2007.", Metadata: map[string]any{"id": "1"}},
    {PageContent: "Go 1.18 introduced generics.",      Metadata: map[string]any{"id": "2"}},
})

results, _ := store.SimilaritySearch(ctx, "golang generics", 3)
for _, doc := range results {
    fmt.Printf("score=%.3f  %s\n", doc.Score, doc.PageContent)
}

// Use as an agent tool
retriever := vectorstore.NewRetrieverTool(store, 5, "search_docs", "Search project documentation.")
executor  := agent.NewAgentExecutor(myAgent, []tools.Tool{retriever})
```

## Design principles

- **Idiomatic Go** — `context.Context` everywhere, error returns, channels for streaming, functional options
- **No panics in library code** — `Must*` variants are provided for init-time use only
- **Interface-first** — swap any component (`LLM`, `Tool`, `Memory`, `Checkpointer`, `Embedder`, `VectorStore`) with your own implementation
- **Streaming first-class** — every LLM call and chain supports `<-chan T` streaming
- **Generics for type safety** — `StateGraph[S]`, `Parser[T]`, `StructOutputParser[T]`, `Checkpointer[S]`
- **Zero global state** — all configuration is passed explicitly

## Examples

| Example | Description |
|---|---|
| [`examples/simple_chain`](examples/simple_chain/main.go) | LCEL pipeline + memory + streaming |
| [`examples/chains`](examples/chains/main.go) | LCEL Runnables (FuncRunnable, Sequential/Map/Router chains) + output parsers |
| [`examples/memory_and_tools`](examples/memory_and_tools/main.go) | Buffer/Window/Summary memory + Calculator/FuncTool |
| [`examples/react_agent`](examples/react_agent/main.go) | Tool-calling agent with streaming events |
| [`examples/state_graph`](examples/state_graph/main.go) | LangGraph-style StateGraph with checkpointing and human-in-the-loop |
| [`examples/vectorstore`](examples/vectorstore/main.go) | Embeddings + InMemoryVectorStore + RetrieverTool |
| [`examples/tracing`](examples/tracing/main.go) | LangSmith-style tracing with PrettyHandler + in-memory run tree |
| [`examples/rag`](examples/rag/main.go) | Full RAG pipeline: loader → splitter → hybrid retriever → QA chain + eval |

All examples use **Azure AI Foundry** by default. Every file contains a
clearly marked comment block showing the two-line swap for the **OpenAI API**.

**Azure AI Foundry** (most examples):

```bash
# .env
AZURE_OPENAI_API_KEY=<your-key>

go run ./examples/simple_chain
go run ./examples/chains
go run ./examples/memory_and_tools
go run ./examples/react_agent
go run ./examples/state_graph
go run ./examples/tracing
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

**Embeddings examples** (`vectorstore`, `rag`) need additional env vars:

```bash
# Azure embeddings (.env)
OPENAI_KEY=<azure-api-key>
OPENAI_ENDPOINT=https://<resource>.cognitiveservices.azure.com
OPENAI_EMBEDDING_DEPLOYMENT=<deployment-name>
OPENAI_API_VERSION=2024-02-01

# OpenAI embeddings alternative — replace the embedder block with:
# embedder, err := embeddings.NewOpenAIEmbedder(os.Getenv("OPENAI_API_KEY"), "text-embedding-3-small")
```

## License

MIT
