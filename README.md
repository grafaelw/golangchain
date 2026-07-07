# golangchain

A production-grade **LangChain + LangGraph equivalent library for Go**.

Composable LLM chains, tool-using agents, conversation memory, vector stores,
and a fully generic StateGraph engine — all idiomatic Go, no code generation,
no reflection magic.

## Install

```bash
go get github.com/grafaelw/golangchain
```

Requires **Go 1.21+** (generics).

## Quick start

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

## Packages

| Package | Purpose |
|---|---|
| `schema` | Shared types: `Message`, `Document`, `ToolCall`, `Generation`, `StreamChunk` |
| `llm` | `LLM` interface + functional call options |
| `llm/openai` | OpenAI Chat Completions |
| `llm/azure` | Azure OpenAI Service |
| `llm/anthropic` | Anthropic Claude |
| `llm/gemini` | Google Gemini |
| `llm/ollama` | Local Ollama server |
| `llm/openaicompat` | Any OpenAI-schema server (vLLM, LM Studio, llama.cpp …) |
| `prompt` | `PromptTemplate`, `ChatPromptTemplate`, `FewShotPromptTemplate` |
| `output` | Typed parsers: `Str`, `JSON`, `Struct[T]`, `List`, `Bool` |
| `chain` | `Runnable` / `Pipe`, `LLMChain`, `SequentialChain`, `MapChain`, `RouterChain` |
| `memory` | `ConversationBufferMemory`, `ConversationWindowMemory`, `ConversationSummaryMemory` |
| `tools` | `Tool` interface, `Calculator`, `HTTPFetch`, `DuckDuckGoSearch`, `ShellTool`, `FuncTool` |
| `agent` | `ReActAgent`, `ToolCallingAgent`, `AgentExecutor`, streaming `AgentEvent` |
| `embeddings` | `Embedder` interface, `OpenAIEmbedder`, `AzureEmbedder` |
| `vectorstore` | `VectorStore` interface, `InMemoryVectorStore` (cosine similarity), `RetrieverTool` |
| `callbacks` | `Handler` interface, `CallbackManager` fan-out, `LoggingHandler` |
| `graph` | `StateGraph[S]`, `CompiledGraph[S]`, `Interrupt`, `MemoryCheckpointer[S]` |

## LLM providers

```go
// OpenAI
model, _ := openai.New(openai.WithAPIKey(key), openai.WithModel("gpt-4o"))

// Azure OpenAI
model, _ := azure.New(
    azure.WithAPIKey(key),
    azure.WithEndpoint(endpoint),
    azure.WithDeployment("gpt-4o"),
)

// Anthropic Claude
model, _ := anthropic.New(anthropic.WithAPIKey(key), anthropic.WithModel("claude-sonnet-4-5"))

// Google Gemini
model, _ := gemini.New(ctx, gemini.WithAPIKey(key), gemini.WithModel("gemini-2.0-flash"))

// Local Ollama
model, _ := ollama.New(ollama.WithModel("llama3.2"))

// Any OpenAI-compatible server
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
embedder, _ := embeddings.NewOpenAIEmbedder(os.Getenv("OPENAI_API_KEY"))
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
| [`examples/react_agent`](examples/react_agent/main.go) | Tool-calling agent with streaming events |
| [`examples/state_graph`](examples/state_graph/main.go) | LangGraph-style StateGraph with checkpointing and human-in-the-loop |
| [`examples/chains`](examples/chains/main.go) | LCEL Runnables (FuncRunnable, Sequential/Map/Router chains) + output parsers |
| [`examples/memory_and_tools`](examples/memory_and_tools/main.go) | Buffer/Window/Summary memory + Calculator/FuncTool |
| [`examples/vectorstore`](examples/vectorstore/main.go) | Embeddings + InMemoryVectorStore + RetrieverTool |

Run any example:

```bash
OPENAI_API_KEY=sk-... go run ./examples/simple_chain
OPENAI_API_KEY=sk-... go run ./examples/chains
OPENAI_API_KEY=sk-... go run ./examples/memory_and_tools
OPENAI_API_KEY=sk-... go run ./examples/vectorstore
```

## License

MIT
