// Package schema defines the core shared types used throughout golangchain.
//
// All other packages import schema; schema itself has no internal dependencies,
// making it safe to embed in any project without pulling in provider SDKs.
//
// # Core types
//
//   - [Message] — a single conversation turn (system / human / ai / tool)
//   - [Document] — a chunk of text with metadata, used in RAG pipelines
//   - [ToolCall] / [ToolDef] — native function-calling structures
//   - [Generation] — complete output of one LLM call
//   - [StreamChunk] — one token emitted during streaming
//   - [AgentAction] / [AgentFinish] / [AgentStep] — agent decision types
//   - [TokenUsage] — prompt / completion / total token counts
//
// # Message constructors
//
//	schema.NewSystemMessage("You are a helpful assistant.")
//	schema.NewHumanMessage("What is Go?")
//	schema.NewAIMessage("Go is a compiled language created at Google.")
//	schema.NewToolMessage(result, callID, toolName)
package schema
