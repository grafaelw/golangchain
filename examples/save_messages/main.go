// Example: save_messages
//
// Demonstrates the MessagesMemory interface introduced to improve message
// persistence.  Unlike SaveContext (which takes raw user/AI strings),
// SaveMessages accepts a full []schema.Message slice — preserving tool
// calls, tool call IDs, names, and other metadata that SaveContext strips.
//
// Highlights:
//   - MessagesMemory interface extends Memory with SaveMessages.
//   - All three memory types implement it: ConversationBufferMemory,
//     ConversationWindowMemory, and ConversationSummaryMemory.
//   - Tool-call messages are preserved intact across saves/loads.
//   - Message.String() provides a compact, readable representation.
//
//	Run this example with:
//	  go run ./examples/save_messages
package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/grafaelw/golangchain/memory"
	"github.com/grafaelw/golangchain/schema"
)

func main() {
	ctx := context.Background()

	// -------------------------------------------------------------------------
	// Simulate a tool-calling conversation.
	// -------------------------------------------------------------------------
	toolCall := schema.ToolCall{
		ID:        "call_001",
		Type:      "function",
		Name:      "weather",
		Arguments: json.RawMessage(`{"city":"Paris"}`),
	}
	msgs := []schema.Message{
		schema.NewSystemMessage("You are a helpful assistant."),
		schema.NewHumanMessage("What's the weather in Paris?"),
		{
			Role:      schema.RoleAI,
			Content:   "",
			ToolCalls: []schema.ToolCall{toolCall},
		},
		{
			Role:       schema.RoleTool,
			Content:    "Paris: 18°C, partly cloudy",
			ToolCallID: "call_001",
		},
		schema.NewAIMessage("The weather in Paris is 18°C with partly cloudy skies."),
	}

	fmt.Println("Input messages (String preview):")
	for _, m := range msgs {
		fmt.Printf("  %s\n", m.String())
	}

	// -------------------------------------------------------------------------
	// 1. ConversationBufferMemory with SaveMessages
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 1. BufferMemory (SaveMessages) ---")
	buf := memory.NewConversationBufferMemory()
	if err := buf.SaveMessages(ctx, msgs); err != nil {
		panic(err)
	}
	vars, _ := buf.LoadMemoryVariables(ctx)
	history := vars["history"].([]schema.Message)
	for _, m := range history {
		tc := ""
		if len(m.ToolCalls) > 0 {
			tc = fmt.Sprintf("  [tool_call_id=%s]", m.ToolCalls[0].ID)
		}
		if m.ToolCallID != "" {
			tc = fmt.Sprintf("  [tool_call_id=%s]", m.ToolCallID)
		}
		fmt.Printf("  %-8s %q%s\n", m.Role, trunc(m.Content, 40), tc)
	}

	// -------------------------------------------------------------------------
	// 2. Tool-call ID and content preserved after round-trip
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 2. Tool-call round-trip integrity ---")
	toolCallMsg := history[2]  // the assistant message with ToolCalls
	toolResultMsg := history[3] // the tool result
	fmt.Printf("  Assistant has %d tool call(s)\n", len(toolCallMsg.ToolCalls))
	fmt.Printf("  Tool call name:       %s\n", toolCallMsg.ToolCalls[0].Name)
	fmt.Printf("  Tool call arguments:  %s\n", string(toolCallMsg.ToolCalls[0].Arguments))
	fmt.Printf("  Tool result ID:       %s\n", toolResultMsg.ToolCallID)
	fmt.Printf("  Tool result content:  %q\n", toolResultMsg.Content)

	// -------------------------------------------------------------------------
	// 3. ConversationWindowMemory with SaveMessages (k=3)
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 3. WindowMemory (SaveMessages, k=3) ---")
	win := memory.NewConversationWindowMemory(3)
	for i := 0; i < 5; i++ {
		pair := []schema.Message{
			schema.NewHumanMessage(fmt.Sprintf("question %d", i)),
			schema.NewAIMessage(fmt.Sprintf("answer %d", i)),
		}
		_ = win.SaveMessages(ctx, pair)
	}
	wVars, _ := win.LoadMemoryVariables(ctx)
	wHist := wVars["history"].([]schema.Message)
	fmt.Printf("  %d messages visible (oldest pairs trimmed)\n", len(wHist))
	for _, m := range wHist {
		fmt.Printf("  %s\n", m.String())
	}
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}