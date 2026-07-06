// Example: simple_chain
//
// Demonstrates the LCEL-style pipeline:
//   ChatPromptTemplate → LLM → StrOutputParser
//
// Usage:
//   OPENAI_API_KEY=sk-... go run .
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/grafaelw/golangchain/callbacks"
	"github.com/grafaelw/golangchain/chain"
	"github.com/grafaelw/golangchain/llm/openai"
	"github.com/grafaelw/golangchain/memory"
	"github.com/grafaelw/golangchain/output"
	"github.com/grafaelw/golangchain/prompt"
)

func main() {
	ctx := context.Background()

	// 1. Create the LLM
	model, err := openai.New(
		openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
		openai.WithModel("gpt-4o-mini"),
	)
	if err != nil {
		log.Fatal(err)
	}

	// 2. Build a chat prompt with conversation history placeholder
	chatPrompt := prompt.MustNewChatPromptTemplate(
		prompt.MustSystem("You are a knowledgeable assistant. Answer concisely."),
		prompt.NewMessagePlaceholder("history"),
		prompt.MustHuman("{{.question}}"),
	)

	// 3. Attach a callback for logging
	cb := callbacks.NewCallbackManager(
		callbacks.NewLoggingHandler(log.Printf),
	)

	// 4. Build the LLMChain
	c := chain.NewLLMChain(
		chatPrompt,
		model,
		output.AsAny(output.StrOutputParser{}),
		chain.WithChainName("SimpleQAChain"),
		chain.WithChainCallbacks(cb),
	)

	// 5. Set up conversation memory
	mem := memory.NewConversationWindowMemory(5)

	// 6. Run a multi-turn conversation
	questions := []string{
		"What is the capital of the Netherlands?",
		"What is its most famous museum?",
		"How many visitors does it get per year?",
	}

	for _, q := range questions {
		// Load history
		vars, _ := mem.LoadMemoryVariables(ctx)
		vars["question"] = q

		result, err := c.Invoke(ctx, vars)
		if err != nil {
			log.Fatalf("chain error: %v", err)
		}
		answer := result.(string)
		fmt.Printf("Q: %s\nA: %s\n\n", q, answer)

		// Save turn to memory
		_ = mem.SaveContext(ctx, q, answer)
	}

	// 7. Demonstrate streaming on the last question
	fmt.Println("--- Streaming demo ---")
	vars, _ := mem.LoadMemoryVariables(ctx)
	vars["question"] = "Can you recommend one book about Amsterdam?"

	streamCh, err := c.Stream(ctx, vars)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Print("A: ")
	for chunk := range streamCh {
		if chunk.Err != nil {
			log.Fatal(chunk.Err)
		}
		if s, ok := chunk.Value.(string); ok {
			fmt.Print(s)
		}
	}
	fmt.Println()
}
