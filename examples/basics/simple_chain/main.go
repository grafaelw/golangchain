// Example: simple_chain
//
// Demonstrates the LCEL-style pipeline:
//
//	ChatPromptTemplate → LLM → StrOutputParser
//
// # Usage — Azure AI Foundry (default)
//
// Create a .env file with:
//
//	AZURE_OPENAI_API_KEY=<your-key>
//
// Then run:
//
//	go run ./examples/basics/simple_chain
//
// # Usage — OpenAI API
//
// Replace the model initialisation block with:
//
//	model, err := openai.New(
//	    openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
//	    openai.WithModel("gpt-4o-mini"),
//	)
//
// Create a .env file with:
//
//	OPENAI_API_KEY=sk-...
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/joho/godotenv"

	"github.com/grafaelw/golangchain/callbacks"
	"github.com/grafaelw/golangchain/chain"
	"github.com/grafaelw/golangchain/llm/openai"
	"github.com/grafaelw/golangchain/memory"
	"github.com/grafaelw/golangchain/output"
	"github.com/grafaelw/golangchain/prompt"
)

func main() {
	ctx := context.Background()

	// -------------------------------------------------------------------------
	// 1. Load .env (silently ignored if the file doesn't exist,
	// 	  so real environment variables still work in CI/production)
	// -------------------------------------------------------------------------
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found - using environment variables")
	}

	// -------------------------------------------------------------------------
	// 2. Create the LLM — Azure AI Foundry via the openai package.
	//
	// To use the OpenAI API instead, replace this block with:
	//
	//     model, err := openai.New(
	//         openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
	//         openai.WithModel("gpt-4o-mini"),
	//     )
	//
	// and set OPENAI_API_KEY in your .env.
	// -------------------------------------------------------------------------
	model, err := openai.New(
		openai.WithAPIKey(os.Getenv("AZURE_OPENAI_API_KEY")),
		openai.WithModel("gpt-5.4-nano"),
		openai.WithBaseURL("https://ai-lab-nl-sweden-foundry.services.ai.azure.com/openai/v1/"),
	)
	if err != nil {
		log.Fatal(err)
	}

	// -------------------------------------------------------------------------
	// 3. Build a chat prompt with conversation history placeholder
	// -------------------------------------------------------------------------
	chatPrompt := prompt.MustNewChatPromptTemplate(
		prompt.MustSystem("You are a knowledgeable assistant. Answer concisely."),
		prompt.NewMessagePlaceholder("history"),
		prompt.MustHuman("{{.question}}"),
	)

	// -------------------------------------------------------------------------
	// 4. Attach a callback for logging
	// -------------------------------------------------------------------------
	cb := callbacks.NewCallbackManager(
		callbacks.NewLoggingHandler(log.Printf),
	)

	// -------------------------------------------------------------------------
	// 5. Build the LLMChain
	// -------------------------------------------------------------------------
	c := chain.NewLLMChain(
		chatPrompt,
		model,
		output.AsAny(output.StrOutputParser{}),
		chain.WithChainName("SimpleQAChain"),
		chain.WithChainCallbacks(cb),
	)

	// -------------------------------------------------------------------------
	// 6. Set up conversation memory
	// -------------------------------------------------------------------------
	mem := memory.NewConversationWindowMemory(5)

	// -------------------------------------------------------------------------
	// 7. Run a multi-turn conversation
	// -------------------------------------------------------------------------
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

	// -------------------------------------------------------------------------
	// 8. Demonstrate streaming on the last question
	// -------------------------------------------------------------------------
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
