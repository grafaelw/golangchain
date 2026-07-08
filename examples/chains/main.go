// Example: chains
//
// Demonstrates the LCEL-style Runnable building blocks in the chain package,
// plus the output parsers used to type LLM responses:
//
//   - FuncRunnable & Pipe  — wrap any func(ctx, any) (any, error) and compose
//   - LLMChain             — prompt → LLM → parser
//   - SequentialChain      — thread output of step N into step N+1
//   - MapChain             — fan input to parallel branches
//   - RouterChain          — pick a branch based on a routing function
//   - output.*Parser       — Str / JSON / Struct / List / Bool
//
// # Usage — Azure AI Foundry (default)
//
// Create a .env file with:
//
//	AZURE_OPENAI_API_KEY=<your-key>
//
// Then run:
//
//	go run ./examples/chains
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
	"strings"
	"unicode/utf8"

	"github.com/joho/godotenv"

	"github.com/grafaelw/golangchain/chain"
	"github.com/grafaelw/golangchain/llm/openai"
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
	// 3. FuncRunnable & Pipe
	// -------------------------------------------------------------------------
	fmt.Println("--- 3. FuncRunnable & Pipe ---")
	upper := chain.NewFuncRunnable("upper", func(_ context.Context, in any) (any, error) {
		return strings.ToUpper(in.(string)), nil
	})
	exclaim := chain.NewFuncRunnable("exclaim", func(_ context.Context, in any) (any, error) {
		return in.(string) + "!!!", nil
	})

	pipe := upper.Pipe(exclaim)
	result, err := pipe.Invoke(ctx, "hello world")
	if err != nil {
		panic(err)
	}
	fmt.Println("Pipe result:", result)

	// -------------------------------------------------------------------------
	// 4. LLMChain: prompt → LLM → StrOutputParser
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 4. LLMChain ---")

	chatPrompt := prompt.MustNewChatPromptTemplate(
		prompt.MustSystem("You are a geography expert. Answer concisely."),
		prompt.MustHuman("What is the capital of {{.Country}}?"),
	)

	llmChain := chain.NewLLMChain(
		chatPrompt,
		model,
		output.AsAny(output.StrOutputParser{}),
		chain.WithChainName("GeoChain"),
	)

	ans, err := llmChain.Invoke(ctx, map[string]any{"Country": "Germany"})
	if err != nil {
		panic(err)
	}
	fmt.Println("LLMChain answer:", ans)

	// -------------------------------------------------------------------------
	// 5. SequentialChain
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 5. SequentialChain ---")
	tokenise := chain.NewFuncRunnable("tokenise", func(_ context.Context, in any) (any, error) {
		return strings.Fields(in.(string)), nil
	})
	count := chain.NewFuncRunnable("count", func(_ context.Context, in any) (any, error) {
		return fmt.Sprintf("%d words", len(in.([]string))), nil
	})
	shout := chain.NewFuncRunnable("shout", func(_ context.Context, in any) (any, error) {
		return strings.ToUpper(in.(string)), nil
	})

	seq := chain.NewSequentialChain("WordCounter", tokenise, count, shout)
	seqOut, err := seq.Invoke(ctx, "The quick brown fox jumps over the lazy dog")
	if err != nil {
		panic(err)
	}
	fmt.Println("SequentialChain:", seqOut)

	// -------------------------------------------------------------------------
	// 6. MapChain: fan input to parallel branches
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 6. MapChain ---")
	branches := map[string]chain.Runnable{
		"upper": chain.NewFuncRunnable("upper", func(_ context.Context, in any) (any, error) {
			return strings.ToUpper(in.(string)), nil
		}),
		"word_count": chain.NewFuncRunnable("wc", func(_ context.Context, in any) (any, error) {
			return len(strings.Fields(in.(string))), nil
		}),
		"char_count": chain.NewFuncRunnable("cc", func(_ context.Context, in any) (any, error) {
			return utf8.RuneCountInString(in.(string)), nil
		}),
	}

	mc := chain.NewMapChain("TextAnalyser", branches)
	mapResult, err := mc.Invoke(ctx, "Hello, golangchain!")
	if err != nil {
		panic(err)
	}
	m := mapResult.(map[string]any)
	fmt.Printf("MapChain results:\n  upper=%s\n  words=%v\n  chars=%v\n",
		m["upper"], m["word_count"], m["char_count"])

	// -------------------------------------------------------------------------
	// 7. RouterChain: pick a branch based on a routing function
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 7. RouterChain ---")
	mathChain := chain.NewFuncRunnable("math", func(_ context.Context, in any) (any, error) {
		return "[MATH] " + in.(string), nil
	})
	scienceChain := chain.NewFuncRunnable("science", func(_ context.Context, in any) (any, error) {
		return "[SCIENCE] " + in.(string), nil
	})
	defaultChain := chain.NewFuncRunnable("default", func(_ context.Context, in any) (any, error) {
		return "[GENERAL] " + in.(string), nil
	})

	router := chain.NewRouterChain(
		"TopicRouter",
		func(_ context.Context, in any) (string, error) {
			s := strings.ToLower(in.(string))
			switch {
			case strings.Contains(s, "math") || strings.Contains(s, "calculat"):
				return "math", nil
			case strings.Contains(s, "physics") || strings.Contains(s, "biology"):
				return "science", nil
			default:
				return "unknown", nil
			}
		},
		map[string]chain.Runnable{
			"math":    mathChain,
			"science": scienceChain,
		},
		defaultChain,
	)

	for _, q := range []string{
		"Calculate 2+2",
		"Explain quantum physics",
		"What is Go?",
	} {
		out, err := router.Invoke(ctx, q)
		if err != nil {
			panic(err)
		}
		fmt.Printf("Q: %-30s -> %s\n", q, out)
	}

	// -------------------------------------------------------------------------
	// 8. Output parsers
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 8. Output parsers ---")

	str, _ := output.StrOutputParser{}.Parse("  Hello, golangchain!  ")
	fmt.Println("Str:", str)

	jsonRaw := "```json\n{\"answer\": 42, \"unit\": \"km\"}\n```"
	parsed, err := output.JSONOutputParser{}.Parse(jsonRaw)
	if err != nil {
		panic(err)
	}
	fmt.Printf("JSON: answer=%.0f unit=%s\n", parsed["answer"], parsed["unit"])

	type WeatherReport struct {
		City   string  `json:"city"`
		TempC  float64 `json:"temp_c"`
		Cloudy bool    `json:"cloudy"`
	}
	structParser := output.NewStructOutputParser[WeatherReport]()
	report, err := structParser.Parse(`{"city":"Amsterdam","temp_c":14.5,"cloudy":true}`)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Struct: %s %.1f°C cloudy=%v\n", report.City, report.TempC, report.Cloudy)

	items, _ := output.NewListOutputParser(output.SepComma).Parse("Go, Python, Rust, TypeScript")
	fmt.Printf("List (%d items): %v\n", len(items), items)

	yes, _ := output.BoolOutputParser{}.Parse("yes")
	no, _ := output.BoolOutputParser{}.Parse("false")
	fmt.Printf("Bool: yes=%v, false=%v\n", yes, no)
}
