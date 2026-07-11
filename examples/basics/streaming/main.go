// Example: streaming
//
// Demonstrates the unified schema.StreamChunk type used across the entire
// library and how Runnable.Stream() pipes typed values through a chain
// without the boilerplate of an LLM round-trip.
//
// Highlights:
//   - FuncRunnable.Stream() populates StreamChunk.Value with the concrete Go
//     type produced by the function.
//   - Piped runnables receive StreamChunk.Value and can transform it, keeping
//     the chain type-safe end to end.
//   - LLMChain.Stream() automatically sets StreamChunk.Value = chunk.Text so
//     downstream consumers work identically regardless of whether the
//     producer is an LLM or a pure-Go function.
//   - The streaming contract is explicit: if a runnable does not support
//     streaming it must return an error (see MapChain).
//
//	Run this example with:
//	  go run ./examples/basics/streaming
package main

import (
	"context"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/grafaelw/golangchain/chain"
)

func main() {
	ctx := context.Background()

	// -------------------------------------------------------------------------
	// 1. Single runnable streaming — StreamChunk.Value
	// -------------------------------------------------------------------------
	fmt.Println("--- 1. Single-runnable stream ---")
	toUpper := chain.NewFuncRunnable("upper", func(_ context.Context, in any) (any, error) {
		return strings.ToUpper(in.(string)), nil
	})

	ch, err := toUpper.Stream(ctx, "hello streaming")
	if err != nil {
		panic(err)
	}
	for chunk := range ch {
		if chunk.Err != nil {
			panic(chunk.Err)
		}
		fmt.Printf("  chunk.Value=%v  chunk.Text=\"%s\"\n", chunk.Value, chunk.Text)
	}

	// -------------------------------------------------------------------------
	// 2. Pipeline streaming — values flow through
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 2. Pipeline stream ---")
	splitWords := chain.NewFuncRunnable("split", func(_ context.Context, in any) (any, error) {
		return strings.Fields(in.(string)), nil
	})
	countWords := chain.NewFuncRunnable("count", func(_ context.Context, in any) (any, error) {
		words := in.([]string)
		return fmt.Sprintf("%d words, %d chars", len(words), totalChars(words)), nil
	})

	pipe := splitWords.Pipe(countWords)
	ch2, err := pipe.Stream(ctx, "the quick brown fox")
	if err != nil {
		panic(err)
	}
	for chunk := range ch2 {
		if chunk.Err != nil {
			panic(chunk.Err)
		}
		fmt.Printf("  result: %v\n", chunk.Value)
	}

	// -------------------------------------------------------------------------
	// 3. MapChain.Stream returns an error (explicit contract)
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 3. MapChain.Stream → error ---")
	branches := map[string]chain.Runnable{
		"upper": toUpper,
	}
	mc := chain.NewMapChain("test", branches)
	_, err = mc.Stream(ctx, "input")
	if err != nil {
		fmt.Printf("  Expected error: %v\n", err)
	}

	// -------------------------------------------------------------------------
	// 4. StreamChunk from an error-producing runnable
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 4. Stream with errors ---")
	failer := chain.NewFuncRunnable("fail", func(_ context.Context, in any) (any, error) {
		if in.(string) == "boom" {
			return nil, fmt.Errorf("simulated failure")
		}
		return in.(string) + "!", nil
	})

	ch3, _ := failer.Stream(ctx, "boom")
	for chunk := range ch3 {
		if chunk.Err != nil {
			fmt.Printf("  chunk.Err: %v\n", chunk.Err)
		}
	}
}

func totalChars(words []string) int {
	n := 0
	for _, w := range words {
		n += utf8.RuneCountInString(w)
	}
	return n
}