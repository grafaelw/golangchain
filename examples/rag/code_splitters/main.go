// Example: code_splitters
//
// Demonstrates language-aware code splitters that respect class/function
// boundaries for Python, JavaScript/TypeScript, and Go source code.
//
// Run this example with:
//
//	go run ./examples/rag/code_splitters
package main

import (
	"fmt"

	"github.com/grafaelw/golangchain/textsplitter"
)

func main() {
	// -------------------------------------------------------------------------
	// 1. Python splitter — splits at class/function boundaries
	// -------------------------------------------------------------------------
	fmt.Println("--- 1. PythonSplitter ---")
	py := textsplitter.NewPythonSplitter(textsplitter.WithChunkSize(120))

	pythonCode := `class UserService:
    def create_user(self, name, email):
        user = User(name=name, email=email)
        return user.save()

    def delete_user(self, user_id):
        return User.delete_by_id(user_id)

class EmailService:
    def send_welcome(self, user):
        return Email.send(user.email, "Welcome!")`

	pyChunks := py.SplitText(pythonCode)
	fmt.Printf("  %d chunks:\n", len(pyChunks))
	for i, c := range pyChunks {
		fmt.Printf("  [%d] %s\n", i+1, trunc(firstLine(c), 60))
	}

	// -------------------------------------------------------------------------
	// 2. Go splitter — splits at func/type boundaries
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 2. GoSplitter ---")
	goSplit := textsplitter.NewGoSplitter(textsplitter.WithChunkSize(100))

	goCode := `package main

import "fmt"

func main() {
    fmt.Println("hello")
}

func helper(x int) int {
    return x * 2
}

type Config struct {
    Port int
    Host string
}`

	goChunks := goSplit.SplitText(goCode)
	fmt.Printf("  %d chunks:\n", len(goChunks))
	for i, c := range goChunks {
		fmt.Printf("  [%d] %s\n", i+1, trunc(firstLine(c), 60))
	}

	// -------------------------------------------------------------------------
	// 3. JavaScript/TypeScript splitter
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 3. JavaScriptSplitter ---")
	js := textsplitter.NewJavaScriptSplitter(textsplitter.WithChunkSize(120))

	jsCode := `function add(a, b) {
    return a + b;
}

function multiply(a, b) {
    return a * b;
}

class Calculator {
    constructor() {
        this.result = 0;
    }

    add(n) {
        this.result += n;
        return this;
    }
}`

	jsChunks := js.SplitText(jsCode)
	fmt.Printf("  %d chunks:\n", len(jsChunks))
	for i, c := range jsChunks {
		fmt.Printf("  [%d] %s\n", i+1, trunc(firstLine(c), 60))
	}

	fmt.Println("\n✅ Code splitters complete.")
}

func firstLine(s string) string {
	for i, ch := range s {
		if ch == '\n' {
			return s[:i]
		}
	}
	return s
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
