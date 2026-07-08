package middleware

import (
	"context"
	"fmt"
	"strings"

	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// TodoListMiddleware
// ---------------------------------------------------------------------------

// TodoListMiddleware injects a todo list into the system prompt and updates it
// based on agent actions and observations.
type TodoListMiddleware struct {
	NoOpMiddleware
	todos []todoItem
}

type todoItem struct {
	Text string
	Done bool
}

// NewTodoListMiddleware constructs a TodoListMiddleware.
func NewTodoListMiddleware() *TodoListMiddleware {
	return &TodoListMiddleware{}
}

func (t *TodoListMiddleware) Name() string { return "TodoList" }

func (t *TodoListMiddleware) BeforeModel(ctx context.Context, messages []schema.Message, steps []schema.AgentStep) ([]schema.Message, error) {
	if len(t.todos) == 0 && len(steps) == 0 {
		return messages, nil
	}

	tl := t.render()
	if tl == "" {
		return messages, nil
	}

	var result []schema.Message
	injected := false
	for i, m := range messages {
		if m.Role == schema.RoleSystem && !injected {
			result = append(result, schema.Message{
				Role:    schema.RoleSystem,
				Content: m.Content + "\n\n" + tl,
			})
			injected = true
			continue
		}
		if i == 0 && !injected {
			result = append(result, schema.NewSystemMessage(tl))
			injected = true
		}
		result = append(result, m)
	}
	if !injected {
		result = append([]schema.Message{schema.NewSystemMessage(tl)}, result...)
	}
	return result, nil
}

func (t *TodoListMiddleware) AfterTool(ctx context.Context, toolName string, output string) (string, error) {
	t.markDone(toolName, output)
	return output, nil
}

// AddTodo appends a new todo item.
func (t *TodoListMiddleware) AddTodo(text string) {
	t.todos = append(t.todos, todoItem{Text: text})
}

// ClearTodos clears all todo items.
func (t *TodoListMiddleware) ClearTodos() {
	t.todos = nil
}

func (t *TodoListMiddleware) markDone(toolName, output string) {
	for i := range t.todos {
		if !t.todos[i].Done && strings.Contains(strings.ToLower(output), strings.ToLower(t.todos[i].Text)) {
			t.todos[i].Done = true
		}
	}
}

func (t *TodoListMiddleware) render() string {
	var sb strings.Builder
	sb.WriteString("Todo list:\n")
	for _, item := range t.todos {
		status := "[ ]"
		if item.Done {
			status = "[x]"
		}
		fmt.Fprintf(&sb, "- %s %s\n", status, item.Text)
	}
	return sb.String()
}
