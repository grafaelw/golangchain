// Package prompt provides prompt templates for golangchain.
//
// Template variables use Go's text/template syntax: {{.VarName}}.
//
// # Types
//
//   - [PromptTemplate] — single string template with variable substitution
//   - [ChatPromptTemplate] — ordered list of message slots and history placeholders
//   - [FewShotPromptTemplate] — automatically prepends examples before the query
//   - [MessagePlaceholder] — injects a []schema.Message slice (e.g. history) into a chat template
//
// # Example
//
//	chat := prompt.MustNewChatPromptTemplate(
//	    prompt.MustSystem("You are an expert in {{.Topic}}."),
//	    prompt.NewMessagePlaceholder("history"),
//	    prompt.MustHuman("{{.Question}}"),
//	)
//	msgs, _ := chat.FormatMessages(map[string]any{
//	    "Topic":    "Go programming",
//	    "history":  previousMessages,
//	    "Question": "What are goroutines?",
//	})
package prompt
