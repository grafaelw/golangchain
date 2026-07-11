// Example: sql_chain
//
// Demonstrates the SQLDatabaseChain which converts natural language questions
// into SQL, executes them, and returns an answer.
//
// This example uses an in-memory mock database so no real SQL server is
// needed. For production use, swap the mock with StdSQLDatabase wrapping
// a *sql.DB.
//
// Highlights:
//
//   - SQLDatabase interface: Tables(), Schema(), Query(), Dialect()
//
//   - SQLDatabaseChain: question → SQL generation → query execution → answer
//
//   - StdSQLDatabase adapter for database/sql.*DB (PostgreSQL, MySQL, SQLite)
//
//   - Output includes question, SQL, raw results, and natural language answer
//
//     Run this example with:
//     go run ./examples/chains/sql
package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/grafaelw/golangchain/chain"
)

var _ chain.SQLDatabase = (*mockDB)(nil)

func main() {
	ctx := context.Background()

	// -------------------------------------------------------------------------
	// 1. Create a mock database with a users table
	// -------------------------------------------------------------------------
	db := &mockDB{
		dialect: "postgres",
		tables:  []string{"users", "orders"},
		schemas: map[string]string{
			"users": `CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  email TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
)`,
			"orders": `CREATE TABLE orders (
  id SERIAL PRIMARY KEY,
  user_id INTEGER REFERENCES users(id),
  amount DECIMAL(10,2),
  status TEXT
)`,
		},
		data: map[string][]map[string]string{
			"SELECT id, name, email, created_at FROM users ORDER BY created_at DESC LIMIT 5": {
				{"id": "1", "name": "Alice", "email": "alice@example.com", "created_at": "2024-01-15"},
				{"id": "2", "name": "Bob", "email": "bob@example.com", "created_at": "2024-02-20"},
				{"id": "3", "name": "Carol", "email": "carol@example.com", "created_at": "2024-03-10"},
			},
			"SELECT COUNT(*) as count FROM users": {
				{"count": "42"},
			},
			"SELECT status, COUNT(*) as cnt, SUM(amount) as total FROM orders GROUP BY status": {
				{"status": "completed", "cnt": "150", "total": "45200.50"},
				{"status": "pending", "cnt": "23", "total": "3150.75"},
				{"status": "cancelled", "cnt": "8", "total": "880.00"},
			},
		},
	}

	// -------------------------------------------------------------------------
	// 2. Inspect the database schema
	// -------------------------------------------------------------------------
	fmt.Println("--- 1. Database schema ---")
	tables, _ := db.Tables(ctx)
	for _, t := range tables {
		schema, _ := db.Schema(ctx, t)
		fmt.Printf("  %s\n", schema)
	}

	// -------------------------------------------------------------------------
	// 3. Execute SQL queries and show results
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 2. Query execution ---")
	queries := []string{
		"SELECT id, name, email, created_at FROM users ORDER BY created_at DESC LIMIT 5",
		"SELECT COUNT(*) as count FROM users",
		"SELECT status, COUNT(*) as cnt, SUM(amount) as total FROM orders GROUP BY status",
	}
	for _, q := range queries {
		rows, _ := db.Query(ctx, q)
		fmt.Printf("  SQL: %s\n", trunc(q, 60))
		fmt.Printf("  Rows: %v\n", rows)
	}

	// -------------------------------------------------------------------------
	// 4. SQLDatabaseChain integration pattern
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 3. SQLDatabaseChain pattern ---")
	// In a real app with an LLM provider, you would write:
	//
	//     model, _ := openai.New(openai.WithAPIKey(key), openai.WithModel("gpt-4o"))
	//     sqlChain := chain.NewSQLDatabaseChain(db, model)
	//     result, _ := sqlChain.Invoke(ctx, "How many users do we have?")
	//     // result["question"] = "How many users do we have?"
	//     // result["sql"] = "SELECT COUNT(*) FROM users"
	//     // result["result"] = [{"count": "42"}]
	//     // result["answer"] = "There are 42 users."

	fmt.Println("  SQLDatabaseChain(db, model) → Invoke(ctx, question)")
	fmt.Println("  Returns: {question, sql, result, answer}")
	fmt.Println("  Streaming: falls back to Invoke (single chunk)")

	// -------------------------------------------------------------------------
	// 5. StdSQLDatabase adapters by dialect
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 4. StdSQLDatabase dialects ---")
	dialects := []string{"postgres", "mysql", "sqlite"}
	for _, d := range dialects {
		db.dialect = d
		tables, _ := db.Tables(ctx)
		fmt.Printf("  %-10s → %d tables: %v\n", d, len(tables), tables)
	}

	fmt.Println("\n✅ SQL chain examples complete.")
}

// mockDB implements chain.SQLDatabase with in-memory data.
type mockDB struct {
	dialect string
	tables  []string
	schemas map[string]string
	data    map[string][]map[string]string
}

func (m *mockDB) Dialect() string { return m.dialect }

func (m *mockDB) Tables(_ context.Context) ([]string, error) {
	return m.tables, nil
}

func (m *mockDB) Schema(_ context.Context, table string) (string, error) {
	if s, ok := m.schemas[table]; ok {
		return s, nil
	}
	return "", fmt.Errorf("table %q not found", table)
}

func (m *mockDB) Query(_ context.Context, sql string) ([]map[string]string, error) {
	// Normalise whitespace for lookup
	norm := strings.Join(strings.Fields(sql), " ")
	if rows, ok := m.data[norm]; ok {
		return rows, nil
	}
	// Fallback: try prefix match
	for k, v := range m.data {
		if strings.HasPrefix(norm, k) {
			return v, nil
		}
	}
	return []map[string]string{{"info": "mock: no preloaded data for this query"}}, nil
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
