package chain

import (
	"context"
	"database/sql"
	"fmt"
	"strings"

	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/schema"
)

// SQLDatabase represents a queryable SQL database.
// Implementations wrap database/sql.*DB or any SQL-compatible backend.
type SQLDatabase interface {
	// Query executes a SQL query and returns rows as a slice of
	// string maps (column name → value). For SELECT queries each
	// element is one row; for non-SELECT the slice is empty.
	Query(ctx context.Context, sql string) ([]map[string]string, error)

	// Tables returns the list of table names in the database.
	Tables(ctx context.Context) ([]string, error)

	// Schema returns the CREATE TABLE DDL for a table, or an
	// informative description of columns and types.
	Schema(ctx context.Context, table string) (string, error)

	// Dialect returns the SQL dialect (e.g. "postgres", "mysql", "sqlite").
	Dialect() string
}

// SQLDatabaseChain converts natural language questions into SQL, executes
// them, and returns the answer.
//
//	db := sqlchain.NewStdSQLDatabase(sqlDB, "postgres")
//	chain := sqlchain.NewSQLDatabaseChain(db, model)
//	ans, _ := chain.Invoke(ctx, "How many users signed up today?")
type SQLDatabaseChain struct {
	DB         SQLDatabase
	LLM        llm.LLM
	LLMOptions []llm.Option
	Name       string
	TopK       int // number of rows to return in the answer (default 10)
}

// NewSQLDatabaseChain creates a SQL chain.
func NewSQLDatabaseChain(db SQLDatabase, model llm.LLM, opts ...llm.Option) *SQLDatabaseChain {
	return &SQLDatabaseChain{
		DB:         db,
		LLM:        model,
		LLMOptions: opts,
		Name:       "SQLDatabaseChain",
		TopK:       10,
	}
}

func (c *SQLDatabaseChain) Invoke(ctx context.Context, input any) (any, error) {
	question, err := extractQuestion(input)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", c.Name, err)
	}

	tables, err := c.DB.Tables(ctx)
	if err != nil {
		return nil, fmt.Errorf("%s: tables: %w", c.Name, err)
	}

	var schemaBuilder strings.Builder
	for _, t := range tables {
		schema, err := c.DB.Schema(ctx, t)
		if err != nil {
			continue
		}
		schemaBuilder.WriteString(schema)
		schemaBuilder.WriteString("\n\n")
	}
	schemaStr := strings.TrimSpace(schemaBuilder.String())

	genSQL, err := c.LLM.Generate(ctx, []schema.Message{
		schema.NewHumanMessage(fmt.Sprintf(sqlGenPrompt, c.DB.Dialect(), schemaStr, question)),
	}, c.LLMOptions...)
	if err != nil {
		return nil, fmt.Errorf("%s: generate sql: %w", c.Name, err)
	}

	rawSQL := extractSQL(genSQL.Text)
	rows, err := c.DB.Query(ctx, rawSQL)
	if err != nil {
		return nil, fmt.Errorf("%s: query: %w (SQL: %s)", c.Name, err, rawSQL)
	}

	rowsStr := formatRows(rows, c.TopK)

	gen, err := c.LLM.Generate(ctx, []schema.Message{
		schema.NewHumanMessage(fmt.Sprintf(sqlAnswerPrompt, question, rawSQL, rowsStr)),
	}, c.LLMOptions...)
	if err != nil {
		return nil, fmt.Errorf("%s: answer: %w", c.Name, err)
	}

	return map[string]any{
		"question": question,
		"sql":      rawSQL,
		"result":   rows,
		"answer":   strings.TrimSpace(gen.Text),
	}, nil
}

func (c *SQLDatabaseChain) Stream(ctx context.Context, input any) (<-chan schema.StreamChunk, error) {
	out, err := c.Invoke(ctx, input)
	ch := make(chan schema.StreamChunk, 1)
	if err != nil {
		ch <- schema.StreamChunk{Err: err}
		close(ch)
		return ch, nil
	}
	ch <- schema.StreamChunk{Value: out, Done: true}
	close(ch)
	return ch, nil
}

func (c *SQLDatabaseChain) Pipe(next Runnable) Runnable {
	return &pipeRunnable{first: c, second: next}
}

func (c *SQLDatabaseChain) Batch(ctx context.Context, inputs []any) ([]any, error) {
	return RunBatch(ctx, c, inputs)
}

const sqlGenPrompt = `You are a %s SQL expert. Given the database schema below, write a syntactically correct SQL query to answer the user's question. Return ONLY the SQL query, no explanation.

Schema:
%s

Question: %s

SQL:`

const sqlAnswerPrompt = `Given the following user question, corresponding SQL query, and SQL result, answer the user's question in natural language.

Question: %s
SQL query: %s
SQL result:
%s

Answer:`

func extractSQL(text string) string {
	text = strings.TrimSpace(text)
	text = strings.TrimPrefix(text, "```sql")
	text = strings.TrimPrefix(text, "```")
	text = strings.TrimSuffix(text, "```")
	return strings.TrimSpace(text)
}

func formatRows(rows []map[string]string, topK int) string {
	if len(rows) == 0 {
		return "(no results)"
	}
	if topK > 0 && len(rows) > topK {
		rows = rows[:topK]
	}
	var sb strings.Builder
	for i, row := range rows {
		sb.WriteString(fmt.Sprintf("[%d] ", i+1))
		first := true
		for k, v := range row {
			if !first {
				sb.WriteString(", ")
			}
			sb.WriteString(fmt.Sprintf("%s=%s", k, v))
			first = false
		}
		sb.WriteString("\n")
	}
	return strings.TrimSpace(sb.String())
}

// ---------------------------------------------------------------------------
// StdSQLDatabase — database/sql adapter
// ---------------------------------------------------------------------------

// StdSQLDatabase wraps a *sql.DB to implement SQLDatabase.
type StdSQLDatabase struct {
	db      *sql.DB
	dialect string
}

// NewStdSQLDatabase creates a SQLDatabase from a standard library *sql.DB.
func NewStdSQLDatabase(db *sql.DB, dialect string) *StdSQLDatabase {
	return &StdSQLDatabase{db: db, dialect: dialect}
}

func (d *StdSQLDatabase) Dialect() string { return d.dialect }

func (d *StdSQLDatabase) Tables(ctx context.Context) ([]string, error) {
	rows, err := d.db.QueryContext(ctx, d.tableQuery())
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var tables []string
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			return nil, err
		}
		tables = append(tables, name)
	}
	return tables, rows.Err()
}

func (d *StdSQLDatabase) tableQuery() string {
	switch d.dialect {
	case "postgres", "postgresql":
		return "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public'"
	case "mysql":
		return "SHOW TABLES"
	case "sqlite", "sqlite3":
		return "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
	default:
		return "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
	}
}

func (d *StdSQLDatabase) Schema(ctx context.Context, table string) (string, error) {
	switch d.dialect {
	case "sqlite", "sqlite3":
		rows, err := d.db.QueryContext(ctx, fmt.Sprintf("PRAGMA table_info(%s)", table))
		if err != nil {
			return "", err
		}
		defer rows.Close()
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("CREATE TABLE %s (\n", table))
		for rows.Next() {
			var cid int
			var name, colType string
			var notNull int
			var dflt sql.NullString
			var pk int
			if err := rows.Scan(&cid, &name, &colType, &notNull, &dflt, &pk); err != nil {
				return "", err
			}
			sb.WriteString(fmt.Sprintf("  %s %s", name, colType))
			if pk > 0 {
				sb.WriteString(" PRIMARY KEY")
			}
			if notNull > 0 {
				sb.WriteString(" NOT NULL")
			}
			sb.WriteString(",\n")
		}
		sb.WriteString(")")
		return sb.String(), rows.Err()
	default:
		rows, err := d.db.QueryContext(ctx,
			"SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name = $1",
			table)
		if err != nil {
			return "", err
		}
		defer rows.Close()
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("Table: %s\n", table))
		for rows.Next() {
			var col, dtype, nullable string
			if err := rows.Scan(&col, &dtype, &nullable); err != nil {
				return "", err
			}
			sb.WriteString(fmt.Sprintf("  %s %s nullable=%s\n", col, dtype, nullable))
		}
		return strings.TrimSpace(sb.String()), rows.Err()
	}
}

func (d *StdSQLDatabase) Query(ctx context.Context, query string) ([]map[string]string, error) {
	rows, err := d.db.QueryContext(ctx, query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	cols, err := rows.Columns()
	if err != nil {
		return nil, err
	}

	var results []map[string]string
	for rows.Next() {
		values := make([]any, len(cols))
		valuePtrs := make([]any, len(cols))
		for i := range cols {
			valuePtrs[i] = &values[i]
		}
		if err := rows.Scan(valuePtrs...); err != nil {
			return nil, err
		}
		row := make(map[string]string, len(cols))
		for i, col := range cols {
			row[col] = fmt.Sprintf("%v", values[i])
		}
		results = append(results, row)
	}
	return results, rows.Err()
}
