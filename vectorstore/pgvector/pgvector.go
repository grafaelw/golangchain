// Package pgvector provides a VectorStore backed by PostgreSQL with the
// pgvector extension. It uses the cosine distance operator (<=>) for
// similarity search.
//
// Requires a PostgreSQL database with the pgvector extension enabled:
//
//	CREATE EXTENSION IF NOT EXISTS vector;
//
// Usage:
//
//	db, _ := sql.Open("postgres", "postgres://...")
//	store, _ := pgvector.New(db, "my_table", 1536, embedder)
//	_ = store.AddDocuments(ctx, docs)
//	results, _ := store.SimilaritySearch(ctx, "query", 5)
package pgvector

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/grafaelw/golangchain/embeddings"
	"github.com/grafaelw/golangchain/schema"
)

// Store is a VectorStore backed by a PostgreSQL table with the pgvector extension.
type Store struct {
	db       *sql.DB
	table    string
	dim      int
	embedder embeddings.Embedder
}

// New creates a pgvector-backed vector store. The table is created
// automatically if it does not exist.
func New(db *sql.DB, table string, dim int, embedder embeddings.Embedder) (*Store, error) {
	s := &Store{db: db, table: table, dim: dim, embedder: embedder}
	if err := s.ensureTable(context.Background()); err != nil {
		return nil, fmt.Errorf("pgvector: ensure table: %w", err)
	}
	return s, nil
}

func (s *Store) ensureTable(ctx context.Context) error {
	_, err := s.db.ExecContext(ctx, fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS %s (
			id          TEXT PRIMARY KEY,
			page_content TEXT NOT NULL,
			metadata    JSONB DEFAULT '{}',
			embedding   vector(%d)
		)`, s.table, s.dim))
	return err
}

// AddDocuments embeds and indexes a batch of documents.
func (s *Store) AddDocuments(ctx context.Context, docs []schema.Document) error {
	texts := make([]string, len(docs))
	for i, d := range docs {
		texts[i] = d.PageContent
	}

	vectors, err := s.embedder.EmbedDocuments(ctx, texts)
	if err != nil {
		return fmt.Errorf("pgvector: embed: %w", err)
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("pgvector: begin tx: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	stmt, err := tx.PrepareContext(ctx, fmt.Sprintf(`
		INSERT INTO %s (id, page_content, metadata, embedding)
		VALUES ($1, $2, $3, $4)
		ON CONFLICT (id) DO UPDATE SET
			page_content = EXCLUDED.page_content,
			metadata    = EXCLUDED.metadata,
			embedding   = EXCLUDED.embedding
	`, s.table))
	if err != nil {
		return fmt.Errorf("pgvector: prepare: %w", err)
	}
	defer func() { _ = stmt.Close() }()

	for i, doc := range docs {
		id := fmt.Sprintf("%d", i)
		if v, ok := doc.Metadata["id"]; ok {
			id = fmt.Sprint(v)
		}
		meta, _ := json.Marshal(doc.Metadata)
		vec := floatsToLiteral(vectors[i])
		if _, err := stmt.ExecContext(ctx, id, doc.PageContent, meta, vec); err != nil {
			return fmt.Errorf("pgvector: insert: %w", err)
		}
	}
	return tx.Commit()
}

// SimilaritySearch returns the k most similar documents using cosine distance.
func (s *Store) SimilaritySearch(ctx context.Context, query string, k int) ([]schema.Document, error) {
	qv, err := s.embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("pgvector: embed query: %w", err)
	}
	return s.SimilaritySearchByVector(ctx, qv, k)
}

// SimilaritySearchByVector returns the k most similar documents for a vector.
func (s *Store) SimilaritySearchByVector(ctx context.Context, vector []float64, k int) ([]schema.Document, error) {
	vec := floatsToLiteral(vector)
	rows, err := s.db.QueryContext(ctx, fmt.Sprintf(`
		SELECT page_content, metadata, 1.0 - (embedding <=> '%s') AS score
		FROM %s
		ORDER BY embedding <=> '%s'
		LIMIT $1
	`, vec, s.table, vec), k)
	if err != nil {
		return nil, fmt.Errorf("pgvector: search: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var docs []schema.Document
	for rows.Next() {
		var content string
		var metaBytes []byte
		var score float64
		if err := rows.Scan(&content, &metaBytes, &score); err != nil {
			return nil, fmt.Errorf("pgvector: scan: %w", err)
		}
		meta := make(map[string]any)
		_ = json.Unmarshal(metaBytes, &meta)
		docs = append(docs, schema.Document{
			PageContent: content,
			Metadata:    meta,
			Score:       score,
		})
	}
	return docs, rows.Err()
}

// Delete removes documents matching the given IDs.
func (s *Store) Delete(ctx context.Context, ids []string) error {
	if len(ids) == 0 {
		return nil
	}
	placeholders := make([]string, len(ids))
	args := make([]any, len(ids))
	for i, id := range ids {
		placeholders[i] = fmt.Sprintf("$%d", i+1)
		args[i] = id
	}
	_, err := s.db.ExecContext(ctx, fmt.Sprintf(
		"DELETE FROM %s WHERE id IN (%s)",
		s.table, strings.Join(placeholders, ",")),
		args...)
	return err
}

// Close closes the underlying database connection pool.
func (s *Store) Close() error {
	return s.db.Close()
}

// floatsToLiteral converts []float64 to a pgvector-compatible string constant.
func floatsToLiteral(v []float64) string {
	parts := make([]string, len(v))
	for i, f := range v {
		parts[i] = fmt.Sprintf("%g", f)
	}
	return "[" + strings.Join(parts, ",") + "]"
}
