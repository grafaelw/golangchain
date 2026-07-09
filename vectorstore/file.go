// This file adds a FileVectorStore that persists InMemoryVectorStore
// contents to a JSON file on disk. It is intended for local development
// and single-node deployments; for production workloads consider a
// dedicated engine (Qdrant, pgvector, Weaviate, …).

package vectorstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/grafaelw/golangchain/embeddings"
	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// FileVectorStore
// ---------------------------------------------------------------------------

// FileVectorStore is an InMemoryVectorStore backed by a JSON file on disk.
// AddDocuments and Delete auto-persist; the file is created on first write.
//
//	store, _ := vectorstore.NewFileVectorStore("./index.json", embedder)
//	_ = store.AddDocuments(ctx, docs) // written through to disk
//	_ = store.Flush()                  // explicit save (optional)
type FileVectorStore struct {
	*InMemoryVectorStore
	Path      string
	writeLock sync.Mutex
}

// NewFileVectorStore opens or creates a FileVectorStore at path.
// If the file exists, its contents (including pre-computed vectors) are loaded
// and embedder is not called for those documents.
func NewFileVectorStore(path string, embedder embeddings.Embedder) (*FileVectorStore, error) {
	s := &FileVectorStore{
		InMemoryVectorStore: NewInMemoryVectorStore(embedder),
		Path:                path,
	}
	if err := s.load(); err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("vectorstore: load %q: %w", path, err)
	}
	return s, nil
}

// AddDocuments embeds new documents, appends them, and persists to disk.
func (s *FileVectorStore) AddDocuments(ctx context.Context, docs []schema.Document) error {
	if err := s.InMemoryVectorStore.AddDocuments(ctx, docs); err != nil {
		return err
	}
	return s.Flush()
}

// Delete removes matching docs from memory and persists.
func (s *FileVectorStore) Delete(ctx context.Context, ids []string) error {
	if err := s.InMemoryVectorStore.Delete(ctx, ids); err != nil {
		return err
	}
	return s.Flush()
}

// Flush writes the current in-memory index to disk atomically (write+rename).
func (s *FileVectorStore) Flush() error {
	s.writeLock.Lock()
	defer s.writeLock.Unlock()

	snapshot := s.snapshot()
	data, err := json.Marshal(snapshot)
	if err != nil {
		return fmt.Errorf("vectorstore: marshal: %w", err)
	}
	if dir := filepath.Dir(s.Path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("vectorstore: mkdir: %w", err)
		}
	}
	tmp := s.Path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("vectorstore: write: %w", err)
	}
	return os.Rename(tmp, s.Path)
}

// Close flushes and clears the in-memory index. Implements io.Closer.
func (s *FileVectorStore) Close() error {
	if err := s.Flush(); err != nil {
		return err
	}
	return s.InMemoryVectorStore.Close()
}

// ---------------------------------------------------------------------------
// On-disk format
// ---------------------------------------------------------------------------

type diskEntry struct {
	ID       string          `json:"id,omitempty"`
	Document schema.Document `json:"document"`
	Vector   []float64       `json:"vector"`
}

type diskFormat struct {
	Version int         `json:"version"`
	Entries []diskEntry `json:"entries"`
}

func (s *FileVectorStore) snapshot() diskFormat {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := diskFormat{Version: 1, Entries: make([]diskEntry, len(s.entries))}
	for i, e := range s.entries {
		out.Entries[i] = diskEntry{ID: e.id, Document: e.doc, Vector: e.vector}
	}
	return out
}

func (s *FileVectorStore) load() error {
	data, err := os.ReadFile(s.Path)
	if err != nil {
		return err
	}
	var d diskFormat
	if err := json.Unmarshal(data, &d); err != nil {
		return fmt.Errorf("unmarshal: %w", err)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries = s.entries[:0]
	for _, e := range d.Entries {
		s.entries = append(s.entries, entry{id: e.ID, doc: e.Document, vector: e.Vector})
	}
	return nil
}
