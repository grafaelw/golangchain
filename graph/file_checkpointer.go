// This file adds a FileCheckpointer that persists graph state to disk as
// JSON files (one directory per thread, one file per checkpoint), suitable
// for local development and single-node deployments.
package graph

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// FileCheckpointer
// ---------------------------------------------------------------------------

// FileCheckpointer stores checkpoints as JSON files under Dir/<threadID>/.
// Each save writes a new file named "<epoch-nanos>.json" so history is
// preserved and List returns chronological order.
//
// Safe for concurrent use within a single process; concurrent processes
// writing to the same directory should stagger their thread IDs.
type FileCheckpointer[S any] struct {
	Dir string
	mu  sync.Mutex
}

// NewFileCheckpointer creates a FileCheckpointer, ensuring dir exists.
func NewFileCheckpointer[S any](dir string) (*FileCheckpointer[S], error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("graph: file checkpointer: %w", err)
	}
	return &FileCheckpointer[S]{Dir: dir}, nil
}

func (f *FileCheckpointer[S]) threadDir(threadID string) string {
	return filepath.Join(f.Dir, sanitizeThreadID(threadID))
}

func (f *FileCheckpointer[S]) Save(_ context.Context, threadID string, cp Checkpoint[S]) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	dir := f.threadDir(threadID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("graph: file checkpointer: mkdir: %w", err)
	}
	data, err := json.Marshal(cp)
	if err != nil {
		return fmt.Errorf("graph: file checkpointer: marshal: %w", err)
	}
	name := fmt.Sprintf("%020d.json", time.Now().UnixNano())
	tmp := filepath.Join(dir, name+".tmp")
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("graph: file checkpointer: write: %w", err)
	}
	return os.Rename(tmp, filepath.Join(dir, name))
}

func (f *FileCheckpointer[S]) Load(_ context.Context, threadID string) (*Checkpoint[S], error) {
	entries, err := f.listFiles(threadID)
	if err != nil || len(entries) == 0 {
		return nil, err
	}
	data, err := os.ReadFile(filepath.Join(f.threadDir(threadID), entries[len(entries)-1]))
	if err != nil {
		return nil, fmt.Errorf("graph: file checkpointer: read: %w", err)
	}
	var cp Checkpoint[S]
	if err := json.Unmarshal(data, &cp); err != nil {
		return nil, fmt.Errorf("graph: file checkpointer: unmarshal: %w", err)
	}
	return &cp, nil
}

func (f *FileCheckpointer[S]) List(_ context.Context, threadID string) ([]Checkpoint[S], error) {
	entries, err := f.listFiles(threadID)
	if err != nil {
		return nil, err
	}
	out := make([]Checkpoint[S], 0, len(entries))
	for _, name := range entries {
		data, err := os.ReadFile(filepath.Join(f.threadDir(threadID), name))
		if err != nil {
			return out, err
		}
		var cp Checkpoint[S]
		if err := json.Unmarshal(data, &cp); err != nil {
			return out, err
		}
		out = append(out, cp)
	}
	return out, nil
}

func (f *FileCheckpointer[S]) Delete(_ context.Context, threadID string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	return os.RemoveAll(f.threadDir(threadID))
}

func (f *FileCheckpointer[S]) listFiles(threadID string) ([]string, error) {
	entries, err := os.ReadDir(f.threadDir(threadID))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("graph: file checkpointer: read dir: %w", err)
	}
	var names []string
	for _, e := range entries {
		if e.IsDir() || strings.HasSuffix(e.Name(), ".tmp") {
			continue
		}
		names = append(names, e.Name())
	}
	sort.Strings(names)
	return names, nil
}

func sanitizeThreadID(id string) string {
	var b strings.Builder
	for _, r := range id {
		switch {
		case r >= 'a' && r <= 'z',
			r >= 'A' && r <= 'Z',
			r >= '0' && r <= '9',
			r == '-', r == '_', r == '.':
			b.WriteRune(r)
		default:
			b.WriteRune('_')
		}
	}
	if b.Len() == 0 {
		return "default"
	}
	return b.String()
}
