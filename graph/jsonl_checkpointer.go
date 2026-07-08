package graph

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// JSONLCheckpointer stores checkpoints as JSONL files.
// Each thread gets one .jsonl file; saves append a line; loads read the last
// valid JSON line. This is simpler and faster than one-file-per-checkpoint
// for long-running agents with many iterations.
//
// Safe for concurrent use within a single process.
type JSONLCheckpointer[S any] struct {
	Dir string
	mu  sync.Mutex
}

// NewJSONLCheckpointer creates a JSONLCheckpointer, ensuring dir exists.
func NewJSONLCheckpointer[S any](dir string) (*JSONLCheckpointer[S], error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("graph: jsonl checkpointer: %w", err)
	}
	return &JSONLCheckpointer[S]{Dir: dir}, nil
}

func (j *JSONLCheckpointer[S]) filePath(threadID string) string {
	return filepath.Join(j.Dir, sanitizeThreadID(threadID)+".jsonl")
}

func (j *JSONLCheckpointer[S]) Save(_ context.Context, threadID string, cp Checkpoint[S]) error {
	j.mu.Lock()
	defer j.mu.Unlock()

	data, err := json.Marshal(cp)
	if err != nil {
		return fmt.Errorf("graph: jsonl checkpointer: marshal: %w", err)
	}

	f, err := os.OpenFile(j.filePath(threadID), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("graph: jsonl checkpointer: open: %w", err)
	}
	defer func() { _ = f.Close() }()

	if _, err := f.Write(append(data, '\n')); err != nil {
		return fmt.Errorf("graph: jsonl checkpointer: write: %w", err)
	}
	return nil
}

func (j *JSONLCheckpointer[S]) Load(_ context.Context, threadID string) (*Checkpoint[S], error) {
	f, err := os.Open(j.filePath(threadID))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("graph: jsonl checkpointer: open: %w", err)
	}
	defer func() { _ = f.Close() }()

	var lastLine string
	scanner := bufio.NewScanner(f)
	scanner.Buffer(nil, 10<<20) // 10 MiB max line
	for scanner.Scan() {
		lastLine = scanner.Text()
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("graph: jsonl checkpointer: scan: %w", err)
	}
	if lastLine == "" {
		return nil, nil
	}

	var cp Checkpoint[S]
	if err := json.Unmarshal([]byte(lastLine), &cp); err != nil {
		return nil, fmt.Errorf("graph: jsonl checkpointer: unmarshal: %w", err)
	}
	return &cp, nil
}

func (j *JSONLCheckpointer[S]) List(_ context.Context, threadID string) ([]Checkpoint[S], error) {
	f, err := os.Open(j.filePath(threadID))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("graph: jsonl checkpointer: open: %w", err)
	}
	defer func() { _ = f.Close() }()

	var cps []Checkpoint[S]
	scanner := bufio.NewScanner(f)
	scanner.Buffer(nil, 10<<20)
	for scanner.Scan() {
		var cp Checkpoint[S]
		if err := json.Unmarshal(scanner.Bytes(), &cp); err != nil {
			continue
		}
		cps = append(cps, cp)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("graph: jsonl checkpointer: scan: %w", err)
	}
	return cps, nil
}

func (j *JSONLCheckpointer[S]) Delete(_ context.Context, threadID string) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	err := os.Remove(j.filePath(threadID))
	if os.IsNotExist(err) {
		return nil
	}
	return err
}
