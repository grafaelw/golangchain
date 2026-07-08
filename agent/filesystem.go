package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"sync"

	"github.com/grafaelw/golangchain/tools"
)

// ---------------------------------------------------------------------------
// FileSystem — virtual filesystem for agents
// ---------------------------------------------------------------------------

// FileSystem provides a virtual filesystem for agents. The agent can read,
// write, list, and delete files. All operations are scoped to the agent's
// state so they persist across turns but are isolated from the host filesystem.
type FileSystem struct {
	mu    sync.RWMutex
	files map[string]string
}

// NewFileSystem creates a FileSystem.
func NewFileSystem() *FileSystem {
	return &FileSystem{files: make(map[string]string)}
}

// Read returns the content of the named file.
func (fs *FileSystem) Read(filename string) (string, error) {
	fs.mu.RLock()
	defer fs.mu.RUnlock()
	content, ok := fs.files[filename]
	if !ok {
		return "", fmt.Errorf("filesystem: file %q not found", filename)
	}
	return content, nil
}

// Write creates or overwrites a file with the given content.
func (fs *FileSystem) Write(filename, content string) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	fs.files[filename] = content
	return nil
}

// List returns the names of all files, sorted.
func (fs *FileSystem) List() ([]string, error) {
	fs.mu.RLock()
	defer fs.mu.RUnlock()
	names := make([]string, 0, len(fs.files))
	for name := range fs.files {
		names = append(names, name)
	}
	sort.Strings(names)
	return names, nil
}

// Delete removes the named file.
func (fs *FileSystem) Delete(filename string) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	if _, ok := fs.files[filename]; !ok {
		return fmt.Errorf("filesystem: file %q not found", filename)
	}
	delete(fs.files, filename)
	return nil
}

// AsTools returns the filesystem operations as a []tools.Tool slice so the
// agent can use them as regular tools.
func (fs *FileSystem) AsTools() []tools.Tool {
	return []tools.Tool{
		tools.NewFuncTool(
			"read_file",
			"Reads the content of a file from the virtual filesystem. Input: JSON {\"filename\":\"...\"}.",
			json.RawMessage(`{"type":"object","properties":{"filename":{"type":"string","description":"The name of the file to read."}},"required":["filename"]}`),
			func(_ context.Context, input string) (string, error) {
				var args struct {
					Filename string `json:"filename"`
				}
				if err := json.Unmarshal([]byte(input), &args); err != nil {
					return "", fmt.Errorf("filesystem read_file: %w", err)
				}
				return fs.Read(args.Filename)
			},
		),
		tools.NewFuncTool(
			"write_file",
			"Writes content to a file in the virtual filesystem. Input: JSON {\"filename\":\"...\",\"content\":\"...\"}.",
			json.RawMessage(`{"type":"object","properties":{"filename":{"type":"string","description":"The name of the file to write."},"content":{"type":"string","description":"The content to write to the file."}},"required":["filename","content"]}`),
			func(_ context.Context, input string) (string, error) {
				var args struct {
					Filename string `json:"filename"`
					Content  string `json:"content"`
				}
				if err := json.Unmarshal([]byte(input), &args); err != nil {
					return "", fmt.Errorf("filesystem write_file: %w", err)
				}
				if err := fs.Write(args.Filename, args.Content); err != nil {
					return "", err
				}
				return fmt.Sprintf("Wrote %d bytes to %q", len(args.Content), args.Filename), nil
			},
		),
		tools.NewFuncTool(
			"list_files",
			"Lists all files in the virtual filesystem. Input: any JSON object or empty string.",
			json.RawMessage(`{"type":"object","properties":{},"required":[]}`),
			func(_ context.Context, input string) (string, error) {
				names, err := fs.List()
				if err != nil {
					return "", err
				}
				if len(names) == 0 {
					return "(no files)", nil
				}
				b, _ := json.Marshal(names)
				return string(b), nil
			},
		),
		tools.NewFuncTool(
			"delete_file",
			"Deletes a file from the virtual filesystem. Input: JSON {\"filename\":\"...\"}.",
			json.RawMessage(`{"type":"object","properties":{"filename":{"type":"string","description":"The name of the file to delete."}},"required":["filename"]}`),
			func(_ context.Context, input string) (string, error) {
				var args struct {
					Filename string `json:"filename"`
				}
				if err := json.Unmarshal([]byte(input), &args); err != nil {
					return "", fmt.Errorf("filesystem delete_file: %w", err)
				}
				if err := fs.Delete(args.Filename); err != nil {
					return "", err
				}
				return fmt.Sprintf("Deleted %q", args.Filename), nil
			},
		),
	}
}
