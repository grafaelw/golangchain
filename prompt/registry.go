package prompt

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"sync"
	"time"
)

// PromptEntry is a versioned prompt in the registry.
type PromptEntry struct {
	Name        string    `json:"name"`
	Version     int       `json:"version"`
	Template    string    `json:"template"`
	Description string    `json:"description,omitempty"`
	CommitMsg   string    `json:"commit_msg,omitempty"`
	CreatedAt   time.Time `json:"created_at"`
	Tags        []string  `json:"tags,omitempty"`
}

// Registry stores and versions prompts.
type Registry struct {
	mu      sync.RWMutex
	prompts map[string][]PromptEntry
}

// NewRegistry creates an empty prompt registry.
func NewRegistry() *Registry {
	return &Registry{prompts: make(map[string][]PromptEntry)}
}

// Register adds or creates a new version of a prompt.
// Returns the new version number.
func (r *Registry) Register(name, template, commitMsg string) (int, error) {
	if name == "" {
		return 0, fmt.Errorf("prompt: register: name must not be empty")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	entries := r.prompts[name]
	version := 1
	if len(entries) > 0 {
		version = entries[len(entries)-1].Version + 1
	}
	entry := PromptEntry{
		Name:      name,
		Version:   version,
		Template:  template,
		CommitMsg: commitMsg,
		CreatedAt: time.Now().UTC(),
	}
	r.prompts[name] = append(r.prompts[name], entry)
	return version, nil
}

// Get retrieves the latest version of a prompt.
func (r *Registry) Get(name string) (*PromptEntry, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	entries, ok := r.prompts[name]
	if !ok || len(entries) == 0 {
		return nil, fmt.Errorf("prompt: %q not found", name)
	}
	latest := entries[len(entries)-1]
	return &latest, nil
}

// GetVersion retrieves a specific version.
func (r *Registry) GetVersion(name string, version int) (*PromptEntry, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	entries, ok := r.prompts[name]
	if !ok {
		return nil, fmt.Errorf("prompt: %q not found", name)
	}
	idx := sort.Search(len(entries), func(i int) bool { return entries[i].Version >= version })
	if idx < len(entries) && entries[idx].Version == version {
		return &entries[idx], nil
	}
	return nil, fmt.Errorf("prompt: %q version %d not found", name, version)
}

// List returns all prompt names.
func (r *Registry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.prompts))
	for name := range r.prompts {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// History returns all versions of a prompt.
func (r *Registry) History(name string) []PromptEntry {
	r.mu.RLock()
	defer r.mu.RUnlock()
	entries := r.prompts[name]
	out := make([]PromptEntry, len(entries))
	copy(out, entries)
	return out
}

// SaveJSONL writes the registry to a JSONL file.
func (r *Registry) SaveJSONL(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("prompt: save %q: %w", path, err)
	}
	defer func() { _ = f.Close() }()
	r.mu.RLock()
	defer r.mu.RUnlock()

	enc := json.NewEncoder(f)
	for _, entries := range r.prompts {
		for _, e := range entries {
			if err := enc.Encode(e); err != nil {
				return fmt.Errorf("prompt: save: %w", err)
			}
		}
	}
	return nil
}

// LoadJSONL reads a registry from a JSONL file.
func (r *Registry) LoadJSONL(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("prompt: load %q: %w", path, err)
	}
	defer func() { _ = f.Close() }()

	r.mu.Lock()
	defer r.mu.Unlock()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		var entry PromptEntry
		if err := json.Unmarshal(scanner.Bytes(), &entry); err != nil {
			return fmt.Errorf("prompt: load line: %w", err)
		}
		r.prompts[entry.Name] = append(r.prompts[entry.Name], entry)
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("prompt: load: %w", err)
	}
	return nil
}
