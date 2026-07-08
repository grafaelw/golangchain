// This file adds two additional memory backends:
//
//   - FileChatHistory:     persists a ConversationBufferMemory to disk as JSON.
//   - VectorStoreMemory:   retrieves the most relevant past turns from a
//                          vector store (semantic long-term memory).
package memory

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/grafaelw/golangchain/schema"
	"github.com/grafaelw/golangchain/vectorstore"
)

// ---------------------------------------------------------------------------
// FileChatHistory
// ---------------------------------------------------------------------------

// FileChatHistory is a ConversationBufferMemory that reads/writes its
// history to a JSON file. Suitable for CLI sessions and single-node apps.
type FileChatHistory struct {
	*ConversationBufferMemory
	Path      string
	writeLock sync.Mutex
}

// NewFileChatHistory opens (or creates) a chat log at path.
func NewFileChatHistory(path string) (*FileChatHistory, error) {
	m := &FileChatHistory{
		ConversationBufferMemory: NewConversationBufferMemory(),
		Path:                     path,
	}
	if err := m.load(); err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("memory: load %q: %w", path, err)
	}
	return m, nil
}

// SaveContext appends the turn and flushes to disk.
func (m *FileChatHistory) SaveContext(ctx context.Context, humanInput, aiOutput string) error {
	if err := m.ConversationBufferMemory.SaveContext(ctx, humanInput, aiOutput); err != nil {
		return err
	}
	return m.flush()
}

// Clear wipes memory and truncates the file.
func (m *FileChatHistory) Clear(ctx context.Context) error {
	if err := m.ConversationBufferMemory.Clear(ctx); err != nil {
		return err
	}
	return m.flush()
}

func (m *FileChatHistory) flush() error {
	m.writeLock.Lock()
	defer m.writeLock.Unlock()

	msgs := m.Messages()
	data, err := json.Marshal(msgs)
	if err != nil {
		return fmt.Errorf("memory: marshal: %w", err)
	}
	if dir := filepath.Dir(m.Path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("memory: mkdir: %w", err)
		}
	}
	tmp := m.Path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("memory: write: %w", err)
	}
	return os.Rename(tmp, m.Path)
}

func (m *FileChatHistory) load() error {
	data, err := os.ReadFile(m.Path)
	if err != nil {
		return err
	}
	var msgs []schema.Message
	if err := json.Unmarshal(data, &msgs); err != nil {
		return fmt.Errorf("unmarshal: %w", err)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.messages = msgs
	return nil
}

// ---------------------------------------------------------------------------
// VectorStoreMemory
// ---------------------------------------------------------------------------

// VectorStoreMemory indexes every completed turn as a Document in a vector
// store, and on LoadMemoryVariables returns the top-K turns most relevant
// to the caller-supplied query. Enables "infinite" long-term memory that
// scales beyond a fixed context window.
//
// Because Memory.LoadMemoryVariables receives no query, the last human input
// registered via SaveContext or Query is used as the retrieval query.
type VectorStoreMemory struct {
	mu         sync.Mutex
	store      vectorstore.VectorStore
	K          int
	HistoryKey string
	lastQuery  string
}

// NewVectorStoreMemory wraps a vector store as a semantic memory that
// returns the top-K most relevant past turns.
func NewVectorStoreMemory(store vectorstore.VectorStore, k int) *VectorStoreMemory {
	return &VectorStoreMemory{store: store, K: k, HistoryKey: "history"}
}

// SetQuery overrides the retrieval query used by the next LoadMemoryVariables.
func (m *VectorStoreMemory) SetQuery(q string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.lastQuery = q
}

func (m *VectorStoreMemory) Messages() []schema.Message {
	// VectorStoreMemory does not maintain an ordered chat log; callers should
	// use LoadMemoryVariables to get a query-relevant slice.
	return nil
}

func (m *VectorStoreMemory) LoadMemoryVariables(ctx context.Context) (map[string]any, error) {
	m.mu.Lock()
	q := m.lastQuery
	m.mu.Unlock()

	if q == "" {
		return map[string]any{m.HistoryKey: []schema.Message{}}, nil
	}
	docs, err := m.store.SimilaritySearch(ctx, q, m.K)
	if err != nil {
		return nil, fmt.Errorf("memory: vector search: %w", err)
	}
	msgs := make([]schema.Message, 0, len(docs))
	for _, d := range docs {
		msgs = append(msgs, schema.NewSystemMessage(d.PageContent))
	}
	return map[string]any{m.HistoryKey: msgs}, nil
}

func (m *VectorStoreMemory) SaveContext(ctx context.Context, humanInput, aiOutput string) error {
	m.mu.Lock()
	m.lastQuery = humanInput
	m.mu.Unlock()

	content := strings.TrimSpace(fmt.Sprintf("Human: %s\nAI: %s", humanInput, aiOutput))
	return m.store.AddDocuments(ctx, []schema.Document{{
		PageContent: content,
		Metadata:    map[string]any{"kind": "turn"},
	}})
}

func (m *VectorStoreMemory) Clear(ctx context.Context) error {
	// No universal Clear on VectorStore — delegate to Delete with an empty id
	// list (a no-op on most stores). Callers wanting a true wipe should
	// operate on the underlying store directly.
	return m.store.Delete(ctx, nil)
}
