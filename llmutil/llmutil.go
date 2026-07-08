// Package llmutil — see doc.go for the package overview.
package llmutil

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/grafaelw/golangchain/llm"
	"github.com/grafaelw/golangchain/schema"
)

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

// Cache stores and retrieves LLM generations keyed by an opaque hash.
// Implementations must be safe for concurrent use.
type Cache interface {
	Get(ctx context.Context, key string) (*schema.Generation, bool, error)
	Set(ctx context.Context, key string, gen *schema.Generation) error
}

// MemoryCache is an in-memory Cache with an optional max entry count (LRU-ish
// via random eviction; suitable for dev/test loads).
type MemoryCache struct {
	mu    sync.RWMutex
	store map[string]*schema.Generation
	Max   int // 0 = unbounded
}

// NewMemoryCache creates an unbounded in-memory cache.
func NewMemoryCache() *MemoryCache {
	return &MemoryCache{store: make(map[string]*schema.Generation)}
}

func (c *MemoryCache) Get(_ context.Context, key string) (*schema.Generation, bool, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	g, ok := c.store[key]
	return g, ok, nil
}

func (c *MemoryCache) Set(_ context.Context, key string, gen *schema.Generation) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.Max > 0 && len(c.store) >= c.Max {
		for k := range c.store { // pick an arbitrary victim
			delete(c.store, k)
			break
		}
	}
	c.store[key] = gen
	return nil
}

// FileCache persists cache entries as one JSON file per key under Dir.
// Safe to share across processes: writes are atomic via rename.
type FileCache struct {
	Dir string
	mu  sync.Mutex
}

// NewFileCache creates a FileCache and ensures dir exists.
func NewFileCache(dir string) (*FileCache, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("llmutil: cache dir: %w", err)
	}
	return &FileCache{Dir: dir}, nil
}

func (c *FileCache) path(key string) string { return filepath.Join(c.Dir, key+".json") }

func (c *FileCache) Get(_ context.Context, key string) (*schema.Generation, bool, error) {
	data, err := os.ReadFile(c.path(key))
	if errors.Is(err, os.ErrNotExist) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, err
	}
	var gen schema.Generation
	if err := json.Unmarshal(data, &gen); err != nil {
		return nil, false, err
	}
	return &gen, true, nil
}

func (c *FileCache) Set(_ context.Context, key string, gen *schema.Generation) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	data, err := json.Marshal(gen)
	if err != nil {
		return err
	}
	tmp := c.path(key) + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, c.path(key))
}

// ---------------------------------------------------------------------------
// CachingLLM
// ---------------------------------------------------------------------------

// CachingLLM wraps an llm.LLM to memoise successful Generate calls.
// Streaming calls bypass the cache.
type CachingLLM struct {
	Inner llm.LLM
	Cache Cache
}

// NewCachingLLM wraps model with cache.
func NewCachingLLM(model llm.LLM, cache Cache) *CachingLLM {
	return &CachingLLM{Inner: model, Cache: cache}
}

func (c *CachingLLM) ModelName() string { return c.Inner.ModelName() }

func (c *CachingLLM) Generate(ctx context.Context, messages []schema.Message, opts ...llm.Option) (*schema.Generation, error) {
	key := hashCall(c.Inner.ModelName(), messages, opts)
	if gen, ok, err := c.Cache.Get(ctx, key); err == nil && ok {
		return gen, nil
	}
	gen, err := c.Inner.Generate(ctx, messages, opts...)
	if err != nil {
		return nil, err
	}
	_ = c.Cache.Set(ctx, key, gen)
	return gen, nil
}

func (c *CachingLLM) Stream(ctx context.Context, messages []schema.Message, opts ...llm.Option) (<-chan schema.StreamChunk, error) {
	return c.Inner.Stream(ctx, messages, opts...)
}

// hashCall computes a deterministic cache key from the model name, messages
// and options (options are hashed by their JSON representation via a helper).
func hashCall(model string, msgs []schema.Message, opts []llm.Option) string {
	h := sha256.New()
	fmt.Fprintf(h, "model=%s\n", model)
	if data, err := json.Marshal(msgs); err == nil {
		h.Write(data)
	}
	// llm.Option is an opaque func; approximate by counting them plus their
	// %p addresses. Users who need stable-across-run keys should pass
	// deterministic options (which is normal in most apps).
	for i, o := range opts {
		fmt.Fprintf(h, "\nopt[%d]=%p", i, o)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// ---------------------------------------------------------------------------
// RetryingLLM
// ---------------------------------------------------------------------------

// RetryConfig controls RetryingLLM.
type RetryConfig struct {
	MaxAttempts int           // total attempts including the first (default 3)
	Base        time.Duration // initial backoff (default 250ms)
	Max         time.Duration // cap (default 8s)
	// Retryable decides whether an error should trigger another attempt.
	// If nil, DefaultRetryable is used (retries on all non-context errors).
	Retryable func(error) bool
}

// DefaultRetryable returns true for any error that is not a context error.
func DefaultRetryable(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	return true
}

func (r RetryConfig) normalise() RetryConfig {
	if r.MaxAttempts <= 0 {
		r.MaxAttempts = 3
	}
	if r.Base <= 0 {
		r.Base = 250 * time.Millisecond
	}
	if r.Max <= 0 {
		r.Max = 8 * time.Second
	}
	if r.Retryable == nil {
		r.Retryable = DefaultRetryable
	}
	return r
}

// RetryingLLM wraps a model with exponential-backoff retries.
type RetryingLLM struct {
	Inner llm.LLM
	Cfg   RetryConfig
}

// NewRetryingLLM wraps model with cfg.
func NewRetryingLLM(model llm.LLM, cfg RetryConfig) *RetryingLLM {
	return &RetryingLLM{Inner: model, Cfg: cfg.normalise()}
}

func (r *RetryingLLM) ModelName() string { return r.Inner.ModelName() }

func (r *RetryingLLM) Generate(ctx context.Context, messages []schema.Message, opts ...llm.Option) (*schema.Generation, error) {
	var last error
	backoff := r.Cfg.Base
	for attempt := 1; attempt <= r.Cfg.MaxAttempts; attempt++ {
		gen, err := r.Inner.Generate(ctx, messages, opts...)
		if err == nil {
			return gen, nil
		}
		last = err
		if !r.Cfg.Retryable(err) || attempt == r.Cfg.MaxAttempts {
			break
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(backoff):
		}
		backoff *= 2
		if backoff > r.Cfg.Max {
			backoff = r.Cfg.Max
		}
	}
	return nil, fmt.Errorf("llmutil: retry exhausted after %d attempts: %w", r.Cfg.MaxAttempts, last)
}

// Stream does not retry (partial streams are hard to roll back); errors are
// returned as-is so the caller can decide.
func (r *RetryingLLM) Stream(ctx context.Context, messages []schema.Message, opts ...llm.Option) (<-chan schema.StreamChunk, error) {
	return r.Inner.Stream(ctx, messages, opts...)
}

// ---------------------------------------------------------------------------
// RateLimitedLLM
// ---------------------------------------------------------------------------

// RateLimitedLLM bounds concurrent in-flight calls (semaphore) and requests
// per second (token bucket).
type RateLimitedLLM struct {
	Inner       llm.LLM
	sem         chan struct{}
	tickets     chan struct{}
	ticker      *time.Ticker
	stopRefill  chan struct{}
	Concurrency int
	QPS         float64
}

// NewRateLimitedLLM wraps model with a concurrency and QPS limit.
// concurrency=0 disables the semaphore; qps=0 disables the token bucket.
func NewRateLimitedLLM(model llm.LLM, concurrency int, qps float64) *RateLimitedLLM {
	r := &RateLimitedLLM{Inner: model, Concurrency: concurrency, QPS: qps}
	if concurrency > 0 {
		r.sem = make(chan struct{}, concurrency)
	}
	if qps > 0 {
		burst := concurrency
		if burst <= 0 {
			burst = int(qps) + 1
		}
		r.tickets = make(chan struct{}, burst)
		for i := 0; i < burst; i++ {
			r.tickets <- struct{}{}
		}
		r.ticker = time.NewTicker(time.Duration(float64(time.Second) / qps))
		r.stopRefill = make(chan struct{})
		go r.refill()
	}
	return r
}

// Close stops the internal ticker. Safe to call multiple times.
func (r *RateLimitedLLM) Close() {
	if r.stopRefill != nil {
		select {
		case <-r.stopRefill:
		default:
			close(r.stopRefill)
		}
	}
	if r.ticker != nil {
		r.ticker.Stop()
	}
}

func (r *RateLimitedLLM) refill() {
	for {
		select {
		case <-r.stopRefill:
			return
		case <-r.ticker.C:
			select {
			case r.tickets <- struct{}{}:
			default:
			}
		}
	}
}

func (r *RateLimitedLLM) acquire(ctx context.Context) error {
	if r.tickets != nil {
		select {
		case <-r.tickets:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	if r.sem != nil {
		select {
		case r.sem <- struct{}{}:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

func (r *RateLimitedLLM) release() {
	if r.sem != nil {
		<-r.sem
	}
}

func (r *RateLimitedLLM) ModelName() string { return r.Inner.ModelName() }

func (r *RateLimitedLLM) Generate(ctx context.Context, messages []schema.Message, opts ...llm.Option) (*schema.Generation, error) {
	if err := r.acquire(ctx); err != nil {
		return nil, err
	}
	defer r.release()
	return r.Inner.Generate(ctx, messages, opts...)
}

func (r *RateLimitedLLM) Stream(ctx context.Context, messages []schema.Message, opts ...llm.Option) (<-chan schema.StreamChunk, error) {
	if err := r.acquire(ctx); err != nil {
		return nil, err
	}
	inner, err := r.Inner.Stream(ctx, messages, opts...)
	if err != nil {
		r.release()
		return nil, err
	}
	out := make(chan schema.StreamChunk, cap(inner))
	go func() {
		defer close(out)
		defer r.release()
		for c := range inner {
			out <- c
		}
	}()
	return out, nil
}
