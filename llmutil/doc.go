// Package llmutil provides drop-in wrappers around any llm.LLM that add
// production concerns — caching, retry, and rate limiting — without
// touching provider packages.
//
// # Wrappers
//
//   - [CachingLLM]     — memoises successful Generate calls by a hash of the
//     model name, messages, and options. Backed by a pluggable [Cache].
//   - [RetryingLLM]    — exponential-backoff retries on transient errors.
//     Controlled by [RetryConfig]; streams are passed through as-is because
//     partial output cannot be safely rewound.
//   - [RateLimitedLLM] — bounds concurrent in-flight calls (semaphore) and
//     requests per second (token bucket).
//
// # Cache backends
//
//   - [MemoryCache] — process-local, in-memory (optional max entries).
//   - [FileCache]   — one JSON file per key under a directory; atomic writes.
//
// # Composition
//
// Wrappers implement llm.LLM themselves, so they stack:
//
//	base := openai.New(...)
//	model := llmutil.NewRetryingLLM(
//	    llmutil.NewCachingLLM(
//	        llmutil.NewRateLimitedLLM(base, 2, 5), // 5 QPS, concurrency 2
//	        llmutil.NewMemoryCache(),
//	    ),
//	    llmutil.RetryConfig{MaxAttempts: 4, Base: 200 * time.Millisecond},
//	)
package llmutil
