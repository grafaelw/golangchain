// Package textsplitter chunks long text into smaller, overlapping pieces
// suitable for embedding and retrieval.
//
// # Splitters
//
//   - [CharacterSplitter]           — split on a single separator string.
//   - [RecursiveCharacterSplitter]  — try a priority list of separators
//     (paragraph → line → sentence → word → character). Recommended default
//     for prose. See [DefaultRecursiveSeparators].
//   - [MarkdownSplitter]            — RecursiveCharacterSplitter pre-loaded
//     with Markdown-aware separators (H1..H6, code fences, list bullets).
//
// All splitters implement the [Splitter] interface and expose SplitDocuments,
// which preserves per-Document metadata and adds "chunk" / "source_chunks".
//
// # Length functions
//
// Chunk size is measured with a pluggable [LenFunc]. Built-ins:
//   - [RuneLen] (default) — counts UTF-8 runes.
//   - [ByteLen]           — counts bytes.
//   - [ApproxTokenLen]    — rune count divided by four (rough BPE proxy).
//
// # Quick start
//
//	splitter := textsplitter.NewRecursiveCharacterSplitter(nil,
//	    textsplitter.WithChunkSize(800),
//	    textsplitter.WithChunkOverlap(100),
//	)
//	chunks := splitter.SplitText(longString)
//	docChunks := splitter.SplitDocuments(loadedDocs)
package textsplitter
