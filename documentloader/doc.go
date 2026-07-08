// Package documentloader loads text-bearing content from files, directories,
// and HTTP URLs into []schema.Document with populated Metadata.
//
// # Loaders
//
//   - [TextLoader]      — plain UTF-8 text file.
//   - [MarkdownLoader]  — Markdown file (loaded as text; pair with
//     textsplitter.MarkdownSplitter for structural chunking).
//   - [CSVLoader]       — one Document per row; column values become
//     metadata keys. Set [CSVLoader.ContentColumn] for column-only content.
//   - [HTMLLoader]      — strips tags and script/style/head/noscript blocks;
//     captures <title> into metadata. Dependency-free.
//   - [HTTPLoader]      — fetches a URL; dispatches to HTML/text extraction
//     based on the response Content-Type.
//   - [DirectoryLoader] — walks a directory tree and dispatches per file
//     extension via [DirectoryLoader.Handlers] (customisable). Recursive by
//     default; supports glob filtering and a Fallback loader.
//
// All loaders implement the [Loader] interface, so downstream pipelines can
// be written against a single type and swapped freely.
//
// # Metadata
//
// Every returned Document carries at least a "source" key. Additional keys
// depend on the loader (e.g. "mime", "size", "status", "row", "title").
//
// # Quick start
//
//	docs, err := documentloader.NewDirectoryLoader("./corpus").Load(ctx)
//	// or:
//	docs, err := documentloader.NewHTTPLoader("https://example.com").Load(ctx)
package documentloader
