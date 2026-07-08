// Package retriever defines the Retriever interface and provides retrieval
// strategies that go beyond a raw vector-store lookup.
//
// # Retrievers
//
//   - [VectorStoreRetriever]         — thin adapter over a
//     vectorstore.VectorStore.
//   - [BM25Retriever]                — classical lexical BM25, no embeddings
//     required. Fast; complements vector search inside an ensemble.
//   - [EnsembleRetriever]            — merges the outputs of multiple
//     retrievers using Reciprocal Rank Fusion (RRF).
//   - [MultiQueryRetriever]          — asks an LLM to generate N alternative
//     phrasings of the query, retrieves for each, then unions the results.
//   - [ContextualCompressionRetriever] — post-processes retrieved docs
//     through a [DocumentCompressor] to filter or shrink them.
//
// # Document compressors
//
//   - [KeywordCompressor]         — drops docs lacking query terms.
//   - [LLMRelevanceCompressor]    — asks an LLM to answer YES/NO per doc.
//
// # Composition
//
// All strategies implement [Retriever], so they compose freely:
//
//	vec  := retriever.NewVectorStoreRetriever(store, 10)
//	lex  := retriever.NewBM25Retriever(chunks, 10)
//	hyb  := retriever.NewEnsembleRetriever([]retriever.Retriever{vec, lex}, nil, 8)
//	mq   := retriever.NewMultiQueryRetriever(hyb, plannerLLM)
//	best := retriever.NewContextualCompressionRetriever(mq, retriever.KeywordCompressor{})
//
//	docs, err := best.GetRelevantDocuments(ctx, "user question")
package retriever
