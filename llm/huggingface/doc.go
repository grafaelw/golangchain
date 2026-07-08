// Package huggingface provides embeddings.Embedder and llm.LLM
// implementations backed by the HuggingFace Inference API
// (https://api-inference.huggingface.co).
//
// # Embedder usage
//
//	emb, err := huggingface.NewEmbedder(
//	    os.Getenv("HF_API_KEY"),
//	    huggingface.WithEmbeddingModel("sentence-transformers/all-MiniLM-L6-v2"),
//	)
//	vectors, _ := emb.EmbedDocuments(ctx, []string{"hello", "world"})
//
// # LLM usage
//
//	model, err := huggingface.NewLLM(
//	    os.Getenv("HF_API_KEY"),
//	    huggingface.WithLLMModel("mistralai/Mistral-7B-Instruct-v0.3"),
//	    huggingface.WithLLMMaxTokens(1024),
//	)
//	gen, _ := model.Generate(ctx, []schema.Message{
//	    schema.NewHumanMessage("What is Go?"),
//	})
//
// Streaming is supported by emitting the full response as a single chunk
// (the Inference API does not support Server-Sent Events).
package huggingface
