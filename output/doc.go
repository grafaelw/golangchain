// Package output provides typed output parsers that transform raw LLM text
// into Go values.
//
// # Parsers
//
//   - [StrOutputParser]      — passes text through unchanged (trimmed)
//   - [JSONOutputParser]     — unmarshals to map[string]any; strips markdown fences
//   - [StructOutputParser]   — generic; unmarshals JSON to a typed struct T
//   - [ListOutputParser]     — splits on newline or comma into []string
//   - [BoolOutputParser]     — recognises yes/true/1/y → true, no/false/0/n → false
//
// # AsAny adapter
//
// [chain.NewLLMChain] expects an untyped parser interface. Use [AsAny] to bridge
// any typed [Parser][T] to that interface without losing type safety elsewhere:
//
//	chain.NewLLMChain(tmpl, model, output.AsAny(output.StrOutputParser{}))
//	chain.NewLLMChain(tmpl, model, output.AsAny(output.NewStructOutputParser[MyStruct]()))
//
// # Format instructions
//
// Every parser exposes [Parser.FormatInstructions] — a hint you can append to
// the prompt so the model knows which output format to produce.
package output
