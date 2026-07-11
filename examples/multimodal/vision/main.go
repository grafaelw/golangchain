// Example: multimodal
//
// Demonstrates the ContentPart type added to schema.Message for multimodal
// input (images, text, mixed content). Providers that support multimodal
// (GPT-4V, Claude 3, Gemini) can accept images alongside text.
//
// Highlights:
//
//   - ContentPart with Type "text" carries plain text.
//
//   - ContentPart with Type "image_url" references an image by URL with
//     optional detail level.
//
//   - When ContentParts is non-empty, it takes precedence over Content for
//     providers that support multimodal input.
//
//   - Message constructors still work with plain Content for backward compat.
//
//     Run this example with:
//     go run ./examples/multimodal/vision
package main

import (
	"fmt"

	"github.com/grafaelw/golangchain/schema"
)

func main() {
	// -------------------------------------------------------------------------
	// 1. Plain text message (backward-compatible)
	// -------------------------------------------------------------------------
	fmt.Println("--- 1. Plain text message ---")
	msg := schema.NewHumanMessage("What is in this image?")
	fmt.Printf("  %s\n", msg.String())

	// -------------------------------------------------------------------------
	// 2. Multimodal message with text + image URL
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 2. Multimodal: text + image_url ---")
	mm := schema.Message{
		Role: schema.RoleHuman,
		ContentParts: []schema.ContentPart{
			{Type: "text", Text: "Describe this image in detail."},
			{
				Type: "image_url",
				ImageURL: &schema.ImageURL{
					URL:    "https://example.com/photo.png",
					Detail: "high",
				},
			},
		},
	}
	fmt.Printf("  %s\n", mm.String())
	fmt.Printf("  Parts: %d (text=%s, image_url=%s detail=%s)\n",
		len(mm.ContentParts),
		mm.ContentParts[0].Text,
		mm.ContentParts[1].ImageURL.URL,
		mm.ContentParts[1].ImageURL.Detail,
	)

	// -------------------------------------------------------------------------
	// 3. Multiple images (GPT-4V supports multiple)
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 3. Multiple images ---")
	multi := schema.Message{
		Role: schema.RoleHuman,
		ContentParts: []schema.ContentPart{
			{Type: "text", Text: "Compare these two diagrams:"},
			{Type: "image_url", ImageURL: &schema.ImageURL{URL: "https://example.com/diagram1.png"}},
			{Type: "image_url", ImageURL: &schema.ImageURL{URL: "https://example.com/diagram2.png", Detail: "low"}},
		},
	}
	fmt.Printf("  %s\n", multi.String())
	fmt.Printf("  %d parts: 1 text + 2 images\n", len(multi.ContentParts))

	// -------------------------------------------------------------------------
	// 4. Base64 inline image (image_base64 type)
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 4. Base64 inline image ---")
	b64 := schema.Message{
		Role: schema.RoleHuman,
		ContentParts: []schema.ContentPart{
			{Type: "text", Text: "What text is in this screenshot?"},
			{
				Type:     "image_base64",
				MimeType: "image/png",
				Data:     "iVBORw0KGgoAAAANSUhEUgAA...", // truncated for demo
			},
		},
	}
	fmt.Printf("  %s\n", b64.String())
	fmt.Printf("  ContentParts[1].MimeType = %s\n", b64.ContentParts[1].MimeType)

	// -------------------------------------------------------------------------
	// 5. Round-trip: ContentParts ↔ JSON
	// -------------------------------------------------------------------------
	fmt.Println("\n--- 5. JSON round-trip ---")
	// The struct tags match OpenAI's content-part format.
	// A provider marshals ContentParts into the API request body.
	parts := []schema.ContentPart{
		{Type: "text", Text: "Hello"},
		{Type: "image_url", ImageURL: &schema.ImageURL{URL: "https://img.example/test.jpg"}},
	}
	fmt.Printf("  Parts serialized → suitable for OpenAI chat completion API\n")
	_ = parts // in a real provider: json.Marshal(parts)

	fmt.Println("\n✅ Multimodal examples complete.")
}
