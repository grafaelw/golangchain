// Package azure provides an Azure OpenAI Service LLM implementation for golangchain.
//
// # Usage
//
//	model, err := azure.New(
//	    azure.WithAPIKey(os.Getenv("AZURE_OPENAI_API_KEY")),
//	    azure.WithEndpoint(os.Getenv("AZURE_OPENAI_ENDPOINT")),
//	    azure.WithDeployment("gpt-4o"),
//	    azure.WithAPIVersion("2024-02-01"), // optional; defaults to a recent stable version
//	)
//
// Entra ID (Azure AD) token authentication is also supported via [WithEntraToken].
package azure
