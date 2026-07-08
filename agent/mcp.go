package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/grafaelw/golangchain/tools"
)

// MCPServer represents a configured MCP server endpoint.
type MCPServer struct {
	Name   string `json:"name"`
	URL    string `json:"url"`
	APIKey string `json:"-"`
	client *http.Client
}

// MCPClient discovers and invokes tools from MCP servers.
type MCPClient struct {
	servers map[string]*MCPServer
	client  *http.Client
	mu      sync.RWMutex
}

// NewMCPClient creates a new MCPClient.
func NewMCPClient() *MCPClient {
	return &MCPClient{
		servers: make(map[string]*MCPServer),
		client:  &http.Client{Timeout: 30 * time.Second},
	}
}

// AddServer adds an MCP server to the client.
func (c *MCPClient) AddServer(server *MCPServer) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if server.client == nil {
		server.client = c.client
	}
	c.servers[server.Name] = server
}

// ListTools returns all tools available from all connected MCP servers.
func (c *MCPClient) ListTools(ctx context.Context) ([]tools.Tool, error) {
	c.mu.RLock()
	servers := make([]*MCPServer, 0, len(c.servers))
	for _, s := range c.servers {
		servers = append(servers, s)
	}
	c.mu.RUnlock()

	var allTools []tools.Tool
	for _, srv := range servers {
		serverTools, err := c.fetchServerTools(ctx, srv)
		if err != nil {
			continue
		}
		for _, st := range serverTools {
			st := st
			srvName := srv.Name
			allTools = append(allTools, tools.NewFuncTool(
				fmt.Sprintf("%s.%s", srvName, st.Name),
				fmt.Sprintf("[MCP/%s] %s", srvName, st.Description),
				st.Schema,
				func(ctx context.Context, input string) (string, error) {
					return c.CallTool(ctx, srvName, st.Name, input)
				},
			))
		}
	}

	return allTools, nil
}

// CallTool invokes a tool on an MCP server by name.
func (c *MCPClient) CallTool(ctx context.Context, serverName, toolName string, input string) (string, error) {
	c.mu.RLock()
	srv, ok := c.servers[serverName]
	c.mu.RUnlock()
	if !ok {
		return "", fmt.Errorf("mcp: server %q not found", serverName)
	}

	var args any
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		args = map[string]string{"input": input}
	}

	reqBody, err := json.Marshal(map[string]any{
		"name":      toolName,
		"arguments": args,
	})
	if err != nil {
		return "", fmt.Errorf("mcp: marshal: %w", err)
	}

	endpoint := srv.URL + "/tools/call"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(reqBody))
	if err != nil {
		return "", fmt.Errorf("mcp: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if srv.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+srv.APIKey)
	}

	resp, err := srv.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("mcp: request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", fmt.Errorf("mcp: read response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("mcp: %s returned status %d: %s", serverName, resp.StatusCode, string(body))
	}

	return string(body), nil
}

type mcpServerTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Schema      json.RawMessage `json:"schema"`
}

func (c *MCPClient) fetchServerTools(ctx context.Context, srv *MCPServer) ([]mcpServerTool, error) {
	endpoint := srv.URL + "/tools/list"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader([]byte("{}")))
	if err != nil {
		return nil, fmt.Errorf("mcp: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if srv.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+srv.APIKey)
	}

	resp, err := srv.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("mcp: request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("mcp: read response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("mcp: %s returned status %d: %s", srv.Name, resp.StatusCode, string(body))
	}

	var result struct {
		Tools []mcpServerTool `json:"tools"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("mcp: decode tools: %w", err)
	}

	return result.Tools, nil
}
