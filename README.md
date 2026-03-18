# nvidia-docs-mcp

MCP server for NVIDIA PTX ISA and CuTe DSL documentation. Enables AI agents to search and query GPU programming documentation.

## Setup

```bash
uv sync
uv run nvidia-docs-mcp --scrape   # fetch and index documentation (~30s)
```

## Usage

### Claude Code

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "nvidia-docs": {
      "command": "uv",
      "args": ["--directory", "/path/to/nvidia-docs-mcp", "run", "nvidia-docs-mcp"]
    }
  }
}
```

### Claude Desktop

Add to Claude Desktop MCP settings:

```json
{
  "nvidia-docs": {
    "command": "uv",
    "args": ["--directory", "/path/to/nvidia-docs-mcp", "run", "nvidia-docs-mcp"]
  }
}
```

## Tools

| Tool | Description |
|------|-------------|
| `search_ptx` | Search PTX ISA docs (instructions, memory model, sync primitives) |
| `get_ptx_instruction` | Get detailed docs for a specific PTX instruction |
| `list_ptx_instructions` | List PTX instructions by category or architecture |
| `search_cutedsl` | Search CuTe DSL docs (kernel patterns, APIs, JIT, framework integration) |
| `get_cutedsl_api` | Get API reference for a specific CuTe DSL module |

## Examples

- "What does the wgmma.mma_async instruction do on Hopper?"
- "How do I use MMA in CuTe DSL?"
- "List all tcgen05 instructions for sm_100"
- "How does cp.async.bulk work?"
- "Show me the CuTe DSL warpgroup API"
