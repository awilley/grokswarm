"""Client-side MCP integration — discover and register tools from MCP servers."""

import json
import httpx
import grokswarm.shared as shared


def _load_mcp_config() -> dict:
    """Read .mcp.json from project root, return mcpServers dict."""
    mcp_path = shared.PROJECT_DIR / ".mcp.json"
    if not mcp_path.exists():
        return {}
    try:
        data = json.loads(mcp_path.read_text(encoding="utf-8"))
        return data.get("mcpServers", {})
    except (json.JSONDecodeError, OSError):
        return {}


def _discover_mcp_tools(server_url: str) -> list[dict]:
    """Call tools/list on an MCP server, return tool definitions."""
    try:
        resp = httpx.post(
            server_url,
            headers={"Content-Type": "application/json"},
            json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 1},
            timeout=15.0,
        )
        resp.raise_for_status()
        result = resp.json()
        return result.get("result", {}).get("tools", [])
    except Exception as e:
        shared._log(f"MCP discovery failed for {server_url}: {e}")
        return []


def _call_mcp_tool(server_url: str, tool_name: str, arguments: dict) -> str:
    """Call a tool on an MCP server, return text result."""
    try:
        resp = httpx.post(
            server_url,
            headers={"Content-Type": "application/json"},
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
                "id": 2,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        result = resp.json()
        if "error" in result:
            return f"MCP error: {result['error'].get('message', str(result['error']))}"
        content = result.get("result", {}).get("content", [])
        parts = [c.get("text", "") for c in content if c.get("type") == "text"]
        text = "\n".join(parts) if parts else "No content returned."
        if len(text) > 8000:
            text = text[:8000] + "\n... (truncated)"
        return text
    except httpx.HTTPStatusError as e:
        return f"MCP error: server returned {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return f"MCP error: {e}"


def _mcp_schema_to_openai(mcp_tool: dict, prefixed_name: str) -> dict:
    """Convert an MCP tool definition to OpenAI function-calling schema."""
    input_schema = mcp_tool.get("inputSchema", {"type": "object", "properties": {}})
    return {
        "type": "function",
        "function": {
            "name": prefixed_name,
            "description": mcp_tool.get("description", "(MCP tool)"),
            "parameters": input_schema,
        },
    }


def register_mcp_tools():
    """Discover tools from all MCP servers in .mcp.json and register them."""
    from grokswarm.tools_registry import TOOL_SCHEMAS, TOOL_DISPATCH, READ_ONLY_TOOLS

    servers = _load_mcp_config()
    if not servers:
        return

    total_registered = 0
    for label, server_cfg in servers.items():
        server_url = server_cfg.get("url", "")
        if not server_url:
            continue
        safe_label = label.replace("-", "_").replace(" ", "_")
        tools = _discover_mcp_tools(server_url)
        for mcp_tool in tools:
            raw_name = mcp_tool.get("name", "")
            if not raw_name:
                continue
            prefixed = f"mcp_{safe_label}_{raw_name}"
            # Skip if already registered
            if any(t.get("function", {}).get("name") == prefixed for t in TOOL_SCHEMAS):
                continue
            schema = _mcp_schema_to_openai(mcp_tool, prefixed)
            TOOL_SCHEMAS.append(schema)
            # Capture loop vars for closure
            _url, _name = server_url, raw_name
            TOOL_DISPATCH[prefixed] = lambda args, u=_url, n=_name: _call_mcp_tool(u, n, args)
            READ_ONLY_TOOLS.add(prefixed)
            total_registered += 1
            shared._log(f"MCP registered: {prefixed} from {label}")

    if total_registered:
        shared.console.print(
            f"[swarm.dim]  mcp: {total_registered} tool{'s' if total_registered != 1 else ''}"
            f" registered from {len(servers)} server{'s' if len(servers) != 1 else ''}[/swarm.dim]"
        )
