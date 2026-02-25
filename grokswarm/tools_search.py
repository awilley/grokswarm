"""Web search, X search, and SSRF guard."""

import re
import httpx

import grokswarm.shared as shared

# S3: SSRF guard — block local/internal URLs
_SSRF_BLOCKED = re.compile(
    r"^https?://"
    r"(localhost|127\.\d+\.\d+\.\d+|\[::1\]|0\.0\.0\.0"
    r"|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+|192\.168\.\d+\.\d+"
    r"|169\.254\.\d+\.\d+)"
    r"(:\d+)?",
    re.IGNORECASE,
)


def _check_ssrf(url: str) -> str | None:
    if not url.startswith(("http://", "https://")):
        return f"Blocked: unsupported URL scheme in '{url[:60]}'. Only http/https allowed."
    if _SSRF_BLOCKED.match(url):
        return f"Blocked: requests to internal/local addresses are not allowed."
    return None


_last_response_ids: dict[str, str] = {}


def _xai_search(query: str, tool_type: str, continue_conversation: bool = False) -> str:
    try:
        payload = {
            "model": shared.MODEL,
            "input": query,
            "tools": [{"type": tool_type}],
        }
        if continue_conversation and tool_type in _last_response_ids:
            payload["previous_response_id"] = _last_response_ids[tool_type]
        resp = httpx.post(
            f"{shared.BASE_URL.replace('/v1', '')}/v1/responses",
            headers={
                "Authorization": f"Bearer {shared.XAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        # Cache response ID for multi-turn chaining
        resp_id = data.get("id", "")
        if resp_id:
            _last_response_ids[tool_type] = resp_id
        parts = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for block in item.get("content", []):
                    if block.get("type") == "output_text":
                        parts.append(block["text"])
        text = "\n".join(parts) if parts else "No results found."
        citations = data.get("citations", [])
        if citations:
            text += "\n\nSources:\n" + "\n".join(f"  - {url}" for url in citations[:15])
        if len(text) > 8000:
            text = text[:8000] + "\n... (truncated)"
        return text
    except httpx.HTTPStatusError as e:
        return f"Error: search API returned {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return f"Error: search failed: {e}"


def web_search(query: str, domains: list[str] | None = None) -> str:
    if domains:
        site_prefix = " ".join(f"site:{d}" for d in domains)
        query = f"{site_prefix} {query}"
    return _xai_search(query, "web_search")


def x_search(query: str) -> str:
    return _xai_search(query, "x_search")


def code_execution(code: str, language: str = "python") -> str:
    """Execute code server-side via the xAI Responses API code_execution tool."""
    try:
        resp = httpx.post(
            f"{shared.BASE_URL.replace('/v1', '')}/v1/responses",
            headers={
                "Authorization": f"Bearer {shared.XAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": shared.CODE_MODEL or shared.MODEL,
                "input": f"Execute this {language} code and return the output:\n```{language}\n{code}\n```",
                "tools": [{"type": "code_execution"}],
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        parts = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for block in item.get("content", []):
                    if block.get("type") == "output_text":
                        parts.append(block["text"])
            elif item.get("type") == "code_execution_result":
                output = item.get("output", "")
                if output:
                    parts.append(f"[execution output]\n{output}")
        text = "\n".join(parts) if parts else "No output returned."
        if len(text) > 8000:
            text = text[:8000] + "\n... (truncated)"
        return text
    except httpx.HTTPStatusError as e:
        return f"Error: code execution API returned {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return f"Error: code execution failed: {e}"
