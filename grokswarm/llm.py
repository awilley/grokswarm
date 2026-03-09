"""Central adapter for xai-sdk. All xai_sdk interaction flows through this module."""

from __future__ import annotations

import json
from typing import Any

from xai_sdk import AsyncClient
from xai_sdk.chat import (
    assistant as _assistant,
    chat_pb2,
    image as _image,
    system as _system,
    text as _text,
    tool as _tool,
    tool_result as _tool_result,
    user as _user,
    Response,
    Chunk,
)

# ---------------------------------------------------------------------------
# Client management  (lazy — AsyncClient created on first use inside the
# running event-loop so gRPC channels bind to the correct loop)
# ---------------------------------------------------------------------------

_client: AsyncClient | None = None
_api_key: str | None = None
_timeout: int = 3600


def init_client(api_key: str, timeout: int = 3600) -> None:
    """Store credentials. The actual AsyncClient is created lazily on first use."""
    global _api_key, _timeout, _client
    _api_key = api_key
    _timeout = timeout
    _client = None  # force re-creation on next get_client()


def reset_client(api_key: str, timeout: int = 3600) -> None:
    """Replace credentials and invalidate any existing client."""
    init_client(api_key, timeout)


def get_client() -> AsyncClient:
    """Return the current AsyncClient, creating it lazily if needed.

    Must be called from inside a running asyncio event-loop so gRPC
    channels bind to the correct loop.
    """
    global _client
    if _client is None:
        if _api_key is None:
            raise RuntimeError("LLM client not initialised — call llm.init_client() first")
        _client = AsyncClient(api_key=_api_key, timeout=_timeout)
    return _client


# ---------------------------------------------------------------------------
# Tool conversion  (OpenAI schema dicts → xai_sdk Tool protos)
# ---------------------------------------------------------------------------

def convert_tools(openai_schemas: list[dict]) -> list:
    """Convert TOOL_SCHEMAS list-of-dicts to xai_sdk tool() objects.

    Each dict has shape: {"type": "function", "function": {"name", "description", "parameters"}}
    """
    out = []
    for schema in openai_schemas:
        func = schema.get("function", schema)
        out.append(_tool(
            name=func["name"],
            description=func.get("description", ""),
            parameters=func.get("parameters", {}),
        ))
    return out


# ---------------------------------------------------------------------------
# Chat creation
# ---------------------------------------------------------------------------

def create_chat(
    model: str,
    *,
    tools: list | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    response_format: str | Any | None = None,
):
    """Create a Chat object on the current client.

    Args:
        model: model name string
        tools: list of xai_sdk tool() objects (already converted)
        max_tokens: optional max output tokens
        temperature: optional temperature
        response_format: 'json_object', 'text', or a Pydantic model
    """
    client = get_client()
    kwargs: dict[str, Any] = {"model": model}
    if tools:
        kwargs["tools"] = tools
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if response_format is not None:
        kwargs["response_format"] = response_format
    return client.chat.create(**kwargs)


# ---------------------------------------------------------------------------
# Message population  (list-of-dicts → chat.append() calls)
# ---------------------------------------------------------------------------

_TOOL_CALL_TYPE_CLIENT = chat_pb2.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL  # enum value 1


def _make_tool_call(tc_dict: dict) -> chat_pb2.ToolCall:
    """Build a ToolCall proto from a conversation dict entry."""
    tc = chat_pb2.ToolCall()
    tc.id = tc_dict.get("id", "")
    tc.type = _TOOL_CALL_TYPE_CLIENT
    func = tc_dict.get("function", {})
    tc.function.name = func.get("name", "")
    tc.function.arguments = func.get("arguments", "")
    return tc


def populate_chat(chat, messages: list[dict]) -> None:
    """Replay a list-of-dicts conversation into a Chat object via append().

    Handles roles: system, user, assistant (with optional tool_calls), tool.
    """
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        if role == "system":
            if content:
                chat.append(_system(content))

        elif role == "user":
            # Handle multimodal content (list of dicts with type/text/image_url)
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append(_text(part["text"]))
                    elif part.get("type") == "image_url":
                        url = part["image_url"]["url"] if isinstance(part["image_url"], dict) else part["image_url"]
                        detail = part["image_url"].get("detail", "auto") if isinstance(part["image_url"], dict) else "auto"
                        parts.append(_image(image_url=url, detail=detail))
                if parts:
                    chat.append(_user(*parts))
            elif content:
                chat.append(_user(content))

        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                # Assistant message with tool calls
                asst_msg = _assistant(content) if content else _assistant()
                for tc_dict in tool_calls:
                    asst_msg.tool_calls.append(_make_tool_call(tc_dict))
                chat.append(asst_msg)
            elif content:
                chat.append(_assistant(content))

        elif role == "tool":
            tool_content = content or ""
            tool_call_id = msg.get("tool_call_id", "")
            chat.append(_tool_result(tool_content, tool_call_id=tool_call_id))


# ---------------------------------------------------------------------------
# Usage extraction
# ---------------------------------------------------------------------------

def extract_usage(response: Response) -> dict:
    """Normalise usage from a Response into a plain dict.

    Returns: {prompt_tokens, completion_tokens, total_tokens, cached_tokens}
    """
    usage = response.usage
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    total = getattr(usage, "total_tokens", 0) or 0
    cached = getattr(usage, "cached_prompt_text_tokens", 0) or 0
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": total,
        "cached_tokens": cached,
    }


def extract_cached_tokens(response: Response) -> int:
    """Extract cached_prompt_text_tokens from a Response."""
    usage = response.usage
    return getattr(usage, "cached_prompt_text_tokens", 0) or 0


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def response_content(response: Response) -> str:
    """Get the text content from a Response."""
    return response.content or ""


def response_tool_calls(response: Response) -> list:
    """Get tool_calls from a Response (list of ToolCall protos)."""
    return list(response.tool_calls) if response.tool_calls else []


def tool_call_to_dict(tc) -> dict:
    """Convert a ToolCall proto to the dict format used by conversation lists."""
    return {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        },
    }
