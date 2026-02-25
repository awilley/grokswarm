"""Image tools — analyze, generate, edit images via xAI APIs."""

import base64
import httpx

import grokswarm.shared as shared
from grokswarm.context import _safe_path


async def analyze_image(path: str, question: str = "Describe this image in detail.", detail: str = "auto") -> str:
    full_path = _safe_path(path)
    if not full_path:
        return "Access denied: outside project directory."
    if not full_path.exists() or not full_path.is_file():
        return f"File not found: {path}"
    ext = full_path.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".webp": "image/webp"}
    mime = mime_map.get(ext)
    if not mime:
        return f"Unsupported image format: {ext}. Supported: png, jpg, jpeg, gif, webp."
    data = full_path.read_bytes()
    if len(data) > 20 * 1024 * 1024:
        return "Error: image too large (max 20 MB)."
    b64 = base64.b64encode(data).decode("ascii")
    try:
        resp = await shared._api_call_with_retry(
            lambda: shared.client.chat.completions.create(
                model=shared.MODEL,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": detail}},
                ]}],
                max_tokens=1024,
            ),
            label="Vision"
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Vision error: {e}"


async def generate_image(prompt: str, n: int = 1, size: str = "1024x1024") -> str:
    """Generate images via the xAI images/generations API."""
    try:
        resp = httpx.post(
            f"{shared.BASE_URL.replace('/v1', '')}/v1/images/generations",
            headers={
                "Authorization": f"Bearer {shared.XAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "grok-2-image",
                "prompt": prompt,
                "n": min(n, 4),
                "size": size,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        urls = [item.get("url", "") for item in data.get("data", []) if item.get("url")]
        if not urls:
            return "No images were generated."
        return "Generated images:\n" + "\n".join(f"  - {url}" for url in urls)
    except httpx.HTTPStatusError as e:
        return f"Error: image generation API returned {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return f"Error: image generation failed: {e}"


async def edit_image(image_path: str, prompt: str) -> str:
    """Edit an image via the xAI images/edits API."""
    full_path = _safe_path(image_path)
    if not full_path:
        return "Access denied: outside project directory."
    if not full_path.exists() or not full_path.is_file():
        return f"File not found: {image_path}"
    try:
        with open(full_path, "rb") as f:
            resp = httpx.post(
                f"{shared.BASE_URL.replace('/v1', '')}/v1/images/edits",
                headers={
                    "Authorization": f"Bearer {shared.XAI_API_KEY}",
                },
                files={"image": (full_path.name, f, "image/png")},
                data={"prompt": prompt, "model": "grok-2-image", "n": "1"},
                timeout=120.0,
            )
        resp.raise_for_status()
        data = resp.json()
        urls = [item.get("url", "") for item in data.get("data", []) if item.get("url")]
        if not urls:
            return "No edited image was returned."
        return "Edited image:\n" + "\n".join(f"  - {url}" for url in urls)
    except httpx.HTTPStatusError as e:
        return f"Error: image edit API returned {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return f"Error: image edit failed: {e}"
