"""Browser tools: init/cleanup, fetch_page, screenshot_page, extract_links."""

import atexit
import subprocess

from rich.prompt import Confirm

import grokswarm.shared as shared
from grokswarm.context import _safe_path

_browser_instance = None
_playwright_instance = None


def _get_browser():
    global _browser_instance, _playwright_instance
    if _browser_instance is None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            shared.console.print("[swarm.warning]Playwright is not installed (needed for browser tools).[/swarm.warning]")
            if Confirm.ask("Install playwright + chromium now?", default=True):
                try:
                    subprocess.run(["pip", "install", "playwright"], check=True, timeout=120)
                    subprocess.run(["playwright", "install", "chromium"], check=True, timeout=180)
                    shared.console.print("[swarm.accent]Playwright installed! Retrying...[/swarm.accent]")
                    from playwright.sync_api import sync_playwright
                except Exception as e:
                    shared.console.print(f"[swarm.error]Installation failed: {e}[/swarm.error]")
                    return None
            else:
                return None
        _playwright_instance = sync_playwright().start()
        _browser_instance = _playwright_instance.chromium.launch(headless=True)
    return _browser_instance


def _close_browser():
    global _browser_instance, _playwright_instance
    if _browser_instance:
        try:
            _browser_instance.close()
        except Exception:
            pass
        _browser_instance = None
    if _playwright_instance:
        try:
            _playwright_instance.stop()
        except Exception:
            pass
        _playwright_instance = None


def _atexit_close_browser():
    import threading
    t = threading.Thread(target=_close_browser, daemon=True)
    t.start()
    t.join(timeout=3)


atexit.register(_atexit_close_browser)


def fetch_page(url: str) -> str:
    from grokswarm.tools_search import _check_ssrf
    ssrf = _check_ssrf(url)
    if ssrf:
        return ssrf
    browser = _get_browser()
    if browser is None:
        return "Error: playwright is not installed. Run: pip install playwright && playwright install chromium"
    page = browser.new_page()
    try:
        page.goto(url, timeout=15000, wait_until="domcontentloaded")
        title = page.title()
        text = page.inner_text("body")
        if len(text) > 8000:
            text = text[:8000] + "\n... (truncated)"
        return f"Title: {title}\n\n{text}"
    except Exception as e:
        return f"Error fetching page: {e}"
    finally:
        page.close()


def screenshot_page(url: str, save_path: str = "screenshot.png") -> str:
    from grokswarm.tools_search import _check_ssrf
    ssrf = _check_ssrf(url)
    if ssrf:
        return ssrf
    full_path = _safe_path(save_path)
    if not full_path:
        return "Access denied: outside project directory."
    shared.console.print(f"[bold yellow]About to SCREENSHOT:[/bold yellow] {url}")
    shared.console.print(f"[dim]Save to: {full_path}[/dim]")
    if not shared._auto_approve("Approve screenshot?"):
        return "Screenshot cancelled by user."
    browser = _get_browser()
    if browser is None:
        return "Error: playwright is not installed. Run: pip install playwright && playwright install chromium"
    page = browser.new_page()
    try:
        page.goto(url, timeout=15000, wait_until="domcontentloaded")
        full_path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(full_path), full_page=False)
        return f"Screenshot saved to {full_path}"
    except Exception as e:
        return f"Error: {e}"
    finally:
        page.close()


def extract_links(url: str) -> str:
    from grokswarm.tools_search import _check_ssrf
    ssrf = _check_ssrf(url)
    if ssrf:
        return ssrf
    browser = _get_browser()
    if browser is None:
        return "Error: playwright is not installed. Run: pip install playwright && playwright install chromium"
    page = browser.new_page()
    try:
        page.goto(url, timeout=15000, wait_until="domcontentloaded")
        links = page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => ({text: e.innerText.trim().substring(0, 80), href: e.href})).filter(l => l.href.startsWith('http'))"
        )
        seen = set()
        lines = []
        for link in links:
            if link["href"] not in seen:
                seen.add(link["href"])
                text = link["text"][:60] if link["text"] else "(no text)"
                lines.append(f"  {text} \u2192 {link['href']}")
        if len(lines) > 50:
            lines = lines[:50] + [f"  ... and {len(lines) - 50} more"]
        return "\n".join(lines) if lines else "No links found."
    except Exception as e:
        return f"Error: {e}"
    finally:
        page.close()
