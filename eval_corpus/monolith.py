"""
monolith.py -- WidgetAPI Client

A deliberately tangled HTTP API client. Auth, caching, retry, parsing,
logging, and error handling are all mixed into a single class.

Transport is a callable: (method, url, headers, body) -> (status, headers, body)
This allows testing without network dependencies.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("widget_api")

# Transport type: (method, url, headers, body) -> (status_code, resp_headers, resp_body)
Transport = Callable[[str, str, dict, str | None], tuple[int, dict, str]]


# ---- Models (should be in their own module) ----

@dataclass
class Widget:
    id: str
    name: str
    price: float
    category: str
    in_stock: bool = True
    tags: list[str] = field(default_factory=list)


@dataclass
class WidgetPage:
    widgets: list[Widget]
    total: int
    page: int
    per_page: int
    has_next: bool


class APIError(Exception):
    """API error with status code and message."""
    def __init__(self, status: int, message: str, retry_after: float = 0):
        self.status = status
        self.message = message
        self.retry_after = retry_after
        super().__init__(f"API Error {status}: {message}")


@dataclass
class AuthToken:
    access_token: str
    expires_at: float
    refresh_token: str = ""


# ---- The God Class (should be split into auth, cache, retry, http, models) ----

class WidgetAPIClient:
    """Tangled API client with auth, caching, retry, parsing all in one class."""

    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    CACHE_TTL = 300.0
    BACKOFF_BASE = 0.01  # Small for testing; real code would use 1.0
    BACKOFF_MAX = 30.0

    def __init__(self, base_url: str, api_key: str,
                 transport: Transport | None = None,
                 cache_enabled: bool = True,
                 max_retries: int = 3,
                 cache_ttl: float = 300.0):
        # Validate config (mixed with initialization)
        if not base_url:
            raise ValueError("base_url is required")
        if not api_key:
            raise ValueError("api_key is required")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._transport = transport or self._default_transport
        self._cache: dict[str, tuple[float, Any]] = {}
        self._cache_enabled = cache_enabled
        self._cache_ttl = cache_ttl
        self._max_retries = max_retries
        self._token: AuthToken | None = None
        self._request_count = 0
        self._total_retry_count = 0

        # Logging config tangled into __init__
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s %(message)s"
            ))
            logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

    @staticmethod
    def _default_transport(method: str, url: str, headers: dict,
                           body: str | None) -> tuple[int, dict, str]:
        raise NotImplementedError(
            "No transport configured. Provide a transport function or use an HTTP library."
        )

    # ---- Auth (should be its own module) ----

    def _ensure_auth(self):
        """Check token validity, refresh if needed."""
        if self._token is None:
            logger.debug("No token, requesting initial auth...")
            status, resp_headers, resp_body = self._transport(
                "POST", f"{self.base_url}/auth/token",
                {"X-API-Key": self.api_key, "Content-Type": "application/json"},
                json.dumps({"grant_type": "api_key", "api_key": self.api_key})
            )
            if status != 200:
                raise APIError(status=status, message=f"Auth failed: {resp_body}")
            data = json.loads(resp_body)
            self._token = AuthToken(
                access_token=data["access_token"],
                expires_at=time.time() + data.get("expires_in", 3600),
                refresh_token=data.get("refresh_token", ""),
            )
            logger.info("Obtained auth token")

        elif time.time() >= self._token.expires_at - 60:
            logger.debug("Token expiring soon, refreshing...")
            status, resp_headers, resp_body = self._transport(
                "POST", f"{self.base_url}/auth/refresh",
                {"Authorization": f"Bearer {self._token.access_token}",
                 "Content-Type": "application/json"},
                json.dumps({"refresh_token": self._token.refresh_token})
            )
            if status != 200:
                self._token = None
                self._ensure_auth()
                return
            data = json.loads(resp_body)
            self._token = AuthToken(
                access_token=data["access_token"],
                expires_at=time.time() + data.get("expires_in", 3600),
                refresh_token=data.get("refresh_token",
                                       self._token.refresh_token),
            )
            logger.info("Refreshed auth token")

    # ---- URL building (trivial but tangled into the class) ----

    def _build_url(self, path: str, params: dict | None = None) -> str:
        url = f"{self.base_url}{path}"
        if params:
            filtered = {k: str(v) for k, v in params.items() if v is not None}
            if filtered:
                url += "?" + "&".join(f"{k}={v}" for k, v in filtered.items())
        return url

    # ---- Cache (should be its own module) ----

    def _cache_key(self, method: str, url: str) -> str:
        return hashlib.md5(f"{method}:{url}".encode()).hexdigest()

    def _get_cached(self, key: str) -> Any | None:
        if not self._cache_enabled:
            return None
        entry = self._cache.get(key)
        if entry is None:
            return None
        expires, value = entry
        if time.time() > expires:
            del self._cache[key]
            logger.debug(f"Cache expired for {key}")
            return None
        logger.debug(f"Cache hit for {key}")
        return value

    def _set_cached(self, key: str, value: Any):
        if self._cache_enabled:
            self._cache[key] = (time.time() + self._cache_ttl, value)

    def clear_cache(self):
        """Clear all cached responses."""
        self._cache.clear()

    # ---- THE GOD METHOD (auth + cache + retry + logging + error handling) ----

    def _make_request(self, method: str, path: str,
                      body: dict | None = None,
                      params: dict | None = None,
                      use_cache: bool = True) -> dict:
        """Make an authenticated, cached, retried API request."""
        self._ensure_auth()

        url = self._build_url(path, params)

        # Cache check (GET only)
        cache_key = None
        if method == "GET" and use_cache:
            cache_key = self._cache_key(method, url)
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        # Build headers (auth + content type mixed)
        headers = {
            "Authorization": f"Bearer {self._token.access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Request-ID": hashlib.md5(
                f"{time.time()}:{self._request_count}".encode()
            ).hexdigest()[:12],
        }

        body_str = json.dumps(body) if body else None

        # Retry loop with backoff
        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                logger.debug(
                    f"[{attempt+1}/{self._max_retries+1}] {method} {url}"
                )
                if body_str:
                    logger.debug(f"  Body: {body_str[:200]}")

                self._request_count += 1
                status, resp_headers, resp_body = self._transport(
                    method, url, headers, body_str
                )

                logger.debug(f"Response: {status} ({len(resp_body)} bytes)")

                # 401 -- token expired, refresh and retry
                if status == 401:
                    logger.warning("Got 401, refreshing token...")
                    self._token = None
                    self._ensure_auth()
                    headers["Authorization"] = (
                        f"Bearer {self._token.access_token}"
                    )
                    self._total_retry_count += 1
                    continue

                # 429 -- rate limited
                if status == 429:
                    retry_after = float(
                        resp_headers.get("Retry-After", "0.01")
                    )
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    time.sleep(retry_after)
                    self._total_retry_count += 1
                    continue

                # 5xx -- server error, retry with backoff
                if status >= 500:
                    if attempt < self._max_retries:
                        wait = min(
                            self.BACKOFF_BASE * (2 ** attempt),
                            self.BACKOFF_MAX
                        )
                        logger.warning(
                            f"Server error {status}, retry in {wait}s"
                        )
                        time.sleep(wait)
                        self._total_retry_count += 1
                        continue
                    raise APIError(
                        status=status,
                        message=f"Server error after retries: {resp_body}"
                    )

                # 4xx -- client error, don't retry
                if status >= 400:
                    try:
                        error_data = json.loads(resp_body)
                        msg = error_data.get("message", resp_body)
                    except (json.JSONDecodeError, KeyError):
                        msg = resp_body
                    raise APIError(status=status, message=msg)

                # Success -- parse and cache
                data = json.loads(resp_body) if resp_body else {}

                if method == "GET" and use_cache and cache_key:
                    self._set_cached(cache_key, data)

                return data

            except APIError:
                raise
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    wait = min(
                        self.BACKOFF_BASE * (2 ** attempt), self.BACKOFF_MAX
                    )
                    logger.error(f"Request error: {e}, retry in {wait}s")
                    time.sleep(wait)
                    self._total_retry_count += 1
                    continue
                raise APIError(
                    status=0, message=f"Request failed: {last_error}"
                )

        raise APIError(status=0, message="Max retries exceeded")

    # ---- Parsing (should be with models) ----

    def _parse_widget(self, data: dict) -> Widget:
        return Widget(
            id=data["id"],
            name=data["name"],
            price=float(data["price"]),
            category=data.get("category", "uncategorized"),
            in_stock=data.get("in_stock", True),
            tags=data.get("tags", []),
        )

    def _parse_widget_page(self, data: dict) -> WidgetPage:
        raw_items = data.get("widgets", data.get("items", []))
        widgets = [self._parse_widget(w) for w in raw_items]
        return WidgetPage(
            widgets=widgets,
            total=data.get("total", len(widgets)),
            page=data.get("page", 1),
            per_page=data.get("per_page", 20),
            has_next=data.get("has_next", False),
        )

    # ---- Public API ----

    def get_widget(self, widget_id: str) -> Widget:
        """Get a single widget by ID."""
        data = self._make_request("GET", f"/api/widgets/{widget_id}")
        return self._parse_widget(data)

    def list_widgets(self, page: int = 1, per_page: int = 20,
                     category: str | None = None) -> WidgetPage:
        """List widgets with optional filtering."""
        params = {"page": page, "per_page": per_page}
        if category:
            params["category"] = category
        data = self._make_request("GET", "/api/widgets", params=params)
        return self._parse_widget_page(data)

    def create_widget(self, name: str, price: float, category: str,
                      tags: list[str] | None = None) -> Widget:
        """Create a new widget."""
        body = {
            "name": name,
            "price": price,
            "category": category,
            "tags": tags or [],
        }
        data = self._make_request("POST", "/api/widgets", body=body)
        return self._parse_widget(data)

    def update_widget(self, widget_id: str, **fields) -> Widget:
        """Update a widget's fields."""
        data = self._make_request(
            "PATCH", f"/api/widgets/{widget_id}", body=fields
        )
        self.clear_cache()
        return self._parse_widget(data)

    def delete_widget(self, widget_id: str) -> bool:
        """Delete a widget. Returns True on success."""
        self._make_request("DELETE", f"/api/widgets/{widget_id}")
        self.clear_cache()
        return True

    def search_widgets(self, query: str, category: str | None = None,
                       min_price: float | None = None,
                       max_price: float | None = None,
                       page: int = 1) -> WidgetPage:
        """Search widgets by query string."""
        params: dict[str, Any] = {"q": query, "page": page}
        if category:
            params["category"] = category
        if min_price is not None:
            params["min_price"] = min_price
        if max_price is not None:
            params["max_price"] = max_price
        data = self._make_request("GET", "/api/widgets/search", params=params)
        return self._parse_widget_page(data)

    def get_stats(self) -> dict:
        """Get widget statistics (not cached)."""
        return self._make_request("GET", "/api/stats", use_cache=False)

    def bulk_update(self, updates: list[dict]) -> list[Widget]:
        """Update multiple widgets. Each dict needs 'id' + fields to update.

        Continues on individual failures, returns successfully updated widgets.
        """
        results = []
        errors = []
        for update in updates:
            widget_id = update.pop("id")
            try:
                # Manual cache invalidation tangled into business logic
                cache_key = self._cache_key(
                    "GET", self._build_url(f"/api/widgets/{widget_id}")
                )
                if cache_key in self._cache:
                    del self._cache[cache_key]

                widget = self.update_widget(widget_id, **update)
                results.append(widget)
                logger.info(f"Updated widget {widget_id}")
            except APIError as e:
                logger.error(
                    f"Failed to update widget {widget_id}: {e.message}"
                )
                errors.append({"id": widget_id, "error": e.message})
        if errors:
            logger.warning(
                f"Bulk update: {len(results)} ok, {len(errors)} failed"
            )
        return results

    def export_all(self, format: str = "json") -> str:
        """Export all widgets. Paginates internally."""
        all_widgets = []
        page = 1
        while True:
            result = self.list_widgets(page=page, per_page=100)
            all_widgets.extend(result.widgets)
            if not result.has_next:
                break
            page += 1
        if format == "json":
            return json.dumps([
                {"id": w.id, "name": w.name, "price": w.price,
                 "category": w.category, "in_stock": w.in_stock,
                 "tags": w.tags}
                for w in all_widgets
            ], indent=2)
        elif format == "csv":
            lines = ["id,name,price,category,in_stock"]
            for w in all_widgets:
                lines.append(
                    f"{w.id},{w.name},{w.price},{w.category},{w.in_stock}"
                )
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")

    # ---- Properties (stats tangled into the class) ----

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def retry_count(self) -> int:
        return self._total_retry_count

    @property
    def cache_size(self) -> int:
        return len(self._cache)
