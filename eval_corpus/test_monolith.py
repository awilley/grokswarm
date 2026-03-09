"""
test_monolith.py -- Tests for the WidgetAPI client.

Uses a mock transport so no network calls are needed.
After refactoring monolith.py into a client/ package, these tests
must still pass (import from monolith.py which re-exports from client/).
"""

import json
import pytest
from monolith import WidgetAPIClient, Widget, WidgetPage, APIError


# ---- Mock transport ----

WIDGET_DB = {
    "w1": {"id": "w1", "name": "Sprocket", "price": 9.99,
           "category": "mechanical", "in_stock": True, "tags": ["metal"]},
    "w2": {"id": "w2", "name": "Gadget", "price": 24.99,
           "category": "electronic", "in_stock": True, "tags": ["battery"]},
    "w3": {"id": "w3", "name": "Thingamajig", "price": 4.50,
           "category": "mechanical", "in_stock": False, "tags": []},
}


def mock_transport(method, url, headers, body):
    """Simple mock that returns canned responses based on URL path."""
    # Auth endpoints
    if "/auth/token" in url:
        return (200, {}, json.dumps({
            "access_token": "test-token-123",
            "expires_in": 3600,
            "refresh_token": "test-refresh-456",
        }))

    if "/auth/refresh" in url:
        return (200, {}, json.dumps({
            "access_token": "refreshed-token-789",
            "expires_in": 3600,
        }))

    # Extract path (strip base URL and query string)
    path = url.split("//")[-1]  # Remove scheme
    path = "/" + "/".join(path.split("/")[1:])  # Remove host, keep path
    path = path.split("?")[0]  # Remove query string

    # Widget CRUD -- order matters: most specific first
    if path == "/api/widgets/search" and method == "GET":
        return (200, {}, json.dumps({
            "widgets": list(WIDGET_DB.values())[:2],
            "total": 2, "page": 1, "per_page": 20, "has_next": False,
        }))

    if path.startswith("/api/widgets/") and method == "GET":
        widget_id = path.split("/api/widgets/")[-1]
        if widget_id in WIDGET_DB:
            return (200, {}, json.dumps(WIDGET_DB[widget_id]))
        return (404, {}, json.dumps({"message": "Widget not found"}))

    if path == "/api/widgets" and method == "GET":
        widgets = list(WIDGET_DB.values())
        return (200, {}, json.dumps({
            "widgets": widgets, "total": len(widgets),
            "page": 1, "per_page": 20, "has_next": False,
        }))

    if path == "/api/widgets" and method == "POST":
        data = json.loads(body)
        new_id = f"w{len(WIDGET_DB) + 1}"
        widget = {"id": new_id, **data, "in_stock": True}
        return (201, {}, json.dumps(widget))

    if method == "PATCH" and path.startswith("/api/widgets/"):
        widget_id = path.split("/api/widgets/")[-1]
        if widget_id in WIDGET_DB:
            updated = {**WIDGET_DB[widget_id], **json.loads(body)}
            return (200, {}, json.dumps(updated))
        return (404, {}, json.dumps({"message": "Widget not found"}))

    if method == "DELETE" and path.startswith("/api/widgets/"):
        widget_id = path.split("/api/widgets/")[-1]
        if widget_id in WIDGET_DB:
            return (200, {}, json.dumps({"deleted": True}))
        return (404, {}, json.dumps({"message": "Widget not found"}))

    if "/api/stats" in url:
        return (200, {}, json.dumps({
            "total_widgets": len(WIDGET_DB),
            "total_value": sum(w["price"] for w in WIDGET_DB.values()),
        }))

    return (404, {}, json.dumps({"message": "Not found"}))


def make_client(**kwargs):
    return WidgetAPIClient(
        base_url="https://api.widgets.test",
        api_key="test-key-abc",
        transport=mock_transport,
        **kwargs,
    )


# ---- Tests ----

class TestWidgetClient:
    def test_get_widget(self):
        client = make_client()
        widget = client.get_widget("w1")
        assert isinstance(widget, Widget)
        assert widget.id == "w1"
        assert widget.name == "Sprocket"
        assert widget.price == 9.99
        assert widget.category == "mechanical"

    def test_get_widget_not_found(self):
        client = make_client()
        with pytest.raises(APIError) as exc_info:
            client.get_widget("nonexistent")
        assert exc_info.value.status == 404

    def test_list_widgets(self):
        client = make_client()
        page = client.list_widgets()
        assert isinstance(page, WidgetPage)
        assert len(page.widgets) == 3
        assert page.total == 3

    def test_create_widget(self):
        client = make_client()
        widget = client.create_widget("NewWidget", 15.00, "test", tags=["new"])
        assert isinstance(widget, Widget)
        assert widget.name == "NewWidget"
        assert widget.price == 15.00

    def test_update_widget(self):
        client = make_client()
        widget = client.update_widget("w1", name="Updated Sprocket")
        assert widget.name == "Updated Sprocket"

    def test_delete_widget(self):
        client = make_client()
        result = client.delete_widget("w1")
        assert result is True

    def test_search_widgets(self):
        client = make_client()
        page = client.search_widgets("sprocket")
        assert isinstance(page, WidgetPage)
        assert len(page.widgets) >= 1

    def test_caching(self):
        client = make_client(cache_enabled=True)
        client.get_widget("w1")
        assert client.cache_size >= 1
        # Second call should hit cache (no extra transport call)
        count_before = client.request_count
        client.get_widget("w1")
        # request_count should NOT increase (cache hit)
        assert client.request_count == count_before

    def test_cache_disabled(self):
        client = make_client(cache_enabled=False)
        client.get_widget("w1")
        assert client.cache_size == 0

    def test_get_stats(self):
        client = make_client()
        stats = client.get_stats()
        assert "total_widgets" in stats
        assert stats["total_widgets"] == 3

    def test_export_json(self):
        client = make_client()
        exported = client.export_all(format="json")
        data = json.loads(exported)
        assert isinstance(data, list)
        assert len(data) == 3

    def test_export_csv(self):
        client = make_client()
        exported = client.export_all(format="csv")
        lines = exported.strip().split("\n")
        assert lines[0] == "id,name,price,category,in_stock"
        assert len(lines) == 4  # header + 3 widgets

    def test_request_count(self):
        client = make_client(cache_enabled=False)
        assert client.request_count == 0
        client.get_widget("w1")
        assert client.request_count > 0  # auth + get
