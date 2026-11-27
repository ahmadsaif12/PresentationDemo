"""Mocked coverage for the configuration save/test flow (no live API calls)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app
from settings import ApiSettings, CompanyConfig, SettingsManager


class MockResponse:
    def __init__(self, payload, status_code=200, reason_phrase="OK"):
        self._payload = payload
        self.status_code = status_code
        self.reason_phrase = reason_phrase

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("request failed", request=MagicMock(), response=self)


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    manager = SettingsManager(str(tmp_path / "settings.json"))
    manager.configs.clear()
    manager.current_config_id = ""
    manager.add_config(CompanyConfig(
        id="old",
        name="Old Company",
        api=ApiSettings(api_key="server-only-key", api_key_prefix="Bearer "),
    ))
    manager.set_current("old")
    monkeypatch.setattr(app, "settings_manager", manager)
    monkeypatch.delenv("DATA_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Keep SQLite test artefacts inside pytest's temporary directory.
    original_executor = app.SQLiteExecutor
    monkeypatch.setattr(app, "SQLiteExecutor", lambda _: original_executor(str(tmp_path / "loaded.sqlite")))
    return manager


@pytest.mark.asyncio
async def test_test_connection_uses_server_key_with_blank_prefix(isolated_settings):
    """A blank prefix is intentional, and neither it nor the header is inherited."""
    response = MockResponse({"data": [{"VNo": "1", "VDate": "2024-01-01", "Amount": "100"}]})
    with patch("app.httpx.AsyncClient") as client_type:
        client = client_type.return_value.__aenter__.return_value
        client.get = AsyncMock(return_value=response)
        result = await app.test_connection(app.TestConnectionRequest(
            base_url="api.example.test/report?format=json",
            database="ACME one",
            param_name="dbname",
            api_key_header="X-API-Key",
            api_key_prefix="",
        ))

    assert result["success"] is True
    assert result["response_shape"] == "object.data[]"
    assert result["final_url"] == "http://api.example.test/report?format=json&dbname=ACME+one"
    assert client.get.await_args.kwargs["headers"] == {"X-API-Key": "server-only-key"}


@pytest.mark.asyncio
async def test_save_selects_new_config_then_current_reports_loaded_records(isolated_settings):
    response = MockResponse({"rows": [{
        "VNo": "1", "VDate": "2024-01-01", "GlDesc": "Customer A",
        "Source": "SB", "GrpDesc": "Sales", "Catagory": "Revenue",
        "Narration": "Mock invoice", "LocalDrAmt": "100", "LocalCrAmt": "0",
    }]})
    with patch("app.httpx.AsyncClient") as client_type:
        client_type.return_value.__aenter__.return_value.get = AsyncMock(return_value=response)
        result = await app.save_config(app.ApiConfigRequest(
            name="New Company",
            base_url="https://api.example.test/report",
            database="ACME",
            api_key_header="X-API-Key",
            api_key_prefix="",
        ))

    current = await app.get_current_config()
    assert result["success"] is True
    assert result["load_result"]["success"] is True, result
    assert result["load_result"]["record_count"] == 1
    assert isolated_settings.current_config_id == result["config_id"]
    assert current["id"] == result["config_id"]
    assert current["record_count"] == 1
    assert "api_key" not in current["api"]


@pytest.mark.asyncio
async def test_test_connection_returns_safe_shape_error(isolated_settings):
    with patch("app.httpx.AsyncClient") as client_type:
        client_type.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=MockResponse({"message": "not a record envelope"})
        )
        result = await app.test_connection(app.TestConnectionRequest(
            base_url="https://api.example.test/report",
            database="ACME",
        ))

    assert result["success"] is False
    assert result["response_shape"] == "object"
    assert "Expected an array" in result["error"]
    assert "server-only-key" not in result["error"]


def test_browser_payload_has_no_secret_fields():
    page = open("index.html", encoding="utf-8").read()
    assert "api_key: $('apiKey')" not in page
    assert "openai_key: $('openaiKey')" not in page
    assert 'id="apiKey"' not in page
    assert 'id="openaiKey"' not in page


def test_environment_llm_provider_overrides_saved_company_provider(monkeypatch):
    config = CompanyConfig(openai=app.OpenAISettings(
        api_key="old-invalid-key",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
    ))
    monkeypatch.setenv("OPENAI_API_KEY", "ollama")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("OPENAI_MODEL", "qwen3:4b")

    resolved = app.resolve_openai_settings(config)

    assert resolved.api_key == "ollama"
    assert resolved.base_url == "http://127.0.0.1:11434/v1"
    assert resolved.model == "qwen3:4b"
