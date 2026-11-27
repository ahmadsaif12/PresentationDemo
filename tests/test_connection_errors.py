"""Tests for connection error handling."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx


class MockResponse:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(f"HTTP {self.status_code}", request=MagicMock(), response=self)


@pytest.mark.asyncio
async def test_connection_timeout():
    """Test handling of connection timeout."""
    from app import fetch_data_from_api

    with patch('httpx.AsyncClient') as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            side_effect=httpx.TimeoutException("Timeout")
        )

        result = await fetch_data_from_api(
            base_url="https://api.example.com/endpoint?",
            database="DB123",
            param_name="dbname",
            timeout=5
        )

        assert result["success"] is False
        assert "timeout" in result["error"].lower()


@pytest.mark.asyncio
async def test_connection_refused():
    """Test handling of connection refused."""
    from app import fetch_data_from_api

    with patch('httpx.AsyncClient') as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        result = await fetch_data_from_api(
            base_url="https://api.example.com/endpoint?",
            database="DB123",
            param_name="dbname",
            timeout=30
        )

        assert result["success"] is False
        assert "connection" in result["error"].lower()


@pytest.mark.asyncio
async def test_http_401_unauthorized():
    """Test handling of 401 Unauthorized."""
    from app import fetch_data_from_api

    with patch('httpx.AsyncClient') as mock_client:
        mock_response = MockResponse(401, {"error": "Unauthorized"})
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        result = await fetch_data_from_api(
            base_url="https://api.example.com/endpoint?",
            database="DB123",
            param_name="dbname",
            api_key="invalid_key",
            timeout=30
        )

        assert result["success"] is False
        assert result["status_code"] == 401


@pytest.mark.asyncio
async def test_http_404_not_found():
    """Test handling of 404 Not Found."""
    from app import fetch_data_from_api

    with patch('httpx.AsyncClient') as mock_client:
        mock_response = MockResponse(404, {"error": "Not Found"})
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        result = await fetch_data_from_api(
            base_url="https://api.example.com/endpoint?",
            database="DB123",
            param_name="dbname",
            timeout=30
        )

        assert result["success"] is False
        assert result["status_code"] == 404


@pytest.mark.asyncio
async def test_http_500_server_error():
    """Test handling of 500 Server Error."""
    from app import fetch_data_from_api

    with patch('httpx.AsyncClient') as mock_client:
        mock_response = MockResponse(500, {"error": "Internal Server Error"})
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        result = await fetch_data_from_api(
            base_url="https://api.example.com/endpoint?",
            database="DB123",
            param_name="dbname",
            timeout=30
        )

        assert result["success"] is False
        assert result["status_code"] == 500


@pytest.mark.asyncio
async def test_invalid_json_response():
    """Test handling of invalid JSON response."""
    from app import fetch_data_from_api

    with patch('httpx.AsyncClient') as mock_client:
        mock_response = MockResponse(200, text="Not JSON")
        mock_response.json = MagicMock(side_effect=ValueError("Invalid JSON"))
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        result = await fetch_data_from_api(
            base_url="https://api.example.com/endpoint?",
            database="DB123",
            param_name="dbname",
            timeout=30
        )

        assert result["success"] is False
        assert "json" in result["error"].lower()


@pytest.mark.asyncio
async def test_empty_response():
    """Test handling of empty response."""
    from app import fetch_data_from_api

    with patch('httpx.AsyncClient') as mock_client:
        mock_response = MockResponse(200, {"status": True, "data": []})
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        result = await fetch_data_from_api(
            base_url="https://api.example.com/endpoint?",
            database="DB123",
            param_name="dbname",
            timeout=30
        )

        assert result["success"] is True
        assert result["record_count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])