"""Tests for URL builder logic."""
import pytest
from settings import SettingsManager, CompanyConfig, ApiSettings


class TestUrlBuilder:
    """Test the URL joining logic for various base URL formats."""

    def test_base_url_with_question_mark(self):
        """Test base URL that already has query parameters."""
        config = CompanyConfig(
            id="test",
            name="Test",
            api=ApiSettings(
                base_url="https://api.example.com/endpoint?",
                database="DB123",
                param_name="dbname"
            )
        )
        manager = SettingsManager()
        result = manager.build_final_url(config)
        assert result == "https://api.example.com/endpoint?dbname=DB123"

    def test_base_url_with_existing_params(self):
        """Test base URL with existing query parameters."""
        config = CompanyConfig(
            id="test",
            name="Test",
            api=ApiSettings(
                base_url="https://api.example.com/endpoint?token=abc&format=json",
                database="DB123",
                param_name="dbname"
            )
        )
        manager = SettingsManager()
        result = manager.build_final_url(config)
        assert result == "https://api.example.com/endpoint?token=abc&format=json&dbname=DB123"

    def test_base_url_without_question_mark(self):
        """Test base URL without any query parameters."""
        config = CompanyConfig(
            id="test",
            name="Test",
            api=ApiSettings(
                base_url="https://api.example.com/endpoint",
                database="DB123",
                param_name="dbname"
            )
        )
        manager = SettingsManager()
        result = manager.build_final_url(config)
        assert result == "https://api.example.com/endpoint?dbname=DB123"

    def test_base_url_with_placeholder(self):
        """Test base URL with {db} placeholder."""
        config = CompanyConfig(
            id="test",
            name="Test",
            api=ApiSettings(
                base_url="https://api.example.com/endpoint?dbname={db}&format=json",
                database="DB123",
                param_name="dbname"
            )
        )
        manager = SettingsManager()
        result = manager.build_final_url(config)
        assert result == "https://api.example.com/endpoint?dbname=DB123&format=json"

    def test_base_url_placeholder_only(self):
        """Test base URL with only {db} placeholder."""
        config = CompanyConfig(
            id="test",
            name="Test",
            api=ApiSettings(
                base_url="https://api.example.com/endpoint/{db}",
                database="DB123",
                param_name="dbname"
            )
        )
        manager = SettingsManager()
        result = manager.build_final_url(config)
        assert result == "https://api.example.com/endpoint/DB123"

    def test_param_name_custom(self):
        """Test custom parameter name."""
        config = CompanyConfig(
            id="test",
            name="Test",
            api=ApiSettings(
                base_url="https://api.example.com/endpoint?",
                database="DB123",
                param_name="database_id"
            )
        )
        manager = SettingsManager()
        result = manager.build_final_url(config)
        assert result == "https://api.example.com/endpoint?database_id=DB123"

    def test_replaces_existing_param(self):
        """Test that existing param value is replaced."""
        config = CompanyConfig(
            id="test",
            name="Test",
            api=ApiSettings(
                base_url="https://api.example.com/endpoint?dbname=OLD_DB",
                database="NEW_DB",
                param_name="dbname"
            )
        )
        manager = SettingsManager()
        result = manager.build_final_url(config)
        assert result == "https://api.example.com/endpoint?dbname=NEW_DB"

    def test_http_scheme_added(self):
        """Test that http:// is added if missing."""
        config = CompanyConfig(
            id="test",
            name="Test",
            api=ApiSettings(
                base_url="api.example.com/endpoint?",
                database="DB123",
                param_name="dbname"
            )
        )
        manager = SettingsManager()
        result = manager.build_final_url(config)
        assert result == "http://api.example.com/endpoint?dbname=DB123"

    def test_https_preserved(self):
        """Test that https:// is preserved."""
        config = CompanyConfig(
            id="test",
            name="Test",
            api=ApiSettings(
                base_url="https://api.example.com/endpoint?",
                database="DB123",
                param_name="dbname"
            )
        )
        manager = SettingsManager()
        result = manager.build_final_url(config)
        assert result == "https://api.example.com/endpoint?dbname=DB123"

    def test_empty_database_returns_base(self):
        """Test empty database returns base URL unchanged."""
        config = CompanyConfig(
            id="test",
            name="Test",
            api=ApiSettings(
                base_url="https://api.example.com/endpoint?",
                database="",
                param_name="dbname"
            )
        )
        manager = SettingsManager()
        result = manager.build_final_url(config)
        assert result == "https://api.example.com/endpoint?"

    def test_empty_base_returns_empty(self):
        """Test empty base URL returns empty string."""
        config = CompanyConfig(
            id="test",
            name="Test",
            api=ApiSettings(
                base_url="",
                database="DB123",
                param_name="dbname"
            )
        )
        manager = SettingsManager()
        result = manager.build_final_url(config)
        assert result == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])