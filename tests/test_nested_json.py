"""Tests for nested JSON data handling."""
import pytest
from profiler import profile_records, detect_column_type, parse_amount, parse_date


class TestNestedJson:
    """Test handling of nested JSON structures in API responses."""

    def test_flat_records(self):
        """Test profiling of flat records."""
        records = [
            {"VNo": "1", "VDate": "01/01/2024", "Source": "OB", "LocalDrAmt": "1000", "LocalCrAmt": "0"},
            {"VNo": "2", "VDate": "02/01/2024", "Source": "SB", "LocalDrAmt": "500", "LocalCrAmt": "0"},
        ]
        profile = profile_records(records)

        assert profile["record_count"] == 2
        assert profile["columns"]["VNo"]["type"] == "text"
        assert profile["columns"]["VDate"]["type"] == "date"
        assert profile["columns"]["Source"]["type"] == "text"
        assert profile["columns"]["LocalDrAmt"]["type"] == "number"
        assert profile["columns"]["LocalCrAmt"]["type"] == "number"

    def test_records_with_nested_objects(self):
        """Test that nested objects are handled (flattened to string)."""
        records = [
            {"VNo": "1", "Details": {"nested": "value", "amount": 100}, "Amount": "1000"},
            {"VNo": "2", "Details": {"nested": "other", "amount": 200}, "Amount": "2000"},
        ]
        profile = profile_records(records)

        # Nested objects become text type
        assert profile["columns"]["Details"]["type"] == "text"
        assert profile["columns"]["Amount"]["type"] == "number"

    def test_records_with_arrays(self):
        """Test that arrays are handled (flattened to string)."""
        records = [
            {"VNo": "1", "Tags": ["tag1", "tag2"], "Amount": "1000"},
            {"VNo": "2", "Tags": ["tag3"], "Amount": "2000"},
        ]
        profile = profile_records(records)

        assert profile["columns"]["Tags"]["type"] == "text"
        assert profile["columns"]["Amount"]["type"] == "number"

    def test_mixed_types_in_column(self):
        """Test column with mixed numeric and text values."""
        records = [
            {"VNo": "1", "Amount": "1000"},
            {"VNo": "2", "Amount": "2000"},
            {"VNo": "3", "Amount": "N/A"},
        ]
        profile = profile_records(records)

        # Should detect as text since not all are numeric
        assert profile["columns"]["Amount"]["type"] == "text"

    def test_null_and_empty_values(self):
        """Test handling of null and empty values."""
        records = [
            {"VNo": "1", "Amount": "1000", "Note": "First"},
            {"VNo": "2", "Amount": "", "Note": None},
            {"VNo": "3", "Amount": None, "Note": "Third"},
        ]
        profile = profile_records(records)

        assert profile["columns"]["Amount"]["null_count"] == 2
        assert profile["columns"]["Note"]["null_count"] == 2
        assert profile["columns"]["Amount"]["type"] == "number"  # Only non-null values counted

    def test_various_date_formats(self):
        """Test detection of various date formats."""
        records = [
            {"Date": "01/01/2024"},
            {"Date": "02/01/2024"},
            {"Date": "03/01/2024"},
        ]
        profile = profile_records(records)
        assert profile["columns"]["Date"]["type"] == "date"

        records2 = [
            {"Date": "2024-01-01"},
            {"Date": "2024-01-02"},
        ]
        profile2 = profile_records(records2)
        assert profile2["columns"]["Date"]["type"] == "date"

        records3 = [
            {"Date": "01-01-2024"},
            {"Date": "02-01-2024"},
        ]
        profile3 = profile_records(records3)
        assert profile3["columns"]["Date"]["type"] == "date"

    def test_parse_amount_various_formats(self):
        """Test parse_amount with various formats."""
        assert parse_amount("1,000.50") == 1000.50
        assert parse_amount("1000") == 1000.0
        assert parse_amount("-500.25") == -500.25
        assert parse_amount("  1,234.56  ") == 1234.56
        assert parse_amount("") == 0.0
        assert parse_amount(None) == 0.0
        assert parse_amount("N/A") == 0.0
        assert parse_amount("1,00,000") == 100000.0  # Indian format

    def test_parse_date_various_formats(self):
        """Test parse_date with various formats."""
        assert parse_date("01/01/2024") == "2024-01-01"
        assert parse_date("2024-01-01") == "2024-01-01"
        assert parse_date("01-01-2024") == "2024-01-01"
        assert parse_date("") is None
        assert parse_date(None) is None
        assert parse_date("invalid") is None

    def test_unicode_and_special_chars(self):
        """Test handling of unicode and special characters."""
        records = [
            {"Name": "Dhiraj Prasad Chaudhari", "Amount": "1000"},
            {"Name": "测试数据", "Amount": "2000"},
            {"Name": "Москва", "Amount": "3000"},
        ]
        profile = profile_records(records)

        assert profile["columns"]["Name"]["type"] == "text"
        assert profile["columns"]["Name"]["distinct_count"] == 3

    def test_large_numbers(self):
        """Test handling of large numbers."""
        records = [
            {"Amount": "1000000000"},
            {"Amount": "999999999999.99"},
        ]
        profile = profile_records(records)

        assert profile["columns"]["Amount"]["type"] == "number"
        assert profile["columns"]["Amount"]["max"] == 999999999999.99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])