"""Tests for multi-turn conversation handling."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import json


class TestMultiTurnConversation:
    """Test multi-turn conversation with session history."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        client = AsyncMock()
        return client

    @pytest.fixture
    def mock_db(self):
        """Create a mock database with test data."""
        # This would be a real SQLite connection in integration tests
        pass

    @pytest.mark.asyncio
    async def test_session_persistence(self):
        """Test that session_id persists across turns."""
        from app import chat_completion

        with patch('app.openai_client') as mock_client, \
             patch('app.get_db_connection') as mock_db:

            # Mock DB
            mock_conn = MagicMock()
            mock_db.return_value = mock_conn

            # Mock tool execution
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchall.return_value = [
                ('Party A', 1000.0, 0.0, 1000.0),
                ('Party B', 500.0, 0.0, 500.0),
            ]
            mock_cursor.description = [('GlDesc',), ('dr_total',), ('cr_total',), ('balance',)]

            # Mock OpenAI response - first turn
            mock_response1 = MagicMock()
            mock_response1.choices = [MagicMock()]
            mock_response1.choices[0].message.content = "Here are the balances."
            mock_response1.choices[0].message.tool_calls = None

            # Second turn
            mock_response2 = MagicMock()
            mock_response2.choices = [MagicMock()]
            mock_response2.choices[0].message.content = "Party A has higher balance."
            mock_response2.choices[0].message.tool_calls = None

            mock_client.chat.completions.create = AsyncMock(
                side_effect=[mock_response1, mock_response2]
            )

            # First turn
            result1 = await chat_completion(
                message="Show me party balances",
                session_id="",
                config_id="default",
                as_of_date="2024-12-31"
            )

            assert result1.session_id != ""
            session_id = result1.session_id

            # Second turn with same session_id
            result2 = await chat_completion(
                message="Which party has the highest?",
                session_id=session_id,
                config_id="default",
                as_of_date="2024-12-31"
            )

            assert result2.session_id == session_id

    @pytest.mark.asyncio
    async def test_max_turns_limit(self):
        """Test that history is limited to 12 turns."""
        from app import ChatHistory

        history = ChatHistory(max_turns=12)

        # Add 15 turns
        for i in range(15):
            history.add(f"User message {i}", f"Assistant response {i}")

        # Should only keep last 12
        messages = history.get_messages()
        assert len(messages) == 24  # 12 user + 12 assistant = 24 messages

        # First message should be "User message 3"
        assert messages[0]['content'] == "User message 3"

    @pytest.mark.asyncio
    async def test_context_carried_between_turns(self):
        """Test that context from previous turns is available."""
        from app import ChatHistory

        history = ChatHistory(max_turns=12)

        history.add("What is the total sales?", "Total sales is 100,000")
        history.add("What about purchases?", "Total purchases is 50,000")

        messages = history.get_messages()

        # Check that both turns are in history
        contents = [m['content'] for m in messages]
        assert any("total sales" in c.lower() for c in contents)
        assert any("purchases" in c.lower() for c in contents)

    @pytest.mark.asyncio
    async def test_new_chat_clears_history(self):
        """Test that new chat starts fresh session."""
        from app import chat_completion

        with patch('app.openai_client') as mock_client, \
             patch('app.get_db_connection') as mock_db:

            mock_conn = MagicMock()
            mock_db.return_value = mock_conn
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchall.return_value = []
            mock_cursor.description = []

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Response"
            mock_response.choices[0].message.tool_calls = None
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            # First chat
            result1 = await chat_completion(
                message="Hello",
                session_id="",
                config_id="default",
                as_of_date="2024-12-31"
            )

            # New chat (empty session_id)
            result2 = await chat_completion(
                message="Hello again",
                session_id="",
                config_id="default",
                as_of_date="2024-12-31"
            )

            # Should be different sessions
            assert result1.session_id != result2.session_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])