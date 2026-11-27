from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import app
from settings import CompanyConfig


@pytest.mark.asyncio
async def test_gl_desc_context_does_not_precompute_unrequested_calculations(monkeypatch):
    config = CompanyConfig(id="test", name="Test Company")
    settings = MagicMock()
    settings.get_current.return_value = config
    monkeypatch.setattr(app, "settings_manager", settings)

    tools = MagicMock()
    monkeypatch.setattr(app, "data_tools", {config.id: tools})
    sessions = {}
    monkeypatch.setattr(app, "chat_sessions", sessions)

    completion = AsyncMock(return_value={"answer": "Party A ka balance ...", "messages": []})
    monkeypatch.setattr(app, "run_chat_completion", completion)

    response = await app.chat(app.ChatMessage(
        message="Party A ka balance kya hai?",
        gl_desc="Party A",
    ))

    assert response["answer"] == "Party A ka balance ..."
    tools._matching_ledger.assert_not_called()
    tools.receivables_metrics.assert_not_called()
    tools.fifo_outstanding.assert_not_called()

    messages = completion.await_args.args[1]
    ledger_context = next(message["content"] for message in messages if "GlDesc=" in message["content"])
    assert 'GlDesc="Party A"' in ledger_context
    assert "Sirf user ke sawaal ka jawab do" in ledger_context
    assert messages[-1] == {"role": "user", "content": "Party A ka balance kya hai?"}


@pytest.mark.asyncio
async def test_ledger_report_tool_is_dispatched(monkeypatch):
    config = CompanyConfig(id="test", name="Test Company")
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="ledger_report",
            arguments='{"ledger":"Party A","as_of_date":"2026-07-16"}',
        ),
    )
    tool_message = SimpleNamespace(
        tool_calls=[tool_call],
        content=None,
        model_dump=lambda: {"role": "assistant", "content": None},
    )
    final_message = SimpleNamespace(
        tool_calls=None,
        content="Party A report.",
        model_dump=lambda: {"role": "assistant", "content": "Party A report."},
    )

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=[
        SimpleNamespace(choices=[SimpleNamespace(message=tool_message)]),
        SimpleNamespace(choices=[SimpleNamespace(message=final_message)]),
    ])
    monkeypatch.setattr(app, "AsyncOpenAI", MagicMock(return_value=client))

    tools = MagicMock()
    tools.ledger_report.return_value = app.ToolResult(success=True, data={"ledger": "Party A"})
    monkeypatch.setattr(app, "data_tools", {config.id: tools})

    result = await app.run_chat_completion(config, [], "session")

    assert result["answer"] == "Party A report."
    tools.ledger_report.assert_called_once_with("Party A", "2026-07-16")
    sent_tools = client.chat.completions.create.await_args_list[0].kwargs["tools"]
    sent_tool_names = {tool["function"]["name"] for tool in sent_tools}
    assert "ledger_report" in sent_tool_names
    assert "export_table" not in sent_tool_names
    assert "make_chart" not in sent_tool_names


def test_export_tool_is_only_available_for_explicit_export_requests():
    amount_question = [{"role": "user", "content": "Sirf amount batao"}]
    export_question = [{"role": "user", "content": "Export this report as xlsx"}]

    amount_tools = {tool["function"]["name"] for tool in app.chat_tools_for(amount_question)}
    export_tools = {tool["function"]["name"] for tool in app.chat_tools_for(export_question)}

    assert "export_table" not in amount_tools
    assert "export_table" in export_tools
    assert "make_chart" not in amount_tools
    assert "make_chart" not in export_tools


@pytest.mark.asyncio
async def test_data_question_without_successful_tool_result_is_not_answered_from_guess(monkeypatch):
    config = CompanyConfig(id="test", name="Test Company")
    fake_answer = (
        "Based on the tool response, I assume the function is receivableAgingReport."
    )
    message = SimpleNamespace(
        tool_calls=None,
        content=fake_answer,
        model_dump=lambda: {"role": "assistant", "content": fake_answer, "tool_calls": None},
    )
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(choices=[SimpleNamespace(message=message)])
    )
    monkeypatch.setattr(app, "AsyncOpenAI", MagicMock(return_value=client))
    monkeypatch.setattr(app, "data_tools", {config.id: MagicMock()})

    result = await app.run_chat_completion(config, [{
        "role": "user",
        "content": "A.B.N.Enterprises ka total outstanding kitna hai?",
    }], "session")

    assert "error" in result
    assert fake_answer not in result.get("answer", "")
    assert client.chat.completions.create.await_args.kwargs["tool_choice"] == "required"
