import json
from types import SimpleNamespace
from unittest.mock import Mock

import agent_tools
from agent_runner import run_agent


def completion(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_run_agent_with_tool(monkeypatch):
    source = {"page": 2, "chunk_id": 4, "score": 0.8, "preview": "示例证据"}
    retrieve = Mock(return_value=("示例论文片段", [source]))
    monkeypatch.setattr(agent_tools, "retrieve_context_with_sources", retrieve)
    tool_call = SimpleNamespace(
        id="call_123",
        function=SimpleNamespace(name="search_paper", arguments='{"query":" limit "}'),
    )
    client = Mock()
    create = client.chat.completions.create
    create.side_effect = [completion(tool_calls=[tool_call]), completion("示例回答")]

    chunks, model, embeddings = [], object(), []
    result = run_agent("检测限是多少？", client, chunks, model, embeddings)

    assert result == {"answer": "示例回答", "sources": [source]}
    retrieve.assert_called_once_with("limit", chunks, model, embeddings, k=10)
    first, followup = [request.kwargs for request in create.call_args_list]
    assert first["tools"] == [agent_tools.PAPER_SEARCH_TOOL]
    messages = followup["messages"]
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool"]
    assert messages[2]["tool_calls"][0]["id"] == messages[3]["tool_call_id"] == "call_123"
    assert json.loads(messages[3]["content"]) == {"context": "示例论文片段", "sources": [source]}


def test_run_agent_direct_answer(monkeypatch):
    retrieve = Mock()
    monkeypatch.setattr(agent_tools, "retrieve_context_with_sources", retrieve)
    client = Mock()
    create = client.chat.completions.create
    create.return_value = completion("你好")

    assert run_agent("你好", client, [], None, []) == {"answer": "你好", "sources": []}
    create.assert_called_once()
    retrieve.assert_not_called()
