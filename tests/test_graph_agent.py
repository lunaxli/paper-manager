import json
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

import agent_tools
import config
from graph_agent import V2_SEARCH_PROMPT, create_research_agent


def completion(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def search_call(call_id, query):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name="search_paper",
            arguments=json.dumps({"query": query}),
        ),
    )


def test_direct_branch_skips_retrieval_and_followup(monkeypatch):
    client = Mock()
    create = client.chat.completions.create
    create.return_value = completion("你好")
    retrieve = Mock()
    monkeypatch.setattr(agent_tools, "retrieve_context_with_sources", retrieve)

    graph = create_research_agent(client, [], None, [])
    result = graph.invoke({"question": "  你好  "})

    assert result["answer"] == "你好"
    assert not result.get("sources")
    retrieve.assert_not_called()
    create.assert_called_once()
    assert create.call_args.kwargs["messages"] == [
        {"role": "system", "content": V2_SEARCH_PROMPT},
        {"role": "user", "content": "你好"},
    ]


def test_search_branch_sends_evidence_and_ranks_unique_sources(monkeypatch):
    client = Mock()
    create = client.chat.completions.create
    create.side_effect = [
        completion(tool_calls=[
            search_call("call_limit", "detection limit"),
            search_call("call_range", "linear range"),
        ]),
        completion("根据示例论文回答"),
    ]
    low = {"page": 2, "chunk_id": 4, "score": 0.6, "preview": "较低分片段"}
    high = {**low, "score": 0.95}
    other = {"page": 3, "chunk_id": 5, "score": 0.8, "preview": "另一片段"}
    retrieve = Mock(side_effect=[
        ("检测限证据", [low, other]),
        ("线性范围证据", [high]),
    ])
    monkeypatch.setattr(agent_tools, "retrieve_context_with_sources", retrieve)
    monkeypatch.setattr(config, "DEEPSEEK_MODEL", "test-model")
    monkeypatch.setattr(config, "DEEPSEEK_MAX_TOKENS", 123)

    chunks, model, embeddings = [], object(), []
    graph = create_research_agent(client, chunks, model, embeddings)
    result = graph.invoke({"question": "检测限和线性范围是多少？"})

    assert result["answer"] == "根据示例论文回答"
    assert result["sources"] == [high, other]
    assert retrieve.call_args_list == [
        call("detection limit", chunks, model, embeddings, k=10),
        call("linear range", chunks, model, embeddings, k=10),
    ]
    first, followup = [request.kwargs for request in create.call_args_list]
    assert first["messages"][0]["content"] == V2_SEARCH_PROMPT
    assert first["tools"] == [agent_tools.PAPER_SEARCH_TOOL]
    assert all(request["model"] == "test-model" for request in (first, followup))
    assert all(request["max_tokens"] == 123 for request in (first, followup))
    messages = followup["messages"]
    assert [message["role"] for message in messages] == [
        "system", "user", "assistant", "tool", "tool",
    ]
    assert [item["id"] for item in messages[2]["tool_calls"]] == ["call_limit", "call_range"]
    assert [item["tool_call_id"] for item in messages[3:]] == ["call_limit", "call_range"]
    assert json.loads(messages[3]["content"]) == {
        "context": "检测限证据", "sources": [low, other],
    }
    assert json.loads(messages[4]["content"]) == {
        "context": "线性范围证据", "sources": [high],
    }


def test_blank_question_stops_before_model_call():
    client = Mock()
    graph = create_research_agent(client, [], None, [])

    with pytest.raises(ValueError, match="问题不能为空"):
        graph.invoke({"question": "  "})

    client.chat.completions.create.assert_not_called()
