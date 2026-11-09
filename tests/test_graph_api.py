import logging
from unittest.mock import Mock

import httpx
import pytest
from fastapi.exceptions import ResponseValidationError
from fastapi.testclient import TestClient
from openai import APIConnectionError

import api

SOURCE = {"page": 2, "chunk_id": 4, "score": 0.9, "preview": "示例论文证据"}


@pytest.fixture
def graph_dependencies(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(api.config, "DEEPSEEK_BASE_URL", "https://example.invalid")
    retriever = Mock(return_value=([], object(), []))
    openai = Mock()
    graph = Mock()
    create_graph = Mock(return_value=graph)
    monkeypatch.setattr(api, "get_retriever", retriever)
    monkeypatch.setattr(api, "OpenAI", openai)
    monkeypatch.setattr(api, "create_research_agent", create_graph)
    return graph, retriever, openai, create_graph


@pytest.mark.parametrize("state", [
    {"answer": "你好"},
    {"answer": "根据论文回答", "sources": [SOURCE]},
])
def test_graph_ask_success(state, graph_dependencies, caplog):
    graph, retriever, openai, create_graph = graph_dependencies
    graph.invoke.return_value = state

    with caplog.at_level(logging.INFO, logger="api"):
        response = TestClient(api.app).post("/graph/ask", json={"question": "  测试  "})

    assert response.status_code == 200
    assert response.json() == {"answer": state["answer"], "sources": state.get("sources", [])}
    openai.assert_called_once_with(api_key="test-key", base_url="https://example.invalid")
    create_graph.assert_called_once_with(openai.return_value, *retriever.return_value)
    graph.invoke.assert_called_once_with({"question": "测试"})
    assert "LangGraph 执行耗时:" in caplog.text


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "   "}])
def test_graph_ask_invalid_question(payload, graph_dependencies):
    _, retriever, openai, _ = graph_dependencies
    response = TestClient(api.app).post("/graph/ask", json=payload)

    assert response.status_code == 422
    retriever.assert_not_called()
    openai.assert_not_called()


def test_graph_ask_missing_key(graph_dependencies, monkeypatch):
    _, retriever, openai, _ = graph_dependencies
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    response = TestClient(api.app).post("/graph/ask", json={"question": "测试"})

    assert response.status_code == 500
    assert response.json()["detail"] == "DEEPSEEK_API_KEY 未配置"
    retriever.assert_not_called()
    openai.assert_not_called()


def test_graph_ask_api_error_is_logged(graph_dependencies, caplog):
    graph, *_ = graph_dependencies
    graph.invoke.side_effect = APIConnectionError(
        request=httpx.Request("POST", "https://example.invalid"),
    )

    with caplog.at_level(logging.INFO, logger="api"):
        response = TestClient(api.app).post("/graph/ask", json={"question": "测试"})

    assert response.status_code == 502
    assert response.json()["detail"] == "DeepSeek API 请求失败"
    assert any(record.levelno == logging.ERROR and record.exc_info for record in caplog.records)
    assert "LangGraph 执行耗时:" in caplog.text


@pytest.mark.parametrize("state", [{}, {"answer": None}, {"answer": " "}, {"answer": 123}])
def test_graph_ask_invalid_answer(state, graph_dependencies):
    graph, *_ = graph_dependencies
    graph.invoke.return_value = state

    response = TestClient(api.app).post("/graph/ask", json={"question": "测试"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Agent 未生成有效回答"


@pytest.mark.parametrize(("source", "invalid_field"), [
    ({"page": 2, "chunk_id": 4, "score": 0.9}, "preview"),
    ({**SOURCE, "page": "unknown"}, "page"),
    ({**SOURCE, "score": "unknown"}, "score"),
])
def test_graph_ask_validates_source_schema(source, invalid_field, graph_dependencies):
    graph, *_ = graph_dependencies
    graph.invoke.return_value = {"answer": "测试回答", "sources": [source]}

    with pytest.raises(ResponseValidationError) as error:
        TestClient(api.app).post("/graph/ask", json={"question": "测试"})

    assert error.value.errors()[0]["loc"] == ("response", "sources", 0, invalid_field)
