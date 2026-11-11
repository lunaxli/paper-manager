
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError

from api import app


@pytest.fixture
def mock_v3(monkeypatch):
    """Mock DeepSeek、MCP Client 和 LangGraph。"""

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    fake_mcp_client = object()

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock()

    with (
        patch("api.Client") as mock_client,
        patch("api.OpenAI") as mock_openai,
        patch(
            "api.create_mcp_research_agent",
            return_value=fake_graph
        ) as mock_create_graph
    ):

        mock_client.return_value.__aenter__.return_value = (
            fake_mcp_client
        )

        yield {
            "mcp_client": fake_mcp_client,
            "graph": fake_graph,
            "client_mock": mock_client,
            "openai_mock": mock_openai,
            "create_graph_mock": mock_create_graph,
        }


def test_graph_mcp_ask_success(mock_v3):
    """正常请求应返回 HTTP 200。"""

    graph = mock_v3["graph"]

    graph.ainvoke.return_value = {
        "answer": "葡萄糖检测限为 0.4631 μM。",
        "sources": [
            {
                "page": 6,
                "chunk_id": 66,
                "score": 0.89,
                "preview": "Glucose detection limit"
            }
        ]
    }

    with TestClient(app) as client:
        response = client.post(
            "/graph-mcp/ask",
            json={"question": "葡萄糖检测限是多少？"}
        )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "葡萄糖检测限为 0.4631 μM。"
    assert len(data["sources"]) == 1
    assert data["sources"][0]["page"] == 6

    graph.ainvoke.assert_awaited_once_with(
        {"question": "葡萄糖检测限是多少？"}
    )

    mock_v3["create_graph_mock"].assert_called_once_with(
        mock_v3["openai_mock"].return_value,
        mock_v3["mcp_client"]
    )


def test_graph_mcp_ask_deepseek_error(mock_v3):
    """DeepSeek API 异常组应返回 HTTP 502。"""

    api_error = APIConnectionError(
        message="模拟连接失败",
        request=httpx.Request(
            "POST",
            "https://api.deepseek.com/chat/completions"
        )
    )

    mock_v3["graph"].ainvoke.side_effect = ExceptionGroup(
        "DeepSeek 运行失败",
        [api_error]
    )

    with TestClient(
        app,
        raise_server_exceptions=False
    ) as client:

        response = client.post(
            "/graph-mcp/ask",
            json={"question": "葡萄糖检测限是多少？"}
        )

    assert response.status_code == 502
    assert response.json()["detail"] == "DeepSeek API 请求失败"


def test_graph_mcp_ask_mcp_error(mock_v3):
    """MCP 通信异常组应返回 HTTP 502。"""

    mock_v3["graph"].ainvoke.side_effect = ExceptionGroup(
        "MCP 连接失败",
        [OSError("stdio 连接中断")]
    )

    with TestClient(
        app,
        raise_server_exceptions=False
    ) as client:

        response = client.post(
            "/graph-mcp/ask",
            json={"question": "葡萄糖检测限是多少？"}
        )

    assert response.status_code == 502
    assert response.json()["detail"] == "MCP 服务调用失败"


@pytest.mark.parametrize(
    "answer",
    [None, "", "   "]
)
def test_graph_mcp_ask_invalid_answer(mock_v3, answer):
    """无效回答应返回 HTTP 500。"""

    mock_v3["graph"].ainvoke.return_value = {
        "answer": answer,
        "sources": []
    }

    with TestClient(app) as client:
        response = client.post(
            "/graph-mcp/ask",
            json={"question": "葡萄糖检测限是多少？"}
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Agent 未返回有效回答"


def test_graph_mcp_ask_empty_question(mock_v3):
    """空问题应返回 HTTP 422。"""

    with TestClient(app) as client:
        response = client.post(
            "/graph-mcp/ask",
            json={"question": "   "}
        )

    assert response.status_code == 422

    mock_v3["graph"].ainvoke.assert_not_awaited()


def test_graph_mcp_ask_missing_api_key(monkeypatch):
    """没有 API Key 时应返回 HTTP 500。"""

    monkeypatch.delenv(
        "DEEPSEEK_API_KEY",
        raising=False
    )

    with TestClient(app) as client:
        response = client.post(
            "/graph-mcp/ask",
            json={"question": "葡萄糖检测限是多少？"}
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "缺少 DEEPSEEK_API_KEY"
