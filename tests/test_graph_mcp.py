import asyncio
import json
from unittest.mock import AsyncMock, patch

from graph_agent import make_mcp_tool_node
from types import SimpleNamespace
from unittest.mock import MagicMock

from graph_agent import create_mcp_research_agent


def test_mcp_tool_node_calls_client():
    """验证 LangGraph 节点调用 MCP 工具。"""

    fake_client = object()

    tool_calls = [
        {
            "id": "call_1",
            "name": "search_paper",
            "arguments": {"query": "glucose LOD"},
        }
    ]

    expected_data = {
        "context": "Glucose detection limit",
        "sources": [
            {
                "page": 6,
                "chunk_id": 66,
                "score": 0.85,
                "preview": "Detection limit",
            }
        ],
    }

    with patch(
        "graph_agent.call_search_paper",
        new_callable=AsyncMock,
    ) as mock_search:

        mock_search.return_value = expected_data

        node = make_mcp_tool_node(fake_client)

        result = asyncio.run(
            node({"tool_calls": tool_calls})
        )

        mock_search.assert_awaited_once_with(
            fake_client,
            "glucose LOD",
        )

    assert len(result["tool_messages"]) == 1

    message = result["tool_messages"][0]

    assert message["role"] == "tool"
    assert message["tool_call_id"] == "call_1"
    assert json.loads(message["content"]) == expected_data

    assert result["sources"] == expected_data["sources"]


def test_mcp_tool_node_deduplicates_sources():
    """重复片段保留最高分，并按分数排序。"""

    fake_client = object()

    tool_calls = [
        {
            "id": "call_1",
            "name": "search_paper",
            "arguments": {"query": "query A"},
        },
        {
            "id": "call_2",
            "name": "search_paper",
            "arguments": {"query": "query B"},
        },
    ]

    first_result = {
        "context": "First search",
        "sources": [
            {
                "page": 6,
                "chunk_id": 66,
                "score": 0.65,
                "preview": "Old result",
            },
            {
                "page": 5,
                "chunk_id": 49,
                "score": 0.70,
                "preview": "Another result",
            },
        ],
    }

    second_result = {
        "context": "Second search",
        "sources": [
            {
                "page": 6,
                "chunk_id": 66,
                "score": 0.89,
                "preview": "Better result",
            }
        ],
    }

    with patch(
        "graph_agent.call_search_paper",
        new_callable=AsyncMock,
    ) as mock_search:

        mock_search.side_effect = [
            first_result,
            second_result,
        ]

        node = make_mcp_tool_node(fake_client)

        result = asyncio.run(
            node({"tool_calls": tool_calls})
        )

        assert mock_search.await_count == 2

    assert len(result["tool_messages"]) == 2
    assert len(result["sources"]) == 2

    assert result["sources"][0]["page"] == 6
    assert result["sources"][0]["chunk_id"] == 66
    assert result["sources"][0]["score"] == 0.89

    assert result["sources"][1]["score"] == 0.70


def test_mcp_graph_complete_search_flow():
    """验证完整的 LangGraph + MCP 检索回答路径。"""

    fake_llm = MagicMock()
    fake_mcp_client = object()

    question = "根据论文，葡萄糖检测限是多少？"

    tool_calls = [
        {
            "id": "call_1",
            "name": "search_paper",
            "arguments": {
                "query": "glucose limit of detection"
            }
        }
    ]

    mcp_data = {
        "context": "The glucose LOD is 0.4631 μM.",
        "sources": [
            {
                "page": 6,
                "chunk_id": 66,
                "score": 0.89,
                "preview": "glucose limit of detection"
            }
        ]
    }

    # 模拟第二次 DeepSeek 调用生成最终回答
    fake_llm.chat.completions.create.return_value = (
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="葡萄糖检测限为 0.4631 μM。"
                    )
                )
            ]
        )
    )

    with patch(
        "graph_agent.request_tool_decision",
        return_value=SimpleNamespace(content=None)
    ) as mock_decision, patch(
        "graph_agent.parse_tool_calls",
        return_value=tool_calls
    ), patch(
        "graph_agent.call_search_paper",
        new_callable=AsyncMock
    ) as mock_search:

        mock_search.return_value = mcp_data

        graph = create_mcp_research_agent(
            fake_llm,
            fake_mcp_client
        )

        result = asyncio.run(
            graph.ainvoke({"question": question})
        )

    # 1. DeepSeek 完成工具决策
    mock_decision.assert_called_once()

    # 2. MCP 工具真正被异步调用
    mock_search.assert_awaited_once_with(
        fake_mcp_client,
        "glucose limit of detection"
    )

    # 3. 工作流生成最终回答
    assert result["answer"] == (
        "葡萄糖检测限为 0.4631 μM。"
    )

    # 4. 检索来源保留
    assert result["sources"] == mcp_data["sources"]

    # 5. 工具消息保留
    assert len(result["tool_messages"]) == 1

    # 6. 最终回答节点调用了 DeepSeek
    fake_llm.chat.completions.create.assert_called_once()

    messages = (
        fake_llm.chat.completions.create.call_args
        .kwargs["messages"]
    )

    assert messages[-1]["role"] == "tool"
    assert messages[-1]["tool_call_id"] == "call_1"
    assert json.loads(messages[-1]["content"]) == mcp_data


def test_mcp_graph_direct_answer_flow():
    """普通问候应直接回答，不调用 MCP 工具。"""

    fake_llm = MagicMock()
    fake_mcp_client = object()

    with patch(
        "graph_agent.request_tool_decision",
        return_value=SimpleNamespace(
            content="你好，我是科研论文助手。"
        )
    ), patch(
        "graph_agent.parse_tool_calls",
        return_value=[]
    ), patch(
        "graph_agent.call_search_paper",
        new_callable=AsyncMock
    ) as mock_search:

        graph = create_mcp_research_agent(
            fake_llm,
            fake_mcp_client
        )

        result = asyncio.run(
            graph.ainvoke({"question": "  你好  "})
        )

    assert result["search_query"] == "你好"

    assert result["answer"] == (
        "你好，我是科研论文助手。"
    )

    assert result["need_search"] is False

    assert result.get("sources", []) == []

    # 不应该执行 MCP 检索
    mock_search.assert_not_awaited()

    # 不应该进入第二次模型回答节点
    fake_llm.chat.completions.create.assert_not_called()
