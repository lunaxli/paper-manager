
import asyncio
from unittest.mock import patch

import pytest
from mcp import Client

import mcp_server


@pytest.fixture(autouse=True)
def clear_retriever_cache():
    mcp_server.get_mcp_retriever.cache_clear()
    yield
    mcp_server.get_mcp_retriever.cache_clear()


def test_mcp_retriever_cache():
    """重复获取检索资源时，只初始化一次。"""

    resources = ("chunks", "model", "embeddings")

    with patch(
        "mcp_server.prepare_retriever",
        return_value=resources
    ) as mock_prepare:

        first = mcp_server.get_mcp_retriever()
        second = mcp_server.get_mcp_retriever()

    assert first == resources
    assert first is second

    mock_prepare.assert_called_once_with("data/sample.pdf")


def test_search_paper_mcp_calls_existing_tool():
    """MCP 封装必须复用原有 search_paper。"""

    expected = {
        "context": "glucose detection limit",
        "sources": []
    }

    with patch(
        "mcp_server.get_mcp_retriever",
        return_value=("chunks", "model", "embeddings")
    ) as mock_retriever:

        with patch(
            "mcp_server.search_paper",
            return_value=expected
        ) as mock_search:

            result = mcp_server.search_paper_mcp(
                "  glucose LOD  "
            )

    assert result == expected

    mock_retriever.assert_called_once_with()

    mock_search.assert_called_once_with(
        "glucose LOD",
        "chunks",
        "model",
        "embeddings"
    )


@pytest.mark.parametrize(
    "query",
    ["", "   ", "\n\t"]
)
def test_search_paper_mcp_rejects_empty_query(query):
    """空问题不得触发检索资源加载。"""

    with patch(
        "mcp_server.get_mcp_retriever"
    ) as mock_retriever:

        with pytest.raises(
            ValueError,
            match="query 不能为空"
        ):
            mcp_server.search_paper_mcp(query)

    mock_retriever.assert_not_called()


def test_mcp_client_discovers_and_calls_tool():
    """通过 MCP Client 验证注册和调用协议。"""

    expected = {
        "context": "glucose detection limit",
        "sources": [
            {
                "page": 6,
                "chunk_id": 66,
                "score": 0.8,
                "preview": "glucose detection"
            }
        ]
    }

    async def run_client():
        async with Client(mcp_server.mcp) as client:

            # 检查工具是否已注册
            tools_result = await client.list_tools()

            tool = next(
                tool
                for tool in tools_result.tools
                if tool.name == "search_paper"
            )

            assert "query" in tool.input_schema["properties"]
            assert "query" in tool.input_schema["required"]

            # 真正通过 MCP 协议调用工具
            result = await client.call_tool(
                "search_paper",
                {"query": "  glucose LOD  "}
            )

            assert result.is_error is False
            assert result.structured_content == expected

    with patch(
        "mcp_server.get_mcp_retriever",
        return_value=("chunks", "model", "embeddings")
    ):
        with patch(
            "mcp_server.search_paper",
            return_value=expected
        ) as mock_search:

            asyncio.run(run_client())

    mock_search.assert_called_once_with(
        "glucose LOD",
        "chunks",
        "model",
        "embeddings"
    )
