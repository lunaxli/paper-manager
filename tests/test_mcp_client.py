import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import mcp_client


def test_mcp_client_calls_search_tool(capsys):
    """客户端成功发现并调用 MCP 工具。"""

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

    fake_client = SimpleNamespace(
        list_tools=AsyncMock(
            return_value=SimpleNamespace(
                tools=[SimpleNamespace(name="search_paper")]
            )
        ),
        call_tool=AsyncMock(
            return_value=SimpleNamespace(
                is_error=False,
                structured_content=expected
            )
        )
    )

    with patch("mcp_client.Client") as mock_client:
        mock_client.return_value.__aenter__.return_value = (
            fake_client
        )

        asyncio.run(mcp_client.main())

        # 验证独立服务器启动参数
        server = mock_client.call_args.args[0]

        assert server.command == sys.executable
        assert server.args == [
            str(mcp_client.BASE_DIR / "mcp_server.py")
        ]
        assert server.cwd == str(mcp_client.BASE_DIR)

    fake_client.list_tools.assert_awaited_once_with()

    fake_client.call_tool.assert_awaited_once_with(
        "search_paper",
        {"query": "glucose limit of detection LOD"}
    )

    output = capsys.readouterr().out

    assert "工具名称：search_paper" in output
    assert "glucose detection limit" in output
    assert "sources 数量： 1" in output


def test_mcp_client_handles_tool_error():
    """MCP 工具失败时应抛出 RuntimeError。"""

    fake_client = SimpleNamespace(
        list_tools=AsyncMock(
            return_value=SimpleNamespace(
                tools=[SimpleNamespace(name="search_paper")]
            )
        ),
        call_tool=AsyncMock(
            return_value=SimpleNamespace(
                is_error=True,
                structured_content=None
            )
        )
    )

    with patch("mcp_client.Client") as mock_client:
        mock_client.return_value.__aenter__.return_value = (
            fake_client
        )

        with pytest.raises(
            RuntimeError,
            match="MCP 工具调用失败"
        ):
            asyncio.run(mcp_client.main())

    fake_client.call_tool.assert_awaited_once_with(
        "search_paper",
        {"query": "glucose limit of detection LOD"}
    )
