import asyncio
import sys
from pathlib import Path
from typing import Any
from mcp import Client, StdioServerParameters


BASE_DIR = Path(__file__).resolve().parent

async def call_search_paper(client: Client, query: str) -> dict[str, Any]:
    query = query.strip()

    if not query:
        raise ValueError("query 不能为空")

    result = await client.call_tool(
        "search_paper",
        {"query": query}
    )

    if result.is_error:
        raise RuntimeError("MCP 工具调用失败")

    data = result.structured_content

    if not isinstance(data, dict):
        raise ValueError("MCP 返回数据格式错误")

    return data

async def main():

    server = StdioServerParameters(
        command=sys.executable,
        args=[str(BASE_DIR / "mcp_server.py")],
        cwd=str(BASE_DIR),
    )

    async with Client(server) as client:
        tools_result = await client.list_tools()
        for tool in tools_result.tools:
            print(f"工具名称：{tool.name}")

        data = await call_search_paper(
            client,
            "glucose limit of detection LOD"
        )

        print(data["context"][:300])
        print("sources 数量：", len(data["sources"]))


if __name__ == "__main__":
    asyncio.run(main())
