from functools import lru_cache
from mcp.server import MCPServer
from agent_tools import search_paper
from rag_retrieval import prepare_retriever
from typing import Any


mcp = MCPServer("ResearchAgent")


@lru_cache(maxsize=1)
def get_mcp_retriever():
    all_chunks,model,embeddings = prepare_retriever("data/sample.pdf")
    return all_chunks,model,embeddings

@mcp.tool(name="search_paper")
def search_paper_mcp(query: str) -> dict[str,Any]:
    """从本地科研论文中检索相关文本和来源信息。"""

    query = query.strip()
    if not query:
        raise ValueError("query 不能为空")

    all_chunks, model, embeddings = get_mcp_retriever()

    result = search_paper(query, all_chunks, model, embeddings)
    return result

if __name__ == "__main__":
    mcp.run()
