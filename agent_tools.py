from rag_retrieval import retrieve_context_with_sources
PAPER_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_paper",

        "description":"用于从已加载的科研论文中检索相关证据，获取和问题相关的论文片段、页码信息",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type":"string",
                    "description":"论文检索的查询问题，用来在论文库里面查找相关内容"
                }
            },

            "required": ["query"],

            "additionalProperties": False
        }
    }
}


def search_paper(query, all_chunks, model, embeddings):
    context,sources = retrieve_context_with_sources(query,all_chunks,model,embeddings,k=10)
    return{
        "context": context,
        "sources": sources
    }


def execute_tool(tool_name, arguments, all_chunks, model, embeddings):

    if tool_name != "search_paper":
        raise ValueError("不支持的工具：" + tool_name)

    if not isinstance(arguments, dict):
        raise ValueError("工具参数必须是字典")

    query = arguments.get("query")

    if not isinstance(query, str):
        raise ValueError("query 必须是字符串")

    query = query.strip()

    if not query:
        raise ValueError("query 不能为空")

    return search_paper(query, all_chunks, model, embeddings)
