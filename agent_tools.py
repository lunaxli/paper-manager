from rag_retrieval import retrieve_context_with_sources
PAPER_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_paper",

        # TODO 1：
        # 添加 description，告诉模型这个工具
        # 用于从已加载的科研论文中检索相关证据
        "description":"用于从已加载的科研论文中检索相关证据，获取和问题相关的论文片段、页码信息",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    # TODO 2：
                    # 指定 query 的类型为字符串
                    "type":"string",
                    # TODO 3：
                    # 添加 description，
                    # 说明 query 是论文检索问题
                    "description":"论文检索的查询问题，用来在论文库里面查找相关内容"
                }
            },

            # TODO 4：
            # 设置 query 为必填参数
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

    # 1. 检查工具名称
    if tool_name != "search_paper":
        raise ValueError("不支持的工具：" + tool_name)

    # 2. 检查 arguments 是否为字典
    if not isinstance(arguments, dict):
        raise ValueError("工具参数必须是字典")

    # 3. 获取 query
    query = arguments.get("query")

    # 4. 检查 query 是否为字符串
    if not isinstance(query, str):
        raise ValueError("query 必须是字符串")

    # 5. 检查 query 是否为空
    query = query.strip()

    if not query:
        raise ValueError("query 不能为空")

    # 6. 执行工具
    return search_paper(query, all_chunks, model, embeddings)