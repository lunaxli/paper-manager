import json
from agent_tools import PAPER_SEARCH_TOOL,execute_tool

def request_tool_decision(question, client):
    messages = [
        {
            "role": "system",
            "content": (
                "你是一名科研论文分析助手。"
                "涉及论文内容、实验数据和研究结论时，"
                "应使用 search_paper 工具获取证据，"
                "不要编造论文中的信息。"
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    response = client.chat.completions.create(
        model="deepseek-flash",
        
        # TODO 1：
        # 将 messages 传给模型
        messages = messages,

        # TODO 2：
        # 使用 tools 参数，
        # 将 PAPER_SEARCH_TOOL 放入列表
        tools = [PAPER_SEARCH_TOOL],

        # TODO 3：
        # 设置 tool_choice="auto"
        tool_choice = "auto",

        extra_body={"thinking": {"type": "disabled"}},
        max_tokens=700
    )

    # TODO 4：
    # 获取 response.choices[0].message
    # 并返回该对象
    message = response.choices[0].message
    return message


def parse_tool_calls(message):
    parsed_calls = []

    if not message.tool_calls:
        return parsed_calls

    for tool_call in message.tool_calls:
        # TODO 1：获取工具调用 ID
        call_id = tool_call.id

        # TODO 2：获取工具名称
        tool_name = tool_call.function.name

        # TODO 3：使用 json.loads()
        # 将 JSON 字符串解析成 Python 字典
        arguments = json.loads(tool_call.function.arguments)

        # TODO 4：将字典添加到 parsed_calls
        parsed_calls.append({
            "id":call_id,
            "name":tool_name,
            "arguments":arguments
        })

    return parsed_calls

def execute_tool_calls(parsed_calls, all_chunks, model, embeddings):

    tool_messages = []

    for call in parsed_calls:
        tool_result = execute_tool(call["name"],call["arguments"],all_chunks,model,embeddings)

        tool_message = {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": json.dumps(tool_result, ensure_ascii=False)
        }
        tool_messages.append(tool_message)

    return tool_messages

def run_agent(question, client, all_chunks, model, embeddings):

    # 1. 让模型决定是否调用工具
    message = request_tool_decision(question, client)

    # 2. 解析工具调用
    parsed_calls = parse_tool_calls(message)

    if not parsed_calls:
        return {
            "answer":message.content,
            "sources": []
        }

    tool_messages = execute_tool_calls(parsed_calls,all_chunks,model,embeddings)
    sources = []
    for tool_message in tool_messages:
        # 1. 将 JSON 字符串转换成 Python 字典
        tool_result = json.loads(tool_message["content"])
        # 2. 提取工具返回的 sources
        tool_sources = tool_result.get("sources", [])
        # 3. 将来源添加到总列表
        sources.extend(tool_sources)
    assistant_message = {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            }
            for tc in message.tool_calls
        ]
    }
    followup_messages = [
        # 1. system 消息
        {
            "role": "system",
            "content": "你是一名科研论文分析助手。涉及论文内容、实验数据和研究结论时，应使用 search_paper 工具获取证据，不要编造论文中的信息。"
        },
        # 2. user 原始问题
        {
            "role": "user",
            "content": question
        },
        # 3. assistant 消息：模型上一轮想要调用工具的响应
        assistant_message,
        # 4. 展开工具返回结果列表
        *tool_messages
    ]
    # 第二次 DeepSeek 请求
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=followup_messages,
        extra_body={"thinking": {"type": "disabled"}},
        max_tokens=700
    )
    # 提取最终回答
    answer = response.choices[0].message.content
    return {
        "answer": answer,
        "sources": sources
    }

    raise NotImplementedError("第二次模型请求尚未实现")