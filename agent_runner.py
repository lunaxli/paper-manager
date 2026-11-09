import json
from agent_tools import PAPER_SEARCH_TOOL,execute_tool
import config


def request_tool_decision(question, client, system_prompt=None):
    default_prompt = (
        "你是一名科研论文分析助手。"
        "涉及论文内容、实验数据和研究结论时，"
        "应使用 search_paper 工具获取证据，"
        "不要编造论文中的信息。"
    )
    if system_prompt is None:
        actual_prompt = default_prompt
    else:
        actual_prompt = system_prompt
    messages = [
        {
            "role": "system",
            "content": actual_prompt
        },
        {
            "role": "user",
            "content": question
        }
    ]
    response = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages = messages,
        tools = [PAPER_SEARCH_TOOL],
        tool_choice = "auto",
        extra_body={"thinking": {"type": "disabled"}},
        max_tokens=config.DEEPSEEK_MAX_TOKENS
    )

    message = response.choices[0].message
    return message


def parse_tool_calls(message):
    parsed_calls = []

    if not message.tool_calls:
        return parsed_calls

    for tool_call in message.tool_calls:

        call_id = tool_call.id

        tool_name = tool_call.function.name

        arguments = json.loads(tool_call.function.arguments)

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


def collect_tool_sources(tool_messages):
    """Collect sources in tool response order, including duplicates."""
    sources = []
    for tool_message in tool_messages:
        tool_result = json.loads(tool_message["content"])
        sources.extend(tool_result.get("sources", []))
    return sources


def run_agent(question, client, all_chunks, model, embeddings):

    message = request_tool_decision(question, client)

    parsed_calls = parse_tool_calls(message)

    if not parsed_calls:
        return {
            "answer":message.content,
            "sources": []
        }

    tool_messages = execute_tool_calls(parsed_calls,all_chunks,model,embeddings)
    sources = collect_tool_sources(tool_messages)
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
        {
            "role": "system",
            "content": "你是一名科研论文分析助手。涉及论文内容、实验数据和研究结论时，应使用 search_paper 工具获取证据，不要编造论文中的信息。"
        },
        {
            "role": "user",
            "content": question
        },
        assistant_message,
        *tool_messages
    ]
    response = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=followup_messages,
        extra_body={"thinking": {"type": "disabled"}},
        max_tokens=config.DEEPSEEK_MAX_TOKENS
    )
    answer = response.choices[0].message.content
    return {
        "answer": answer,
        "sources": sources
    }
