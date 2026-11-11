import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

import config
from agent_runner import (
    collect_tool_sources,
    execute_tool_calls,
    parse_tool_calls,
    request_tool_decision,
)

from mcp_client import call_search_paper

V2_SEARCH_PROMPT = (
    "你是一名科研论文分析助手。"
    "涉及论文内容、实验数据或研究结论时，"
    "应调用 search_paper 工具获取证据，不要编造信息。"
    "当前检索的科研论文主要使用英文。"
    "调用 search_paper 时，应优先生成简洁、准确的英文检索关键词。"
    "关键词应聚焦用户询问的研究对象和具体指标，"
    "尽量使用论文中的常见专业术语。"
    "避免使用宽泛、不相关或重复的检索关键词。"
    "只有在需要检索不同证据时，才发起多次工具调用。"
    "对于不涉及论文内容的普通问候，可以直接回答。"
)


class ResearchState(TypedDict, total=False):
    question: str
    search_query: str
    answer: str
    sources: list[dict]
    need_search: bool
    tool_calls: list[dict]
    tool_messages: list[dict]


def prepare_question(state: ResearchState):
    question = state["question"].strip()
    if not question:
        raise ValueError("问题不能为空")
    return {"search_query": question}


def route_after_decision(state: ResearchState) -> str:

    need_search = state["need_search"]
    if need_search:
        return "search"
    return "direct"


def make_decision_node(client):

    def decide_tool(state: ResearchState):

        question = state["search_query"]
        message = request_tool_decision(question, client, system_prompt=V2_SEARCH_PROMPT)
        parsed_calls = parse_tool_calls(message)
        if not parsed_calls:
            return {"need_search": False, "answer": message.content, "tool_calls": []}
        return {"need_search": True, "tool_calls": parsed_calls}

    return decide_tool


def make_tool_node(all_chunks, model, embeddings):

    def execute_tools(state: ResearchState):

        tool_calls = state["tool_calls"]

        tool_messages = execute_tool_calls(
            tool_calls,
            all_chunks,
            model,
            embeddings
        )

        # Keep the highest scoring source for each page and chunk in V2.
        unique_sources = {}
        for source in collect_tool_sources(tool_messages):
            page = source["page"]
            chunk_id = source["chunk_id"]
            key = (page, chunk_id)

            if key not in unique_sources or source["score"] > unique_sources[key]["score"]:
                unique_sources[key] = source

        sorted_sources = sorted(unique_sources.values(), key=lambda x: x["score"], reverse=True)

        return {
            "tool_messages": tool_messages,
            "sources": sorted_sources
        }

    return execute_tools


def make_mcp_tool_node(mcp_client):

    async def mcp_tool_node(state: ResearchState):

        tool_calls = state.get("tool_calls", [])

        tool_messages = []
        sources_by_key = {}

        for call in tool_calls:

            arguments = call["arguments"]
            if isinstance(arguments,str):
                arguments = json.loads(arguments)
            query = arguments["query"]
            tool_call_id = call["id"]

            data = await call_search_paper(
                mcp_client,
                query
            )

            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(
                    data,
                    ensure_ascii=False
                )
            }
            tool_messages.append(tool_message)
            
            for source in data["sources"]:

                key = (
                    source["page"],
                    source["chunk_id"]
                )

                old_source = sources_by_key.get(key)

                if (
                    old_source is None
                    or source["score"] > old_source["score"]
                ):
                    sources_by_key[key] = source

        sources = sorted(
            sources_by_key.values(),
            key=lambda source: source["score"],
            reverse=True
        )

        return {
            "tool_messages": tool_messages,
            "sources": sources
        }

    return mcp_tool_node


def make_final_answer_node(client):

    def generate_final_answer(state: ResearchState):

        question = state["search_query"]
        tool_calls = state["tool_calls"]
        tool_messages = state["tool_messages"]

        assistant_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(
                            call["arguments"],
                            ensure_ascii=False
                        )
                    }
                }
                for call in tool_calls
            ]
        }

        followup_messages = [
            {
                "role": "system",
                "content": "你是科研论文分析助手，根据检索到的论文证据回答用户问题，不要编造信息。"
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
        return {"answer": answer}
    return generate_final_answer


def create_research_agent(client, all_chunks, model, embeddings):
    builder = StateGraph(ResearchState)
    builder.add_node("prepare_question", prepare_question)
    builder.add_node("decide_tool", make_decision_node(client))
    builder.add_node("execute_tools", make_tool_node(all_chunks, model, embeddings))
    builder.add_node("generate_final_answer", make_final_answer_node(client))

    builder.add_edge(START, "prepare_question")
    builder.add_edge("prepare_question", "decide_tool")
    builder.add_conditional_edges(
        "decide_tool",
        route_after_decision,
        {"search": "execute_tools", "direct": END},
    )
    builder.add_edge("execute_tools", "generate_final_answer")
    builder.add_edge("generate_final_answer", END)
    return builder.compile()


def create_mcp_research_agent(client, mcp_client):
    builder = StateGraph(ResearchState)
    builder.add_node("prepare_question",prepare_question)
    builder.add_node("decide_tool",make_decision_node(client))
    builder.add_node("execute_tools",make_mcp_tool_node(mcp_client))
    builder.add_node("generate_final_answer",make_final_answer_node(client))
    
    builder.add_edge(START, "prepare_question")
    builder.add_edge("prepare_question","decide_tool")
    builder.add_conditional_edges(
        "decide_tool",
        route_after_decision,
        {
            "search": "execute_tools",
            "direct": END
        }
    )
    builder.add_edge("execute_tools","generate_final_answer")
    builder.add_edge("generate_final_answer",END)
    return builder.compile()
