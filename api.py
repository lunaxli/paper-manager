import logging
import os
import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from openai import APIError, OpenAI
from pydantic import BaseModel, Field

# Import RAG first so it loads .env before config reads environment variables.
from rag_answer import generate_answer

import config
from agent_runner import run_agent
from graph_agent import create_research_agent
from rag_retrieval import prepare_retriever, retrieve_context_with_sources

import sys
from pathlib import Path

from mcp import Client, StdioServerParameters
from graph_agent import create_mcp_research_agent
from mcp import MCPError

logger = logging.getLogger(__name__)
level = getattr(logging, config.LOG_LEVEL, None)
if not isinstance(level, int):
    raise ValueError("无效的日志级别")
logger.setLevel(level)


app = FastAPI()


class RagAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class AgentAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class SourceInfo(BaseModel):
    page: int
    chunk_id: int
    score: float
    preview: str


class GraphAskResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


@lru_cache(maxsize=1)
def get_retriever():
    return prepare_retriever("data/sample.pdf")


@app.post("/rag/ask")
def rag_ask(req: RagAskRequest):
    all_chunks, model, embeddings = get_retriever()
    context, sources = retrieve_context_with_sources(
        req.question, all_chunks, model, embeddings, k=10
    )
    try:
        answer = generate_answer(req.question, context)
    except APIError:
        raise HTTPException(
            status_code=502,
            detail="AI 服务暂时不可用，请稍后再试"
        )
    return {
        "question": req.question,
        "answer": answer,
        "sources": sources
    }


@app.post("/agent/ask")
def agent_ask(req: AgentAskRequest):

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="缺少 DEEPSEEK_API_KEY"
        )

    all_chunks, model, embeddings = get_retriever()

    client = OpenAI(
        api_key=api_key,
        base_url=config.DEEPSEEK_BASE_URL
    )

    try:
        result = run_agent(
            question=req.question,
            client=client,
            all_chunks=all_chunks,
            model=model,
            embeddings=embeddings
        )
    except APIError:
        raise HTTPException(
            status_code=502,
            detail="AI 服务暂时不可用，请稍后再试"
        )

    return result


@app.post("/graph/ask", response_model=GraphAskResponse)
def graph_ask(request: AgentAskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=422,
            detail="问题不能为空"
        )
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="DEEPSEEK_API_KEY 未配置"
        )

    all_chunks, model, embeddings = get_retriever()

    client = OpenAI(
        api_key=api_key,
        base_url=config.DEEPSEEK_BASE_URL
    )
    graph = create_research_agent(client, all_chunks, model, embeddings)
    start_time = time.perf_counter()
    try:
        state_result = graph.invoke({"question": question})
    except APIError:
        logger.error("DeepSeek API 请求失败", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail="DeepSeek API 请求失败"
        )
    finally:
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        logger.info(f"LangGraph 执行耗时:{elapsed_ms:.2f} ms")

    answer = state_result.get("answer")

    if not isinstance(answer, str) or not answer.strip():
        raise HTTPException(
            status_code=500,
            detail="Agent 未生成有效回答"
        )

    sources = state_result.get("sources", [])
    return {
        "answer": answer,
        "sources": sources
    }


@app.post("/graph-mcp/ask", response_model=GraphAskResponse)
async def graph_mcp_ask(request: AgentAskRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="问题不能为空"
        )

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="缺少 DEEPSEEK_API_KEY"
        )

    llm_client = OpenAI(
        api_key=api_key,
        base_url=config.DEEPSEEK_BASE_URL
    )

    base_dir = Path(__file__).resolve().parent

    server = StdioServerParameters(
        command=sys.executable,
        args=[str(base_dir / "mcp_server.py")],
        cwd=str(base_dir)
    )

    try:
        async with Client(server) as mcp_client:
            graph = create_mcp_research_agent(
                llm_client,
                mcp_client
            )

            state_result = await graph.ainvoke(
                {"question": question}
            )
    except* APIError:
        logger.error(
            "DeepSeek API 请求失败",
            exc_info=True
        )
        raise HTTPException(
            status_code=502,
            detail="DeepSeek API 请求失败"
        )

    except* (MCPError, OSError, RuntimeError):
        logger.error(
            "MCP 服务调用失败",
            exc_info=True
        )
        raise HTTPException(
            status_code=502,
            detail="MCP 服务调用失败"
        )

        
    answer = state_result.get("answer", None)
    if not (isinstance(answer, str) and answer.strip()):
        raise HTTPException(
            status_code=500,
            detail="Agent 未返回有效回答"
        )
    sources = state_result.get("sources", [])

    return {
        "answer": answer,
        "sources": sources
    }

@app.get("/")
def home():
    return {
        "message": "Hello FastAPI"
    }
