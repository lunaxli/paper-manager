import sqlite3
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import HTTPException
from pydantic import Field
import os
from dotenv import load_dotenv
from functools import lru_cache
from rag_retrieval import prepare_retriever,retrieve_context_with_sources
from rag_answer import generate_answer
from agent_runner import run_agent

load_dotenv()

from openai import OpenAI, APIError

app = FastAPI()


class PaperCreate(BaseModel):
    title: str = Field(min_length=1,max_length=200)
    year: int = Field(ge=1900,le=2100)


class PaperUpdate(BaseModel):
    year: int = Field(ge=1900,le=2100)


class AskRequest(BaseModel):
    question:str = Field(min_length=1,max_length=1000)

class RagAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

class AgentAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@lru_cache(maxsize=1)
def get_retriever():
    all_chunks,model,embeddings = prepare_retriever("data/sample.pdf")
    return all_chunks,model,embeddings

@app.post("/rag/ask")
def rag_ask(req:RagAskRequest):
    all_chunks,model,embeddings = get_retriever()
    context,sources = retrieve_context_with_sources(req.question,all_chunks,model,embeddings,k=10)
    try:
        answer = generate_answer(req.question,context)
    except APIError:
        raise HTTPException(
            status_code=502,
            detail="AI 服务暂时不可用，请稍后再试"
        )
    return {
        "question":req.question,
        "answer":answer,
        "sources":sources
    }

@app.post("/agent/ask")
def agent_ask(req: AgentAskRequest):

    # 第一步：先检查 API Key
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="缺少 DEEPSEEK_API_KEY"
        )

    # 第二步：确认 API Key 存在后，再加载检索资源
    all_chunks, model, embeddings = get_retriever()

    # 第三步：创建 DeepSeek 客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    # 第四步：运行 Agent
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

@app.get("/")
def home():
    return {
        "message": "Hello FastAPI"
    }


@app.get("/papers")
def get_papers():

    connection = sqlite3.connect("papers.db")
    cursor = connection.cursor()

    cursor.execute(
        "SELECT * FROM papers"
    )

    rows = cursor.fetchall()

    connection.close()

    papers = []

    for row in rows:
        paper = {
            "id": row[0],
            "title": row[1],
            "year": row[2]
        }

        papers.append(paper)

    return papers


@app.get("/papers/{paper_id}")
def get_paper(paper_id: int):

    connection = sqlite3.connect("papers.db")
    cursor = connection.cursor()

    cursor.execute(
        "SELECT * FROM papers WHERE id = ?",
        (paper_id,)
    )

    row = cursor.fetchone()

    connection.close()

    if row is None:
        return {
            "message": "没有找到论文"
        }

    return {
        "id": row[0],
        "title": row[1],
        "year": row[2]
    }


@app.post("/papers")
def create_paper(paper: PaperCreate):

    connection = sqlite3.connect("papers.db")
    cursor = connection.cursor()

    cursor.execute(
        "INSERT INTO papers (title, year) VALUES (?, ?)",
        (
            paper.title,
            paper.year
        )
    )

    connection.commit()

    paper_id = cursor.lastrowid

    connection.close()

    return {
        "id": paper_id,
        "title": paper.title,
        "year": paper.year
    }


@app.delete("/papers/{paper_id}")
def delete_paper(paper_id:int):

    connection = sqlite3.connect("papers.db")
    cursor = connection.cursor()

    cursor.execute(
        "DELETE FROM papers WHERE id = ?",
        (paper_id,)
    )

    connection.commit()

    if cursor.rowcount > 0:
        result = {"msg":"删除成功"}
        connection.close()
        return result

    else:
        connection.close()
        raise HTTPException(status_code=404,detail="论文不存在")


@app.put("/papers/{paper_id}")
def update_paper(paper_id: int, paper: PaperUpdate):

    connection = sqlite3.connect("papers.db")

    try:
        cursor = connection.cursor()

        cursor.execute(
            "UPDATE papers SET year = ? WHERE id = ?",
            (paper.year, paper_id)
        )

        affected_rows = cursor.rowcount
        connection.commit()

        if affected_rows > 0:
            return {
                "message": "修改成功"
            }
        else:
            raise HTTPException(
                status_code=404,
                detail="没有这个论文"
            )

    finally:
        connection.close()


#09-6：让 FastAPI 真正调用 DeepSeek
#09-7｜AI 接口异常处理

@app.post("/ai/ask")
def ask(request: AskRequest):
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError("没有找到 DEEPSEEK_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    try:
        response = client.chat.completions.create(
            model="deepseek-flash",
            messages=[
                {
                    "role":"user",
                    "content":request.question
                }
            ]
        )

        answer = response.choices[0].message.content
        return {"answer":answer}
    except APIError:
        raise HTTPException(
            status_code=502,
            detail="AI 服务暂时不可用，请稍后再试"
        )

