from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
import pymupdf
from fastapi.testclient import TestClient
from openai import APIConnectionError

import api
import rag_answer
import rag_retrieval


def test_rag_ask_uses_retrieved_pdf_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Synthetic evidence from page one.")
        pdf.save("data/sample.pdf")

    embedder = Mock()
    embedder.embed.side_effect = lambda texts: [np.array([1.0, 0.0]) for _ in texts]
    monkeypatch.setattr(rag_retrieval, "TextEmbedding", Mock(return_value=embedder))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    openai = Mock()
    monkeypatch.setattr(rag_answer, "OpenAI", openai)
    create = openai.return_value.chat.completions.create
    create.return_value = SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content="示例回答")),
    ])
    api.get_retriever.cache_clear()
    try:
        response = TestClient(api.app).post("/rag/ask", json={"question": "What does the paper say?"})
    finally:
        api.get_retriever.cache_clear()

    assert response.status_code == 200
    result = response.json()
    assert result["question"] == "What does the paper say?"
    assert result["answer"] == "示例回答"
    assert result["sources"][0]["page"] == 1
    assert result["sources"][0]["score"] == 1.0
    assert "Synthetic evidence" in result["sources"][0]["preview"]
    assert "Synthetic evidence" in create.call_args.kwargs["messages"][1]["content"]


def test_rag_ask_api_error(monkeypatch):
    monkeypatch.setattr(api, "get_retriever", Mock(return_value=([], None, [])))
    monkeypatch.setattr(api, "retrieve_context_with_sources", Mock(return_value=("示例证据", [])))
    monkeypatch.setattr(api, "generate_answer", Mock(side_effect=APIConnectionError(
        request=httpx.Request("POST", "https://example.invalid"),
    )))

    response = TestClient(api.app).post("/rag/ask", json={"question": "测试"})

    assert response.status_code == 502
    assert response.json()["detail"] == "AI 服务暂时不可用，请稍后再试"


def test_rag_ask_invalid_question():
    response = TestClient(api.app).post("/rag/ask", json={"question": ""})

    assert response.status_code == 422
