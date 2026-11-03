
import os
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError

from api import app


client = TestClient(app)


# 测试1：正常请求
def test_agent_ask_success():
    fake_chunks = [{"chunk_id": 1, "text": "测试论文内容"}]
    fake_model = object()
    fake_embeddings = [[0.1, 0.2]]

    fake_result = {
        "answer": "葡萄糖检测限为 0.4631 μM",
        "sources": [
            {"page": 6, "chunk_id": 66, "score": 0.8}
        ]
    }

    with patch.dict(
        os.environ, {"DEEPSEEK_API_KEY": "test-key"}
    ), patch(
        "api.get_retriever",
        return_value=(fake_chunks, fake_model, fake_embeddings)
    ) as mock_retriever, patch(
        "api.OpenAI"
    ) as mock_openai, patch(
        "api.run_agent",
        return_value=fake_result
    ) as mock_agent:

        response = client.post(
            "/agent/ask",
            json={"question": "葡萄糖检测限是多少？"}
        )

        assert response.status_code == 200
        assert response.json() == fake_result

        mock_retriever.assert_called_once()

        mock_openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.deepseek.com"
        )

        mock_agent.assert_called_once_with(
            question="葡萄糖检测限是多少？",
            client=mock_openai.return_value,
            all_chunks=fake_chunks,
            model=fake_model,
            embeddings=fake_embeddings
        )


# 测试2：缺少 API Key
def test_agent_ask_missing_api_key():
    with patch.dict(
        os.environ, {"DEEPSEEK_API_KEY": ""}
    ), patch("api.get_retriever") as mock_retriever:

        response = client.post(
            "/agent/ask",
            json={"question": "测试问题"}
        )

        assert response.status_code == 500
        assert response.json()["detail"] == (
            "缺少 DEEPSEEK_API_KEY"
        )

        mock_retriever.assert_not_called()


# 测试3：DeepSeek 调用失败
def test_agent_ask_api_error():
    fake_error = APIConnectionError(
        request=httpx.Request(
            "POST",
            "https://api.deepseek.com/chat/completions"
        )
    )

    with patch.dict(
        os.environ, {"DEEPSEEK_API_KEY": "test-key"}
    ), patch(
        "api.get_retriever",
        return_value=([], None, None)
    ), patch(
        "api.OpenAI"
    ), patch(
        "api.run_agent",
        side_effect=fake_error
    ):

        response = client.post(
            "/agent/ask",
            json={"question": "测试问题"}
        )

        assert response.status_code == 502
        assert response.json()["detail"] == (
            "AI 服务暂时不可用，请稍后再试"
        )


# 测试4：非法请求参数
@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "a" * 1001},
        {}
    ]
)
def test_agent_ask_invalid_question(payload):
    response = client.post(
        "/agent/ask",
        json=payload
    )

    assert response.status_code == 422
