#09-7｜编写第一条 AI 异常测试

from unittest.mock import patch
import httpx
from fastapi.testclient import TestClient
from openai import APIConnectionError

from api import app


def test_ai_connection_error(monkeypatch):
    # 1. 使用假的 Key，避免依赖真实密钥
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    # 2. 替换 api.py 使用的 OpenAI 客户端
    with patch("api.OpenAI") as mock_openai:

        # 模拟 DeepSeek 连接失败
        mock_openai.return_value.chat.completions.create.side_effect = (
            APIConnectionError(
                request=httpx.Request(
                    "POST",
                    "https://api.deepseek.com"
                )
            )
        )

        # 3. 向 FastAPI 发送请求
        client = TestClient(app)
        response = client.post(
            "/ai/ask",
            json={"question": "什么是 RAG？"}
        )

    assert response.status_code == 502
    assert response.json()["detail"] == "AI 服务暂时不可用，请稍后再试"
