from unittest.mock import patch
import httpx
from fastapi.testclient import TestClient
from openai import APIConnectionError
from api import app

client = TestClient(app)


def test_rag_ask_api_error():
    # 同时模拟三个外部依赖
    with (
        patch("api.get_retriever") as mock_retriever,
        patch("api.retrieve_context_with_sources") as mock_retrieve,
        patch("api.generate_answer") as mock_answer
    ):
        # 不读取真实 PDF，不加载模型
        mock_retriever.return_value = ([], None, [])

        # 不执行真实向量检索
        mock_retrieve.return_value = (
            "模拟论文内容",
            [
                {
                    "page":6,
                    "chunk_id":66,
                    "score":0.8,
                    "preview":"模拟葡萄糖检测限相关论文内容"
                }
            ]
        )

        # TODO 1：让 mock_answer 在调用时抛出
        # APIConnectionError
        # 提示：使用 side_effect
        # request=httpx.Request(
        #     "POST", "https://api.deepseek.com"
        # )
        mock_answer.side_effect = APIConnectionError(
            request = httpx.Request(
                "POST","https://api.deepseek.com"
            )
        )

        # TODO 2：使用 client.post() 请求 /rag/ask
        # 提交一个非空 question
        response = client.post(
            "/rag/ask",
            json={"question":"测试问题"}
        )

        # TODO 3：断言 HTTP 状态码为 502
        assert response.status_code == 502

        # TODO 4：断言返回的 detail 为
        # "AI 服务暂时不可用，请稍后再试"
        assert response.json()["detail"] == "AI 服务暂时不可用，请稍后再试"


def test_rag_ask_success():
    with (
        patch("api.get_retriever") as mock_retriever,
        patch("api.retrieve_context_with_sources") as mock_retrieve,
        patch("api.generate_answer") as mock_answer
    ):
        mock_retriever.return_value = ([], None, [])

        mock_retrieve.return_value = (
            "PDF 第6页：葡萄糖检测限为0.4631 μM",
            [
                {
                    "page":6,
                    "chunk_id":66,
                    "score":0.8,
                    "preview":"葡萄糖检测限为0.4631 μM"
                }
            ]
        )

        mock_answer.return_value = (
            "根据论文，葡萄糖检测限为0.4631 μM。"
        )

        # TODO 1：使用 client.post()
        # 向 /rag/ask 发送：
        # {"question": "葡萄糖的检测限是多少？"}
        response = client.post(
            "rag/ask",
            json={"question":"葡萄糖的检测限是多少？"}
        )

        # TODO 2：断言状态码为 200
        assert response.status_code == 200
        data = response.json()
        # TODO 3：断言返回的 question 与请求一致
        assert data["question"] == "葡萄糖的检测限是多少？"

        # TODO 4：断言返回的 answer
        # 等于 mock_answer.return_value
        assert data["answer"] == "根据论文，葡萄糖检测限为0.4631 μM。"

        sources = data["sources"]
        assert isinstance(sources,list)
        assert len(sources) == 1
        assert sources[0]["page"] == 6
        assert sources[0]["chunk_id"] == 66

def test_rag_ask_invalid_question():
    response = client.post(
        "/rag/ask",
        json={"question": ""}
    )
    # TODO 2：断言状态码为 422
    assert response.status_code == 422
    data = response.json()
    # TODO 3：断言返回 JSON 包含 detail 字段
    assert "detail" in data
