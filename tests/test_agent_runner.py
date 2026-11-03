import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent_runner import run_agent


def test_run_agent_with_tools():

    # 1. 模拟第一次 DeepSeek 返回工具调用请求
    fake_tool_call = SimpleNamespace(
        id="call_123",
        function=SimpleNamespace(
            name="search_paper",
            arguments='{"query": "葡萄糖检测限"}'
        )
    )

    fake_message = SimpleNamespace(
        content=None,
        tool_calls=[fake_tool_call]
    )

    # 2. 模拟工具执行结果
    fake_tool_messages = [
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": json.dumps(
                {
                    "context": "葡萄糖检测限为 0.4631 μM",
                    "sources": [
                        {
                            "page": 6,
                            "chunk_id": 66,
                            "score": 0.80,
                            "preview": "葡萄糖检测限为 0.4631 μM"
                        }
                    ]
                },
                ensure_ascii=False
            )
        }
    ]

    # 3. 模拟第二次 DeepSeek 回答
    client = Mock()

    client.chat.completions.create.return_value = (
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="论文中葡萄糖检测限为 0.4631 μM"
                    )
                )
            ]
        )
    )

    # 4. 执行 Agent 主流程
    with patch(
        "agent_runner.request_tool_decision",
        return_value=fake_message
    ), patch(
        "agent_runner.execute_tool_calls",
        return_value=fake_tool_messages
    ) as mock_execute:

        result = run_agent(
            "葡萄糖检测限是多少？",
            client,
            None,
            None,
            None
        )

    # 5. 检查最终回答
    assert result["answer"] == (
        "论文中葡萄糖检测限为 0.4631 μM"
    )

    # 6. 检查 sources 已成功整合
    assert len(result["sources"]) == 1
    assert result["sources"][0]["page"] == 6
    assert result["sources"][0]["chunk_id"] == 66
    assert result["sources"][0]["score"] == 0.80

    # 7. 检查工具执行
    mock_execute.assert_called_once()

    # 8. 检查第二次 DeepSeek 请求
    client.chat.completions.create.assert_called_once()

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "deepseek-flash"

    # 9. 检查消息顺序
    messages = kwargs["messages"]

    assert len(messages) == 4
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "tool"
    ]

    # 10. 检查工具调用 ID
    assert messages[2]["tool_calls"][0]["id"] == "call_123"
    assert messages[3]["tool_call_id"] == "call_123"

    # 11. 检查工具证据已发送给 DeepSeek
    tool_data = json.loads(messages[3]["content"])

    assert tool_data["context"] == (
        "葡萄糖检测限为 0.4631 μM"
    )
    assert tool_data["sources"][0]["page"] == 6