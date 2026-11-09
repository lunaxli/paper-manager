from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent_runner import request_tool_decision


def test_request_tool_decision_uses_config():

    client = Mock()

    fake_message = SimpleNamespace(
        content="模拟回答",
        tool_calls=None
    )

    client.chat.completions.create.return_value = (
        SimpleNamespace(
            choices=[
                SimpleNamespace(message=fake_message)
            ]
        )
    )

    # 临时覆盖统一配置
    with patch(
        "agent_runner.config.DEEPSEEK_MODEL",
        "test-model"
    ), patch(
        "agent_runner.config.DEEPSEEK_MAX_TOKENS",
        1200
    ):

        result = request_tool_decision(
            "葡萄糖检测限是多少？",
            client
        )

    # 验证请求只执行一次
    client.chat.completions.create.assert_called_once()

    kwargs = client.chat.completions.create.call_args.kwargs

    # 验证使用自定义配置
    assert kwargs["model"] == "test-model"
    assert kwargs["max_tokens"] == 1200

    # 验证原来的 Tool Calling 配置不变
    assert kwargs["tool_choice"] == "auto"
    assert len(kwargs["tools"]) == 1

    assert kwargs["extra_body"] == {
        "thinking": {"type": "disabled"}
    }

    # 验证用户问题未改变
    assert kwargs["messages"][1]["content"] == (
        "葡萄糖检测限是多少？"
    )

    # 验证返回原始模型消息
    assert result is fake_message
