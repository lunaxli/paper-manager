
from types import SimpleNamespace
from unittest.mock import Mock

from agent_runner import (
    request_tool_decision,
    PAPER_SEARCH_TOOL
)


def make_fake_client():
    client = Mock()

    fake_message = SimpleNamespace(
        content="模拟模型回答",
        tool_calls=None
    )

    client.chat.completions.create.return_value = (
        SimpleNamespace(
            choices=[
                SimpleNamespace(message=fake_message)
            ]
        )
    )

    return client, fake_message


def test_request_tool_decision_default_prompt():

    client, fake_message = make_fake_client()

    result = request_tool_decision(
        "葡萄糖检测限是多少？",
        client
    )

    kwargs = client.chat.completions.create.call_args.kwargs
    messages = kwargs["messages"]

    expected_prompt = (
        "你是一名科研论文分析助手。"
        "涉及论文内容、实验数据和研究结论时，"
        "应使用 search_paper 工具获取证据，"
        "不要编造论文中的信息。"
    )

    # 不传第三个参数，仍然使用 V1 原始 Prompt
    assert messages[0] == {
        "role": "system",
        "content": expected_prompt
    }

    assert messages[1] == {
        "role": "user",
        "content": "葡萄糖检测限是多少？"
    }

    assert kwargs["model"] == "deepseek-flash"
    assert kwargs["tools"] == [PAPER_SEARCH_TOOL]
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["max_tokens"] == 700

    assert result is fake_message
    client.chat.completions.create.assert_called_once()


def test_request_tool_decision_custom_prompt():

    client, fake_message = make_fake_client()

    custom_prompt = (
        "你是一名英文科研论文检索助手。"
        "调用 search_paper 时优先使用英文检索关键词。"
    )

    result = request_tool_decision(
        "葡萄糖检测限是多少？",
        client,
        system_prompt=custom_prompt
    )

    kwargs = client.chat.completions.create.call_args.kwargs
    messages = kwargs["messages"]

    # 自定义 Prompt 应当替换默认 Prompt
    assert messages[0]["content"] == custom_prompt

    # 用户问题不应该被改变
    assert messages[1]["content"] == "葡萄糖检测限是多少？"

    # 工具调用配置保持不变
    assert kwargs["tools"] == [PAPER_SEARCH_TOOL]
    assert kwargs["tool_choice"] == "auto"

    assert result is fake_message
    client.chat.completions.create.assert_called_once()
