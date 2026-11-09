import runpy
from pathlib import Path

import pytest


CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config.py"
)


def test_config_default_values(monkeypatch):

    # 删除环境变量，模拟未配置的情况
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MAX_TOKENS", raising=False)

    # 重新执行配置文件
    config = runpy.run_path(str(CONFIG_PATH))

    assert config["DEEPSEEK_MODEL"] == "deepseek-flash"

    assert config["DEEPSEEK_BASE_URL"] == (
        "https://api.deepseek.com"
    )

    assert config["DEEPSEEK_MAX_TOKENS"] == 700
    assert isinstance(config["DEEPSEEK_MAX_TOKENS"], int)


def test_config_environment_override(monkeypatch):

    # 模拟用户自定义配置
    monkeypatch.setenv(
        "DEEPSEEK_MODEL",
        "test-model"
    )

    monkeypatch.setenv(
        "DEEPSEEK_BASE_URL",
        "https://example.com"
    )

    monkeypatch.setenv(
        "DEEPSEEK_MAX_TOKENS",
        "1200"
    )

    config = runpy.run_path(str(CONFIG_PATH))

    assert config["DEEPSEEK_MODEL"] == "test-model"

    assert config["DEEPSEEK_BASE_URL"] == (
        "https://example.com"
    )

    assert config["DEEPSEEK_MAX_TOKENS"] == 1200
    assert isinstance(config["DEEPSEEK_MAX_TOKENS"], int)


def test_config_invalid_max_tokens(monkeypatch):

    # 模拟错误配置
    monkeypatch.setenv(
        "DEEPSEEK_MAX_TOKENS",
        "not-a-number"
    )

    # 无法转换为整数时，应抛出 ValueError
    with pytest.raises(ValueError):
        runpy.run_path(str(CONFIG_PATH))


def test_log_level_default(monkeypatch):

    # 模拟未设置 LOG_LEVEL
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    config = runpy.run_path(str(CONFIG_PATH))

    assert config["LOG_LEVEL"] == "INFO"


def test_log_level_environment_override(monkeypatch):

    # 测试小写 DEBUG
    monkeypatch.setenv("LOG_LEVEL", "debug")

    config = runpy.run_path(str(CONFIG_PATH))

    assert config["LOG_LEVEL"] == "DEBUG"

    # 测试小写 WARNING
    monkeypatch.setenv("LOG_LEVEL", "warning")

    config = runpy.run_path(str(CONFIG_PATH))

    assert config["LOG_LEVEL"] == "WARNING"

    # 测试大写 ERROR
    monkeypatch.setenv("LOG_LEVEL", "ERROR")

    config = runpy.run_path(str(CONFIG_PATH))

    assert config["LOG_LEVEL"] == "ERROR"
