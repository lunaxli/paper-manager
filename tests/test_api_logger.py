
import os
import sys
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "log_level, expected_level, info_enabled",
    [
        ("debug", 10, True),
        ("INFO", 20, True),
        ("warning", 30, False),
        ("ERROR", 40, False),
    ]
)
def test_api_logger_level(
    log_level,
    expected_level,
    info_enabled
):

    env = os.environ.copy()
    env["LOG_LEVEL"] = log_level

    code = """
import api
import logging

print(f"LEVEL={api.logger.level}")
print(f"INFO_ENABLED={api.logger.isEnabledFor(logging.INFO)}")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60
    )

    assert result.returncode == 0, result.stderr

    assert f"LEVEL={expected_level}" in result.stdout

    assert (
        f"INFO_ENABLED={info_enabled}"
        in result.stdout
    )


def test_api_logger_invalid_level():

    env = os.environ.copy()
    env["LOG_LEVEL"] = "INVALID"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import api"
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60
    )

    # 非法配置应导致模块初始化失败
    assert result.returncode != 0

    assert "ValueError" in result.stderr
    assert "无效的日志级别" in result.stderr
