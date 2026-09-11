# -*- coding: utf-8 -*-
"""
Claude SDK 示例公共工具。

统一负责：
  1. 加载项目根目录 .env
  2. 创建同步 / 异步 Anthropic 客户端
  3. 提取响应中的文本
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.load_env import load, require  # noqa: E402

load()

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")


def _client_kwargs(kwargs: dict) -> dict:
    result = dict(kwargs)
    result.setdefault("api_key", require("ANTHROPIC_API_KEY"))
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if base_url:
        result.setdefault("base_url", base_url)
    return result


def get_client(**kwargs):
    """创建同步 Anthropic 客户端。"""
    from anthropic import Anthropic

    return Anthropic(**_client_kwargs(kwargs))


def get_async_client(**kwargs):
    """创建异步 AsyncAnthropic 客户端。"""
    from anthropic import AsyncAnthropic

    return AsyncAnthropic(**_client_kwargs(kwargs))


def text_from_message(message) -> str:
    """提取一条响应中的所有 text block。"""
    return "".join(
        block.text
        for block in message.content
        if getattr(block, "type", None) == "text"
    )


def print_text_blocks(prefix: str, message) -> None:
    """打印响应中的文本块。"""
    for block in message.content:
        if getattr(block, "type", None) == "text":
            print(f"{prefix}{block.text}")
