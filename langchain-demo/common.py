# -*- coding: utf-8 -*-
"""
LangChain demo 公共工具：加载 .env 并创建 DeepSeek 聊天模型。

DeepSeek 官方兼容 OpenAI Chat Completions 接口，
所以 LangChain 侧使用 langchain-openai 的 ChatOpenAI 接入。
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.load_env import load as load_env  # noqa: E402

load_env()


def build_model(model: str = "deepseek-chat", temperature: float = 0.7, **kwargs):
    """创建一个接入 DeepSeek 的 ChatOpenAI 实例。"""
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    if not api_key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY，请先在项目根目录 .env 中配置")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        **kwargs,
    )
