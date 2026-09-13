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
# 脚本既支持从项目根目录运行，也支持直接运行单文件。
# 提前加入项目根目录，确保下面的 tools.load_env 始终可以导入。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.load_env import load as load_env  # noqa: E402

# 在创建模型前加载根目录 .env，os.environ 中随后就能读取 DeepSeek 配置。
load_env()


def build_model(model: str = "deepseek-chat", temperature: float = 0.7, **kwargs):
    """创建一个接入 DeepSeek 的 ChatOpenAI 实例。"""
    # 延迟导入 ChatOpenAI，避免仅导入 common.py 时就必须安装 OpenAI 依赖。
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    # DeepSeek 使用 OpenAI 兼容协议；这里只覆盖 base_url，其他调用方式不变。
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
