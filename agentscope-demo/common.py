# -*- coding: utf-8 -*-
"""
AgentScope 学习模块公共工具。

统一负责：
  1. 加载项目根目录 .env
  2. 创建 DeepSeekChatModel
  3. 打印 / 收集 ChatResponse 中的文本与 Thinking 内容
  4. 显示 token 用量
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.load_env import load, require  # noqa: E402

load()

DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEFAULT_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEFAULT_CONTEXT_SIZE = 1_000_000


def get_model(
    *,
    model: str | None = None,
    stream: bool = True,
    thinking_enable: bool = False,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    max_retries: int = 3,
    **kwargs,
):
    """创建统一的 DeepSeek Chat Model。

    Args:
        model: DeepSeek 模型名，默认读 DEEPSEEK_MODEL，再退回 deepseek-v4-flash。
        stream: 是否使用流式 API。
        thinking_enable: 是否开启 DeepSeek reasoning_content。
        reasoning_effort: "high" 或 "max"；模型层会兼容处理其他值。
        temperature / top_p / max_tokens: DeepSeek 原生参数。
        context_size: AgentScope 用于上下文压缩的窗口大小。
        max_retries: API 重试次数。
        **kwargs: 透传给 DeepSeekChatModel 的其他参数。
    """
    from agentscope.credential import DeepSeekCredential
    from agentscope.model import DeepSeekChatModel

    api_key = require("DEEPSEEK_API_KEY")
    parameters = DeepSeekChatModel.Parameters(
        thinking_enable=thinking_enable,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    return DeepSeekChatModel(
        credential=DeepSeekCredential(
            api_key=api_key,
            base_url=DEFAULT_BASE_URL,
        ),
        model=model or DEFAULT_MODEL,
        parameters=parameters,
        stream=stream,
        context_size=context_size,
        max_retries=max_retries,
        **kwargs,
    )


def make_user(content, name: str = "user"):
    """把字符串包装成 UserMsg。"""
    from agentscope.message import UserMsg

    return UserMsg(name=name, content=content)


def text_from_response(response) -> str:
    """提取 ChatResponse 或 Msg 中的文本。"""
    from agentscope.message import TextBlock

    if response is None:
        return ""
    return "".join(
        block.text
        for block in response.content
        if isinstance(block, TextBlock)
    )


def thinking_from_response(response) -> str:
    """提取 ChatResponse 或 Msg 中的思考内容。"""
    from agentscope.message import ThinkingBlock

    if response is None:
        return ""
    return "".join(
        block.thinking
        for block in response.content
        if isinstance(block, ThinkingBlock)
    )


def print_usage(response, prefix: str = "[usage]") -> None:
    """打印 ChatResponse 的 token 统计。"""
    if response is None or response.usage is None:
        print(f"{prefix} 无 usage")
        return
    usage = response.usage
    print(
        f"{prefix} in={usage.input_tokens}, out={usage.output_tokens}, "
        f"cache={getattr(usage, 'cache_input_tokens', 0)}, "
        f"time={usage.time:.2f}s"
    )


async def stream_and_collect(model, messages, **kwargs):
    """调用模型，流式时打印增量并返回最后一个累计响应。

    返回：
        stream=False 时是 ChatResponse；
        stream=True 时是 is_last=True 的最终累计 ChatResponse。
    """
    from agentscope.message import ThinkingBlock, TextBlock

    raw = await model(messages, **kwargs)
    if not model.stream:
        return raw

    final = None
    in_thinking = False
    text_seen = False
    async for chunk in raw:
        if chunk.is_last:
            final = chunk
            continue

        for block in chunk.content:
            if isinstance(block, ThinkingBlock):
                if not in_thinking:
                    print("\n[Thinking] ", end="", flush=True)
                    in_thinking = True
                print(block.thinking, end="", flush=True)
            elif isinstance(block, TextBlock):
                if in_thinking:
                    print("\n\n[Answer] ", end="", flush=True)
                    in_thinking = False
                print(block.text, end="", flush=True)
                text_seen = True

    if in_thinking:
        print("\n")
    elif not text_seen and final is not None:
        print(text_from_response(final))
    elif text_seen:
        print()

    if final is None:
        raise RuntimeError("流式模型调用没有返回最终累计响应")
    return final


async def call_and_print(model, messages, **kwargs):
    """调用模型并统一打印文本、思考和 usage。"""
    response = await stream_and_collect(model, messages, **kwargs)
    print_usage(response)
    return response


def mask_key(value: str | None) -> str:
    """只展示 API Key 的掩码。"""
    if not value:
        return "<未配置>"
    if len(value) <= 8:
        return "<太短>"
    return f"{value[:6]}...{value[-4:]}"
