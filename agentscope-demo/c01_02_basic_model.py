# -*- coding: utf-8 -*-
"""
AgentScope DeepSeek 模型基础调用。

演示：
  1. stream=False 的一次性调用
  2. stream=True 的增量流式输出
  3. DeepSeek thinking 模式与最终回答

运行:
    python agentscope-demo/c01_02_basic_model.py
"""

import asyncio

from agentscope.message import TextBlock, ThinkingBlock

from common import DEFAULT_MODEL, call_and_print, get_model, make_user


async def non_streaming_call() -> None:
    print("\n=== 1. 非流式调用（一次拿到完整响应）===")
    model = get_model(stream=False)
    response = await call_and_print(
        model,
        [make_user("用两句话说明 Python 装饰器的作用。")],
    )

    print("响应内容块:")
    for block in response.content:
        print(f"  - type={block.type}, data={getattr(block, 'text', '')[:80]}")
    print(f"  finished_reason={response.finished_reason}")


async def streaming_call() -> None:
    print("\n=== 2. 流式调用（增量 ChatResponse）===")
    model = get_model(stream=True)
    response = await call_and_print(
        model,
        [make_user("用 5 个词介绍杭州。")],
    )
    print(f"\n最终累计文本长度={len(response.content[0].text)}")


async def thinking_call() -> None:
    print("\n=== 3. DeepSeek Thinking 模式 ===")
    model = get_model(
        stream=True,
        thinking_enable=True,
        reasoning_effort="high",
        max_tokens=1200,
    )
    response = await call_and_print(
        model,
        [
            make_user(
                "小明 3 点从杭州出发，先坐 40 分钟地铁到火车站，"
                "再坐 1.5 小时高铁到上海。他到上海时是几点？"
            )
        ],
    )

    thinking = "".join(
        block.thinking
        for block in response.content
        if isinstance(block, ThinkingBlock)
    )
    text = "".join(
        block.text
        for block in response.content
        if isinstance(block, TextBlock)
    )
    print(f"\n[统计] thinking_chars={len(thinking)}, answer_chars={len(text)}")


async def main() -> None:
    print(f"模型: {DEFAULT_MODEL}")
    await non_streaming_call()
    await streaming_call()
    await thinking_call()


if __name__ == "__main__":
    asyncio.run(main())
