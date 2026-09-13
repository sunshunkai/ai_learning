# -*- coding: utf-8 -*-
"""
DeepSeek Thinking 与输出参数控制。

DeepSeek v4-flash 的思考内容在 API 中是 reasoning_content，AgentScope
会把它转换为 ThinkingBlock。这里额外演示：
  - reasoning_effort
  - temperature / top_p
  - max_tokens

运行:
    python agentscope-demo/c02_01_thinking_chat.py
"""

import asyncio

from agentscope.message import TextBlock, ThinkingBlock

from common import call_and_print, get_model, make_user


def inspect_blocks(response) -> None:
    print("\n[内容块明细]")
    for block in response.content:
        if isinstance(block, ThinkingBlock):
            print(f"  thinking: {len(block.thinking)} chars")
        elif isinstance(block, TextBlock):
            print(f"  text    : {len(block.text)} chars")
        else:
            print(f"  {block.type}")


async def high_effort_example() -> None:
    model = get_model(
        stream=True,
        thinking_enable=True,
        reasoning_effort="high",
        temperature=0.2,
        top_p=0.9,
        max_tokens=3000,
    )
    response = await call_and_print(
        model,
        [
            make_user(
                "一件商品原价 200 元，先涨价 25%，再打八折。"
                "最终价格是多少元？"
            )
        ],
    )
    inspect_blocks(response)


async def concise_example() -> None:
    print("\n=== 控制温度与回复长度 ===")
    model = get_model(
        stream=True,
        thinking_enable=False,
        temperature=0.1,
        max_tokens=120,
    )
    response = await call_and_print(
        model,
        [make_user("用不超过 30 个汉字介绍 AgentScope 2.0。")],
    )


async def main() -> None:
    print("=== 1. Thinking + 低随机性 ===")
    await high_effort_example()
    await concise_example()


if __name__ == "__main__":
    asyncio.run(main())
