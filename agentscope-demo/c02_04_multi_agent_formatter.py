# -*- coding: utf-8 -*-
"""
DeepSeek 多智能体 Formatter。

多智能体历史中存在多个 assistant 发言者。DeepSeekMultiAgentFormatter 会
把历史打包到 <history> 标签中，并保留每个发言者的 name。

运行:
    python agentscope-demo/c02_04_multi_agent_formatter.py
"""

import asyncio

from agentscope.formatter import DeepSeekMultiAgentFormatter
from agentscope.message import Msg, TextBlock

from common import call_and_print, get_model


async def main() -> None:
    print("=== 多智能体历史 -> 主持人总结 ===")
    model = get_model(
        stream=True,
        thinking_enable=False,
        formatter=DeepSeekMultiAgentFormatter(),
    )

    messages = [
        Msg(
            name="system",
            role="system",
            content=[TextBlock(text="你是会议主持人，最后用三句话总结讨论。")],
        ),
        Msg(
            name="alice",
            role="user",
            content=[TextBlock(text="我认为新模块应该先补测试，再写功能。")],
        ),
        Msg(
            name="bob",
            role="assistant",
            content=[
                TextBlock(text="我同意测试优先，但核心 API 可以先固定下来。")
            ],
        ),
        Msg(
            name="carol",
            role="assistant",
            content=[
                TextBlock(text="API 固定和测试并不冲突，建议先写契约测试。")
            ],
        ),
        Msg(
            name="moderator",
            role="user",
            content=[TextBlock(text="请作为主持人总结上面的讨论。")],
        ),
    ]

    await call_and_print(model, messages)


if __name__ == "__main__":
    asyncio.run(main())
