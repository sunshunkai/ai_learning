# -*- coding: utf-8 -*-
"""
中断运行中的 Agent。

Agent 的中断机制基于 asyncio 取消：cancel 正在运行的任务后，Agent 会保留
已经生成的部分内容，并给出 INTERRUPTED 的 ReplyEndEvent，上下文保持一致。

运行:
    python agentscope-demo/c03_06_interrupt_agent.py
"""

import asyncio

from agentscope.agent import Agent
from agentscope.event import ReplyEndEvent, TextBlockDeltaEvent
from agentscope.tool import Toolkit

from common import get_model, make_user


async def main() -> None:
    agent = Agent(
        name="LongWriter",
        system_prompt=(
            "请生成一段很长的、分点展开的技术说明，"
            "让输出过程持续足够长，便于演示中断。"
        ),
        model=get_model(stream=True, thinking_enable=False, max_tokens=4000),
        toolkit=Toolkit(),
        react_config=None,
    )

    text_parts: list[str] = []
    end_reason = None

    async def run_reply() -> None:
        nonlocal end_reason
        async for event in agent.reply_stream(
            make_user("请详细说明一个生产级 AI Agent 系统由哪些部分组成。")
        ):
            if isinstance(event, TextBlockDeltaEvent):
                text_parts.append(event.delta)
                print(event.delta, end="", flush=True)
            elif isinstance(event, ReplyEndEvent):
                end_reason = event.finished_reason

    task = asyncio.create_task(run_reply())
    await asyncio.sleep(1.0)
    print("\n\n[main] 发送取消信号...")
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        print("[main] 外层任务收到 CancelledError")

    print("\n=== 中断结果 ===")
    print(f"已收到文本块字符数: {len(''.join(text_parts))}")
    print(f"ReplyEndEvent     : {end_reason}")
    print(f"Agent context 数   : {len(agent.state.context)}")
    print("现在仍可继续发送新消息开始下一轮回复。")


if __name__ == "__main__":
    asyncio.run(main())
