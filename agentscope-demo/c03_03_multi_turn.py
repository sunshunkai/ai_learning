# -*- coding: utf-8 -*-
"""
Agent 多轮上下文与 observe。

Agent 自身无状态，但 AgentState.context 保存了本会话的完整消息。
连续调用 reply 会自动把新消息与回复追加到同一 context。

运行:
    python agentscope-demo/c03_03_multi_turn.py
"""

import asyncio

from agentscope.agent import Agent
from agentscope.tool import Toolkit

from common import get_model, make_user


async def main() -> None:
    agent = Agent(
        name="Assistant",
        system_prompt="你是亲切的中文学习助手。",
        model=get_model(stream=False),
        toolkit=Toolkit(),
    )

    first = await agent.reply(
        make_user("我叫 Nathan，现在主要写 Java，正在学习 Python。")
    )
    print("第 1 轮:", first.get_text_content())

    second = await agent.reply(
        make_user("你还记得我叫什么、目前主要在写什么语言吗？")
    )
    print("\n第 2 轮:", second.get_text_content())

    print("\n=== observe：只记录消息，不触发推理 ===")
    await agent.observe(
        make_user("系统消息：明天下午我有一个面试，不需要现在回复。")
    )
    print(f"observe 后 context 消息数: {len(agent.state.context)}")

    third = await agent.reply(
        make_user("我之前说明天下午有什么安排？")
    )
    print("\n第 3 轮:", third.get_text_content())

    print("\n=== 最终 context 摘要 ===")
    for message in agent.state.context:
        text = message.get_text_content() or "<非文本内容>"
        print(f"[{message.role}:{message.name}] {text[:80]}")


if __name__ == "__main__":
    asyncio.run(main())
