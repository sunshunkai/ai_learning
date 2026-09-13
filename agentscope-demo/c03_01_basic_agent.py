# -*- coding: utf-8 -*-
"""
AgentScope Agent 最简 ReAct 循环。

Agent 把 model、toolkit、state、permission、middleware、context 组合成
一个统一的推理-行动接口。本示例只保留 model，先看 reply 的完整运行。

运行:
    python agentscope-demo/c03_01_basic_agent.py
"""

import asyncio

from agentscope.agent import Agent
from agentscope.tool import Toolkit

from common import get_model, make_user


async def main() -> None:
    agent = Agent(
        name="Friday",
        system_prompt=(
            "你是中文技术助教 Friday。回答要准确、简洁，"
            "并在结尾给出一条可执行的学习建议。"
        ),
        model=get_model(stream=False),
        toolkit=Toolkit(),
    )

    user_msg = make_user(
        "我是 Java 开发，刚开始学 Python，解释一下 async/await 为什么重要。"
    )
    reply = await agent.reply(user_msg)

    print("=== Agent 最终消息 ===")
    print(reply.get_text_content())
    print("\n=== 运行后状态 ===")
    print(f"name           : {agent.name}")
    print(f"session_id     : {agent.state.session_id}")
    print(f"reply_id       : {agent.state.reply_id}")
    print(f"context 消息数  : {len(agent.state.context)}")
    print(f"finished_reason: {reply.finished_reason}")
    if reply.usage:
        print(
            f"usage          : in={reply.usage.input_tokens}, "
            f"out={reply.usage.output_tokens}"
        )


if __name__ == "__main__":
    asyncio.run(main())
