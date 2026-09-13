# -*- coding: utf-8 -*-
"""
Console：交互式调试与嵌入式事件渲染。

默认进入 launch_console 交互循环；设置 AGENTSCOPE_MESSAGE 时自动发送
一次消息，并用 ConsoleRenderer 渲染事件，方便 CI / 快速回归。

运行:
    python agentscope-demo/c07_06_console_chat.py
    AGENTSCOPE_MESSAGE="你好" python agentscope-demo/c07_06_console_chat.py
"""

import asyncio
import os

from agentscope.agent import Agent
from agentscope.console import ConsoleRenderer, launch_console
from agentscope.message import UserMsg
from agentscope.tool import Toolkit

from common import get_model


async def render_one_message(agent: Agent) -> None:
    message = os.environ["AGENTSCOPE_MESSAGE"]
    renderer = ConsoleRenderer(verbosity="default")
    async for event in agent.reply_stream(
        UserMsg(name="user", content=message)
    ):
        renderer.render(event)
    print("\n=== Renderer 重建的最终消息 ===")
    print(renderer.last_msg.get_text_content())


async def main() -> None:
    agent = Agent(
        name="Friday",
        system_prompt="你是友好的中文助手 Friday。",
        model=get_model(stream=True, thinking_enable=True),
        toolkit=Toolkit(),
    )

    if os.environ.get("AGENTSCOPE_MESSAGE"):
        await render_one_message(agent)
    else:
        await launch_console(agent, verbosity="default")


if __name__ == "__main__":
    asyncio.run(main())
