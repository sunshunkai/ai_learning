# -*- coding: utf-8 -*-
"""
使用 A2AAgent 连接 c07_04_a2a_server.py 暴露的远端智能体。

先启动 server，然后运行:
    python agentscope-demo/c07_05_a2a_client.py [--url http://127.0.0.1:9999]

运行:
    python agentscope-demo/c07_05_a2a_client.py
"""

import argparse
import asyncio

import httpx
from a2a.client import A2ACardResolver

from agentscope.agent import A2AAgent
from agentscope.message import UserMsg


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:9999")
    args = parser.parse_args()

    async with httpx.AsyncClient() as httpx_client:
        card = await A2ACardResolver(
            httpx_client=httpx_client,
            base_url=args.url,
        ).get_agent_card()

    print(f"Connected to {card.name!r} at {args.url}.")
    async with A2AAgent(card) as agent:
        first = await agent.reply(
            UserMsg(name="user", content="请用一句话介绍 AgentScope。")
        )
        print("\n第 1 轮:", first.get_text_content())

        second = await agent.reply(
            UserMsg(name="user", content="请把它说得更短一些。")
        )
        print("\n第 2 轮:", second.get_text_content())

        print(
            "\n远端 context_id 保存在 A2AAgentState，同一实例内的多轮"
            "对话会自动复用。"
        )


if __name__ == "__main__":
    asyncio.run(main())
