# -*- coding: utf-8 -*-
"""
AgentState 持久化与恢复。

Agent 本身无状态，真正的会话状态在 AgentState。Pydantic 提供
model_dump_json / model_validate_json，因此可以保存到 Redis、数据库或文件。

运行:
    python agentscope-demo/c03_05_state_persistence.py
"""

import asyncio

from agentscope.agent import Agent
from agentscope.state import AgentState
from agentscope.tool import Toolkit

from common import get_model, make_user


async def main() -> None:
    session_id = "state-demo-session-001"
    model_kwargs = {"stream": False}

    first = Agent(
        name="Memory",
        system_prompt="你是有记忆的助手，会记住用户告诉你的事实。",
        model=get_model(**model_kwargs),
        toolkit=Toolkit(),
        state=AgentState(session_id=session_id),
    )
    reply = await first.reply(
        make_user("记住：我住在杭州，希望答案用简洁中文，尽量不超过三句话。")
    )
    print("原会话:", reply.get_text_content())

    serialized = first.state.model_dump_json()
    print(f"\n序列化状态长度: {len(serialized)} chars")
    print(f"序列化消息数  : {len(first.state.context)}")

    restored_state = AgentState.model_validate_json(serialized)
    second = Agent(
        name="Memory",
        system_prompt="你是有记忆的助手，会记住用户告诉你的事实。",
        model=get_model(**model_kwargs),
        toolkit=Toolkit(),
        state=restored_state,
    )

    reply = await second.reply(
        make_user("你还记得我住哪里、希望怎么回答吗？")
    )
    print("\n恢复后会话:", reply.get_text_content())
    print(f"\nsession_id 一致: {first.state.session_id == second.state.session_id}")
    print(f"context 消息数 : {len(second.state.context)}")


if __name__ == "__main__":
    asyncio.run(main())
