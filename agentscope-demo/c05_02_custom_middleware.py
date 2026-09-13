# -*- coding: utf-8 -*-
"""
Agent Middleware：观察和扩展 Agent 生命周期。

演示三个钩子：
  - on_system_prompt：动态修改系统提示词
  - on_reasoning：观察每轮推理
  - on_model_call：记录模型调用耗时

运行:
    python agentscope-demo/c05_02_custom_middleware.py
"""

import asyncio
import time
from collections.abc import AsyncGenerator

from agentscope.agent import Agent
from agentscope.event import AgentEvent
from agentscope.middleware import MiddlewareBase
from agentscope.permission import PermissionBehavior, PermissionDecision
from agentscope.tool import FunctionTool, Toolkit

from common import get_model, make_user


class AnswerStyleMiddleware(MiddlewareBase):
    async def on_system_prompt(self, agent, current_prompt: str) -> str:
        return (
            current_prompt
            + "\n\n[运行时规则] 始终使用中文，回答不超过三句话，"
            "最后以“结论：”开头。"
        )


class ReasoningObserver(MiddlewareBase):
    async def on_reasoning(
        self,
        agent,
        input_kwargs: dict,
        next_handler,
    ) -> AsyncGenerator[AgentEvent, None]:
        print(f"\n[on_reasoning] 第 {agent.state.cur_iter + 1} 轮开始")
        async for event in next_handler(**input_kwargs):
            yield event
        print(f"[on_reasoning] 第 {agent.state.cur_iter} 轮结束")


class ModelTimingMiddleware(MiddlewareBase):
    async def on_model_call(self, agent, input_kwargs: dict, next_handler):
        model = input_kwargs["current_model"]
        started = time.perf_counter()
        result = await next_handler(**input_kwargs)
        elapsed = time.perf_counter() - started
        print(f"[on_model_call] {model.model} took {elapsed:.2f}s")
        return result


def local_time() -> str:
    """Return current local time."""
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="seconds")


async def main() -> None:
    time_tool = FunctionTool(
        local_time,
        permission=PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            message="Read-only demo tool.",
        ),
    )
    agent = Agent(
        name="ObservedAgent",
        system_prompt="你是被中间件观察的中文助手。",
        model=get_model(stream=False),
        toolkit=Toolkit(tools=[time_tool]),
        middlewares=[
            AnswerStyleMiddleware(),
            ReasoningObserver(),
            ModelTimingMiddleware(),
        ],
    )

    reply = await agent.reply(
        make_user("现在几点了？用一句话告诉我。")
    )
    print("\n=== 最终回答 ===")
    print(reply.get_text_content())


if __name__ == "__main__":
    asyncio.run(main())
