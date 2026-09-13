# -*- coding: utf-8 -*-
"""
Plan 模式与任务工具。

TaskCreate / TaskList / TaskGet / TaskUpdate 使用同一个
agent.state.tasks_context，帮助模型显式拆分、追踪和协调复杂工作。

运行:
    python agentscope-demo/c05_01_plan_mode.py
"""

import asyncio

from agentscope.agent import Agent
from agentscope.tool import TaskCreate, TaskGet, TaskList, TaskUpdate, Toolkit

from common import get_model, make_user


async def main() -> None:
    agent = Agent(
        name="Planner",
        system_prompt=(
            "你是严谨的执行规划器。对于多步骤任务，先创建结构化任务，"
            "执行时更新状态，最后汇总完成情况。"
        ),
        model=get_model(stream=False, thinking_enable=True),
        toolkit=Toolkit(
            tools=[
                TaskCreate(),
                TaskList(),
                TaskGet(),
                TaskUpdate(),
            ]
        ),
    )

    reply = await agent.reply(
        make_user(
            "请规划一次 AgentScope 学习任务："
            "1) 阅读 Agent 概念；2) 实现一个工具；3) 接入 DeepSeek 跑通；"
            "4) 写一份总结。创建任务清单，并把前两项标记为已完成。"
        )
    )

    print("=== 最终回答 ===")
    print(reply.get_text_content())

    print("\n=== agent.state.tasks_context ===")
    for task in agent.state.tasks_context.tasks:
        print(
            f"[{task.state:10}] id={task.id}, subject={task.subject}, "
            f"owner={task.owner or '-'}, blocked_by={task.blocked_by}"
        )


if __name__ == "__main__":
    asyncio.run(main())
