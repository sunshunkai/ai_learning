# -*- coding: utf-8 -*-
"""
Agent 层结构化输出。

`reply(structured_schema=...)` 会强制智能体最终返回符合 Pydantic schema
的 dict，结果存放在 final_msg.structured_output。

运行:
    python agentscope-demo/c03_02_agent_structured_output.py
"""

import asyncio
import json

from pydantic import BaseModel, Field

from agentscope.agent import Agent
from agentscope.tool import Toolkit

from common import get_model, make_user


class StudyPlan(BaseModel):
    """Python 转 AI 开发学习计划。"""

    current_level: str = Field(description="对提问者当前水平的判断")
    must_learn: list[str] = Field(description="必须学习的 3-5 个主题")
    optional: list[str] = Field(description="可选学习的 1-3 个主题")
    next_action: str = Field(description="今天就可以执行的下一步")


async def main() -> None:
    agent = Agent(
        name="Planner",
        system_prompt="你是严谨的 AI 学习规划师，输出必须覆盖用户指定的结构。",
        model=get_model(stream=False, thinking_enable=True, max_tokens=2000),
        toolkit=Toolkit(),
    )

    reply = await agent.reply(
        make_user(
            "我有 3 年 Java 经验，每天能学 1.5 小时，"
            "目标是一年内转到 AI 应用开发岗位。请制定学习计划。"
        ),
        structured_schema=StudyPlan,
    )

    print("=== 校验后的结构化结果 ===")
    print(json.dumps(reply.structured_output, ensure_ascii=False, indent=2))

    plan = StudyPlan.model_validate(reply.structured_output)
    print("\n=== 业务代码直接使用 ===")
    print(f"当前水平: {plan.current_level}")
    print(f"下一步: {plan.next_action}")
    print(f"必须学习: {len(plan.must_learn)} 项")
    print(f"可选学习: {len(plan.optional)} 项")


if __name__ == "__main__":
    asyncio.run(main())
