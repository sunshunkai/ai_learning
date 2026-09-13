# -*- coding: utf-8 -*-
"""
GoalPipeline：执行者 + 验证者闭环。

执行者完成任务并输出结构化 report；验证者返回 pass / fail / impossible。
fail 时反馈会回给执行者继续修正，直到通过或耗尽 max_iters。

运行:
    python agentscope-demo/c05_03_goal_pipeline.py
"""

import asyncio

from agentscope.agent import Agent
from agentscope.message import Msg
from agentscope.pipeline import GoalPipeline
from agentscope.tool import Toolkit

from common import get_model, make_user


async def main() -> None:
    executor = Agent(
        name="Executor",
        system_prompt=(
            "你是 Python 讲师。针对目标给出简洁、有示例的交付报告。"
            "注意：第一次交付时故意不要包含代码示例。"
        ),
        model=get_model(stream=False, max_tokens=1600),
        toolkit=Toolkit(),
    )
    verifier = Agent(
        name="Verifier",
        system_prompt=(
            "你是验收者。如果执行者报告缺少一个最小 Python 代码示例，"
            "返回 fail 并明确要求补上；否则返回 pass。"
        ),
        model=get_model(stream=False, max_tokens=1200),
        toolkit=Toolkit(),
    )

    pipeline = GoalPipeline(
        executor=executor,
        verifier=verifier,
        verifier_reset_context=True,
        max_iters=3,
    )

    final_messages: list[Msg] = []
    print("=== Pipeline 事件与结构化输出 ===")
    async for event_or_msg in pipeline.reply_stream(
        make_user("写一份关于 Python 装饰器的学习说明。")
    ):
        if isinstance(event_or_msg, Msg):
            final_messages.append(event_or_msg)
            print(
                f"[{event_or_msg.name}] "
                f"finished={event_or_msg.finished_reason}, "
                f"structured={event_or_msg.structured_output}"
            )
        else:
            if event_or_msg.type in {
                "REPLY_START",
                "REPLY_END",
                "MODEL_CALL_START",
                "MODEL_CALL_END",
            }:
                print(f"  event={event_or_msg.type}")

    print("\n=== 最终执行报告 ===")
    executor_messages = [
        msg
        for msg in final_messages
        if msg.name == "Executor" and msg.structured_output
    ]
    if executor_messages:
        report = executor_messages[-1].structured_output or {}
        print(report.get("report", "<无 report>"))
    print("\n=== Verifier 最终判决 ===")
    verdict = pipeline.verifier.state.reply_context.structured_output
    print(verdict or "<未取得验证结果>")


if __name__ == "__main__":
    asyncio.run(main())
