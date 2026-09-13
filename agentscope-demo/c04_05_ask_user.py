# -*- coding: utf-8 -*-
"""
内置 AskUser：外部执行工具与用户作答。

AskUser 没有内部执行逻辑，Agent 调用它时会发出
RequireExternalExecutionEvent 并暂停；调用方把答案包装成
ToolResultBlock，再通过 ExternalExecutionResultEvent 回传。

运行:
    python agentscope-demo/c04_05_ask_user.py
"""

import asyncio
import json

from agentscope.agent import Agent
from agentscope.event import ExternalExecutionResultEvent, RequireExternalExecutionEvent
from agentscope.message import TextBlock, ToolResultBlock, ToolResultState
from agentscope.tool import AskUser, AskUserAnswer, AskUserMetadata, Toolkit

from common import get_model, make_user


def answer_questions(raw_input: str) -> tuple[str, AskUserMetadata]:
    params = json.loads(raw_input)
    answers = []
    rendered_lines = []
    for question in params["questions"]:
        options = question.get("options", [])
        selected = [options[0]["label"]] if options else []
        answers.append(
            AskUserAnswer(
                question=question["question"],
                selected=selected,
                other=None,
            )
        )
        rendered_lines.append(
            f"{question['question']} -> {', '.join(selected)}"
        )
    return "\n".join(rendered_lines), AskUserMetadata(answers=answers)


async def main() -> None:
    agent = Agent(
        name="SetupAssistant",
        system_prompt=(
            "你负责为新项目做技术选型。当你需要用户决定数据库或部署方式时，"
            "调用 AskUser 工具，每次最多问 4 个问题，每题 2-4 个选项。"
        ),
        model=get_model(stream=False, thinking_enable=False),
        toolkit=Toolkit(tools=[AskUser()]),
    )

    pause_event: RequireExternalExecutionEvent | None = None
    print("=== Agent 运行到 AskUser 暂停 ===")
    async for event in agent.reply_stream(
        make_user("我要做一个小型博客，请问我两个技术选型问题。")
    ):
        print(f"  event={event.type}")
        if isinstance(event, RequireExternalExecutionEvent):
            pause_event = event
            break

    if pause_event is None:
        print("模型没有请求用户输入。")
        return

    tool_call = pause_event.tool_calls[0]
    print("\n=== 模拟用户作答 ===")
    output_text, metadata = answer_questions(tool_call.input)
    print(output_text)

    result = ToolResultBlock(
        id=tool_call.id,
        name=tool_call.name,
        output=[TextBlock(text=output_text)],
        metadata=metadata.model_dump(),
        state=ToolResultState.SUCCESS,
    )
    resume = ExternalExecutionResultEvent(
        reply_id=pause_event.reply_id,
        execution_results=[result],
    )

    reply = await agent.reply(resume)
    print("\n=== Agent 基于用户选择继续 ===")
    print(reply.get_text_content())


if __name__ == "__main__":
    asyncio.run(main())
