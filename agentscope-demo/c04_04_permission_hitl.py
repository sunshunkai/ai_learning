# -*- coding: utf-8 -*-
"""
权限系统与 Human-in-the-loop。

默认权限模式下，FunctionTool 未显式放行会返回 ASK。Agent 会暂停并发出
RequireUserConfirmEvent，由调用方决定 allow / deny，再用
UserConfirmResultEvent 恢复。

自动测试:
    AGENTSCOPE_APPROVAL=yes python agentscope-demo/c04_04_permission_hitl.py
    AGENTSCOPE_APPROVAL=no  python agentscope-demo/c04_04_permission_hitl.py

运行（交互式，默认询问）:
    python agentscope-demo/c04_04_permission_hitl.py
"""

import asyncio
import os
import sys

from agentscope.agent import Agent
from agentscope.event import (
    ConfirmResult,
    RequireUserConfirmEvent,
    UserConfirmResultEvent,
)
from agentscope.permission import PermissionBehavior, PermissionDecision
from agentscope.tool import FunctionTool, Toolkit

from common import get_model, make_user


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email.

    Args:
        to: Recipient email address.
        subject: Email subject.
        body: Email body.
    """
    return f"已向 {to} 发送邮件，主题：{subject}"


def decide(tool_input: dict) -> bool:
    override = os.environ.get("AGENTSCOPE_APPROVAL")
    if override is not None:
        return override.strip().lower() in {"yes", "y", "1", "true"}
    if not sys.stdin.isatty():
        return False
    print("\n[需要人工确认]")
    print(f"  工具: send_email")
    print(f"  参数: {tool_input}")
    answer = input("  是否执行？输入 y 批准，其他输入拒绝: ").strip().lower()
    return answer in {"y", "yes"}


async def main() -> None:
    send_email_tool = FunctionTool(
        send_email,
        permission=PermissionDecision(
            behavior=PermissionBehavior.ASK,
            message="发送邮件属于高风险操作，必须由用户确认。",
        ),
    )
    agent = Agent(
        name="MailAssistant",
        system_prompt="你是邮件助手，发送邮件前必须通过工具。",
        model=get_model(stream=False),
        toolkit=Toolkit(tools=[send_email_tool]),
    )

    user_msg = make_user(
        "请给 nathan@example.com 发一封主题为“项目进展”的邮件，"
        "正文说明第一阶段已完成。"
    )

    pause_event: RequireUserConfirmEvent | None = None
    print("=== 第一段事件流：运行到权限暂停 ===")
    async for event in agent.reply_stream(user_msg):
        print(f"  event={event.type}")
        if isinstance(event, RequireUserConfirmEvent):
            pause_event = event
            break

    if pause_event is None:
        print("模型没有请求发邮件。")
        return

    confirm_results: list[ConfirmResult] = []
    for tool_call in pause_event.tool_calls:
        approved = decide({tool_call.name: tool_call.input})
        print(f"[决策] {'ALLOW' if approved else 'DENY'}")
        confirm_results.append(
            ConfirmResult(
                confirmed=approved,
                tool_call=tool_call,
                rules=tool_call.suggested_rules if approved else None,
            )
        )

    resume_event = UserConfirmResultEvent(
        reply_id=pause_event.reply_id,
        confirm_results=confirm_results,
    )
    reply = await agent.reply(resume_event)
    print("\n=== 恢复后的最终回答 ===")
    print(reply.get_text_content())


if __name__ == "__main__":
    asyncio.run(main())
