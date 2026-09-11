# -*- coding: utf-8 -*-
"""
Agent Human-in-the-loop：工具执行前暂停，等待人工批准

运行: python langchain-demo/14_agent_hitl.py

自动化测试:
  CLAUDE_DEMO_APPROVAL=yes python langchain-demo/14_agent_hitl.py
  CLAUDE_DEMO_APPROVAL=no  python langchain-demo/14_agent_hitl.py
"""

import json
import os
import sys

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from common import build_model


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件；真实业务中应接入邮件服务。"""
    return f"邮件已发送至 {to}，主题：{subject}"


def approve() -> bool:
    override = os.environ.get("CLAUDE_DEMO_APPROVAL")
    if override is not None:
        return override.strip().lower() in {"y", "yes", "approve", "approved"}
    if not sys.stdin.isatty():
        return False
    answer = input("是否继续执行工具？输入 y 批准: ").strip().lower()
    return answer in {"y", "yes"}


def main():
    agent = create_agent(
        build_model(temperature=0),
        tools=[send_email],
        system_prompt="需要发送邮件时调用 send_email 工具。",
        checkpointer=InMemorySaver(),
        # 在 tools 节点之前暂停，给人工审批留下机会
        interrupt_before=["tools"],
    )
    config = {"configurable": {"thread_id": "email-approval-1"}}

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "给 alice@example.com 发送邮件，"
                        "主题是项目进度，正文说明第一阶段已完成。"
                    ),
                }
            ]
        },
        config=config,
    )

    last = result["messages"][-1]
    if not getattr(last, "tool_calls", None):
        print("Agent 没有请求工具，直接回答：")
        print(last.content)
        return

    print("=== Agent 已暂停，待执行工具 ===")
    print(json.dumps(last.tool_calls, ensure_ascii=False, indent=2))

    if not approve():
        print("人工拒绝，工具未执行，Agent 保持暂停状态。")
        return

    print("已批准，继续执行工具。")
    final_state = agent.invoke(None, config=config)

    print("\n=== 执行轨迹 ===")
    for message in final_state["messages"]:
        print(f"[{message.type}] {message.content}")


if __name__ == "__main__":
    main()
