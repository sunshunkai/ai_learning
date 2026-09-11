# -*- coding: utf-8 -*-
"""
Claude SDK —— 人工审批后再执行工具（Human-in-the-loop）

适合发送邮件、删除文件、退款、部署等高风险操作：
  模型提出工具调用
  -> 程序展示工具名和参数
  -> 人工批准或拒绝
  -> 批准才真正执行
  -> 把批准/拒绝结果回传模型

运行:
  python claude-sdk/human_approval.py

自动化测试时可通过环境变量跳过交互:
  CLAUDE_DEMO_APPROVAL=yes python claude-sdk/human_approval.py
  CLAUDE_DEMO_APPROVAL=no  python claude-sdk/human_approval.py
"""

import json
import os
import sys

from common import DEFAULT_MODEL, get_client, text_from_message


TOOLS = [
    {
        "name": "send_email",
        "description": "发送一封电子邮件。这是一个高风险操作，必须经过人工审批。",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "收件人邮箱"},
                "subject": {"type": "string", "description": "邮件主题"},
                "body": {"type": "string", "description": "邮件正文"},
            },
            "required": ["to", "subject", "body"],
        },
    }
]


def ask_approval(tool_name: str, tool_input: dict) -> tuple[bool, str]:
    """返回是否批准以及拒绝原因。"""
    print("\n[需要人工审批]")
    print(f"工具: {tool_name}")
    print("参数:")
    print(json.dumps(tool_input, ensure_ascii=False, indent=2))

    override = os.environ.get("CLAUDE_DEMO_APPROVAL")
    if override is not None:
        approved = override.strip().lower() in {"y", "yes", "approve", "approved"}
        reason = "环境变量自动批准" if approved else "环境变量自动拒绝"
        print(f"[自动决策] {reason}")
        return approved, reason

    if not sys.stdin.isatty():
        return False, "当前不是交互式终端，默认拒绝高风险操作"

    answer = input("是否批准执行？输入 y 批准，其他输入拒绝: ").strip().lower()
    return answer in {"y", "yes"}, "用户输入的决定"


def execute_email(tool_input: dict) -> str:
    """演示用，不真的发送邮件。"""
    return (
        f"邮件已发送至 {tool_input['to']}，"
        f"主题：{tool_input['subject']}"
    )


def main():
    client = get_client()
    messages = [
        {
            "role": "user",
            "content": (
                "请给 alice@example.com 发一封邮件，"
                "主题是“项目进度”，正文说明项目已经完成第一阶段。"
            ),
        }
    ]

    for _ in range(5):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            approved, reason = ask_approval(block.name, block.input)
            if approved and block.name == "send_email":
                result = execute_email(block.input)
                is_error = False
            else:
                result = f"操作未执行：{reason}"
                is_error = True

            print(f"[执行结果] {result}")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                    "is_error": is_error,
                }
            )

        if not tool_results:
            print("\n[最终回答]")
            print(text_from_message(response))
            return

        messages.append({"role": "user", "content": tool_results})

    print("达到最大循环次数，已停止。")


if __name__ == "__main__":
    main()
