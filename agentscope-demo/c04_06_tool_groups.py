# -*- coding: utf-8 -*-
"""
元工具 reset_tools 与 ToolGroup。

basic 工具组始终激活；其他工具组可以由模型通过 reset_tools 动态
激活 / 停用，从而节省上下文和降低误调用概率。

运行:
    python agentscope-demo/c04_06_tool_groups.py
"""

import asyncio

from agentscope.agent import Agent
from agentscope.permission import PermissionBehavior, PermissionDecision
from agentscope.tool import FunctionTool, Toolkit, ToolGroup

from common import get_model, make_user


def format_python(code: str) -> str:
    """Format Python code in a simple, readable way.

    Args:
        code: Python source code.

    Returns:
        Normalized Python code.
    """
    lines = [line.rstrip() for line in code.strip().splitlines()]
    return "\n".join(lines)


def now() -> str:
    """Return the current local time."""
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="seconds")


async def main() -> None:
    allow = PermissionDecision(
        behavior=PermissionBehavior.ALLOW,
        message="Demo tool allowed.",
    )
    toolkit = Toolkit(
        tools=[FunctionTool(now, permission=allow)],
        tool_groups=[
            ToolGroup(
                name="code_utils",
                description="Tools for formatting and inspecting Python code.",
                instructions=(
                    "Activate this group when the user asks to format or "
                    "inspect Python code."
                ),
                tools=[FunctionTool(format_python, permission=allow)],
            ),
        ],
    )

    agent = Agent(
        name="ToolManager",
        system_prompt=(
            "你可以用 reset_tools 管理自己的工具组。"
            "当前需求需要先激活 code_utils，再调用 format_python。"
        ),
        model=get_model(stream=False, thinking_enable=False),
        toolkit=toolkit,
    )

    reply = await agent.reply(
        make_user(
            "请激活代码工具组，并格式化这段代码："
            "def add(a,b):    return a+b\n"
        )
    )
    print("=== 最终回答 ===")
    print(reply.get_text_content())
    print(
        "\n当前激活组: "
        + ", ".join(agent.state.tool_context.activated_groups)
    )


if __name__ == "__main__":
    asyncio.run(main())
