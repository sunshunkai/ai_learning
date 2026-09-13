# -*- coding: utf-8 -*-
"""
内置文件与 Shell 工具。

AgentScope 内置 Bash、Read、Write、Edit、Glob、Grep、Task 系列工具。
这里让 Agent 在临时目录中创建、读取、编辑一个 Markdown 文件。

运行:
    python agentscope-demo/c04_03_builtin_tools.py
"""

import asyncio
import shutil
import tempfile
from pathlib import Path

from agentscope.agent import Agent
from agentscope.permission import PermissionMode
from agentscope.state import AgentState
from agentscope.tool import Bash, Edit, Read, Toolkit, Write

from common import get_model, make_user


async def main() -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="agentscope-builtin-"))
    target = temp_dir / "notes.md"
    print(f"工作目录: {temp_dir}")

    state = AgentState()
    state.permission_context.mode = PermissionMode.BYPASS

    agent = Agent(
        name="FileWorker",
        system_prompt=(
            "你是文件操作助手。所有文件操作都在用户给定工作目录内进行，"
            "最后回复任务完成情况。"
        ),
        model=get_model(stream=False),
        toolkit=Toolkit(
            tools=[
                Bash(cwd=str(temp_dir)),
                Read(),
                Write(),
                Edit(),
            ]
        ),
        state=state,
    )

    try:
        reply = await agent.reply(
            make_user(
                f"在 {target} 创建文件，内容为：\n"
                "# 学习笔记\n\nPython 很有趣。\n\n"
                "然后把 'Python 很有趣' 改成 'Python 和 Agent 都很有趣'，"
                "最后读取文件并总结内容。"
            )
        )
        print("=== 最终回答 ===")
        print(reply.get_text_content())
        print(f"\n实际文件内容:\n{target.read_text(encoding='utf-8')}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
