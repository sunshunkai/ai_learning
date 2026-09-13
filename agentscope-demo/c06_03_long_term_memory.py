# -*- coding: utf-8 -*-
"""
Agentic Memory：跨会话的 Markdown 文件记忆。

第一个 Agent 写入用户偏好；使用相同 workdir 创建第二个全新 Agent，
不共享 context，只共享 Memory 目录，验证它能否跨会话回忆。

运行:
    python agentscope-demo/c06_03_long_term_memory.py
"""

import asyncio
import shutil
import tempfile
from pathlib import Path

from agentscope.agent import Agent
from agentscope.middleware import AgenticMemoryMiddleware
from agentscope.permission import AdditionalWorkingDirectory, PermissionMode
from agentscope.state import AgentState
from agentscope.tool import Edit, Grep, Read, Toolkit, Write

from common import get_model, make_user


def build_agent(workdir: str, model) -> Agent:
    state = AgentState()
    state.permission_context.mode = PermissionMode.ACCEPT_EDITS
    state.permission_context.working_directories[workdir] = (
        AdditionalWorkingDirectory(
            path=workdir,
            source="long-term-memory-demo",
        )
    )
    memory = AgenticMemoryMiddleware(workdir=workdir)
    return Agent(
        name="MemoryAssistant",
        system_prompt=(
            "你是中文学习助手。用户要求记住的信息，使用 Write/Edit 写入 "
            "Memory 目录；回答新问题时先考虑已有记忆。"
        ),
        model=model,
        toolkit=Toolkit(
            tools=[
                Read(),
                Write(),
                Edit(),
                Grep(),
            ]
        ),
        middlewares=[memory],
        state=state,
    ), memory


async def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="agentscope-memory-"))
    try:
        print(f"Memory workdir: {workdir}")
        first_agent, _ = build_agent(str(workdir), get_model(stream=False))
        first_reply = await first_agent.reply(
            make_user(
                "请记住：我住在杭州，是 3 年 Java 开发，希望以后答案用"
                "简洁中文，先给结论再给解释。"
            )
        )
        print("\n=== 会话 1 ===")
        print(first_reply.get_text_content())

        memory_files = list((workdir / "Memory").glob("*.md"))
        print("\n=== Memory 文件 ===")
        for file in memory_files:
            print(f"  - {file.name}")

        second_agent, _ = build_agent(
            str(workdir),
            get_model(stream=False),
        )
        second_reply = await second_agent.reply(
            make_user(
                "请回忆我的位置、开发背景，以及我希望你用什么样的回答风格。"
            )
        )
        print("\n=== 全新会话 2（无共享 context）===")
        print(second_reply.get_text_content())
        print(f"\n会话 2 当前 context 消息数: {len(second_agent.state.context)}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
