# -*- coding: utf-8 -*-
"""
LocalWorkspace 与 Skill。

Workspace 统一提供运行环境、内置工具、Skill 目录与 Offloader。
Skill 不是可调用工具，而是 Markdown 指令；Agent 先用 SkillViewer
读取完整指令，再用 Read/Write/Bash 等已有工具执行。

运行:
    python agentscope-demo/c07_01_workspace_skill.py
"""

import asyncio
import shutil
import tempfile
from pathlib import Path

from agentscope.agent import Agent
from agentscope.permission import PermissionMode
from agentscope.state import AgentState
from agentscope.tool import Toolkit
from agentscope.workspace import LocalWorkspace

from common import get_model, make_user


SKILL_MARKDOWN = """---
name: python-project-starter
description: Create a small, clean Python CLI project according to a fixed layout.
---

# Python Project Starter

Follow these steps when the user asks for a new Python example:

1. Use Write to create `main.py` with:
   - a module docstring
   - a `main()` function
   - the `if __name__ == "__main__":` guard
2. Use Write to create `requirements.txt`; keep dependencies minimal.
3. Use Read to verify both files.
4. Summarize the created files in Chinese.
"""


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="agentscope-skill-"))
    workdir = root / "workspace"
    skill_dir = root / "skills" / "python-project-starter"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(SKILL_MARKDOWN, encoding="utf-8")

    try:
        async with LocalWorkspace(
            workdir=str(workdir),
            skill_paths=[str(skill_dir)],
        ) as workspace:
            skills = await workspace.list_skills(agent_id="learner")
            print("=== 已加载 Skills ===")
            for skill in skills:
                print(f"  name={skill.name}")
                print(f"  description={skill.description}")
                print(f"  dir={skill.dir}")

            state = AgentState()
            state.permission_context.mode = PermissionMode.BYPASS
            agent = Agent(
                name="ProjectScaffolder",
                system_prompt=(
                    "你可以查看 Skill 列表并读取其完整指令，然后使用工作区"
                    "工具完成用户任务。\n\n" + await workspace.get_instructions()
                ),
                model=get_model(stream=False),
                toolkit=Toolkit(
                    tools=await workspace.list_tools(),
                    skills_or_loaders=skills,
                ),
                state=state,
                offloader=workspace,
            )

            reply = await agent.reply(
                make_user(
                    "使用 python-project-starter skill，在 workspace 目录下"
                    "创建一个最小 Python 项目，最后总结创建了什么。"
                )
            )
            print("\n=== Agent 最终回答 ===")
            print(reply.get_text_content())

            print("\n=== 工作区文件 ===")
            for file in sorted(workdir.rglob("*")):
                if file.is_file():
                    print(f"  - {file.relative_to(workdir)}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
