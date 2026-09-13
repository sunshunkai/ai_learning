# -*- coding: utf-8 -*-
"""
LocalWorkspace 与 Skill。

Workspace 统一提供运行环境、内置工具、Skill 目录与 Offloader。
Skill 不是可调用工具，而是 Markdown 指令；Agent 先用 SkillViewer
读取完整指令，再用 Read/Write/Bash 等已有工具执行。

本示例使用项目共享的 ``skills/`` 目录，并为两个会话创建独立的
Workspace 和 AgentState。这样 Skill 元数据可以共享，但每个会话的
消息、已加载 Skill 和执行结果不会互相污染。

运行:
    python agentscope-demo/c07_01_workspace_skill.py
"""

import asyncio
import shutil
import tempfile
from contextlib import AsyncExitStack
from pathlib import Path

from agentscope.agent import Agent
from agentscope.permission import PermissionMode
from agentscope.state import AgentState
from agentscope.tool import Toolkit
from agentscope.workspace import LocalWorkspace

from common import get_model, make_user
from tools.skill_loader import discover_skills


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = PROJECT_ROOT / "skills"
# AgentScope 的 skill_paths 要求每个元素都是一个 Skill 目录，不会递归扫描根目录。
SKILL_DIRS = [str(skill.directory) for skill in discover_skills(SKILL_ROOT)]
SESSION_PROMPTS = {
    "session-alpha": (
        "请反转文本：AI Learning",
        "现在分析这个报错：ValueError: invalid literal for int()",
    ),
    "session-beta": (
        "请帮我搭建一个最小 Python CLI 项目，只需要 main.py 和 requirements.txt。",
    ),
}


async def create_session_agent(
    *,
    workspace,
    session_id: str,
    agent_id: str,
) -> tuple[Agent, list]:
    # list_skills() 返回给 Agent 的元数据；完整正文仍由 SkillViewer 按需读取。
    skills = await workspace.list_skills(agent_id=agent_id)
    state = AgentState(session_id=session_id)
    # The demo uses a disposable temporary workspace. Production should set
    # explicit allow/deny rules or run inside a sandbox instead of BYPASS.
    state.permission_context.mode = PermissionMode.BYPASS
    agent = Agent(
        name=f"SkillAgent-{session_id}",
        system_prompt=(
            "你可以查看允许的 Skill 列表，并只在需要时读取完整 Skill。"
            "先判断用户任务与哪个 Skill 最相关，再使用工作区已有工具执行。"
            "不要把其他会话的 Skill 或工作区内容带入当前会话。\n\n"
            + await workspace.get_instructions()
        ),
        model=get_model(stream=False),
        toolkit=Toolkit(
            tools=await workspace.list_tools(),
            # Toolkit 注入 name/description，并注册 SkillViewer 渐进读取工具。
            skills_or_loaders=skills,
        ),
        state=state,
        offloader=workspace,
    )
    return agent, skills


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="agentscope-skill-"))

    try:
        async with AsyncExitStack() as stack:
            sessions = {}
            for session_id in SESSION_PROMPTS:
                workdir = root / session_id / "workspace"
                # 每个会话使用独立工作区，文件产物和 Skill 分区互不干扰。
                workspace = await stack.enter_async_context(
                    LocalWorkspace(
                        workdir=str(workdir),
                        skill_paths=SKILL_DIRS,
                    )
                )
                agent, skills = await create_session_agent(
                    workspace=workspace,
                    session_id=session_id,
                    agent_id=f"{session_id}-developer",
                )
                sessions[session_id] = {
                    "agent": agent,
                    "skills": skills,
                    "workdir": workdir,
                }

                print(f"\n=== 会话 {session_id} 可用 Skills ===")
                for skill in skills:
                    print(f"  {skill.name}: {skill.description}")

            for session_id, prompts in SESSION_PROMPTS.items():
                session = sessions[session_id]
                agent = session["agent"]
                workdir = session["workdir"]
                print(f"\n=== 开始会话 {session_id} ===")

                for turn, prompt in enumerate(prompts, start=1):
                    print(f"\n[第 {turn} 轮] 用户: {prompt}")
                    # 同一个 Agent 连续 reply，AgentState 会保留本会话消息和上下文。
                    reply = await agent.reply(make_user(prompt))
                    print("Agent:", reply.get_text_content())

                print("\n=== 工作区文件 ===")
                for file in sorted(workdir.rglob("*")):
                    if not file.is_file():
                        continue
                    relative = file.relative_to(workdir)
                    # 输出时隐藏框架生成的 Skill 副本和 git 内部文件。
                    if relative.parts[0] in {"skills", ".git"}:
                        continue
                    if ".git" in relative.parts:
                        continue
                    print(f"  - {relative}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
