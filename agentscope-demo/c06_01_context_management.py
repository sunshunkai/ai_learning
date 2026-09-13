# -*- coding: utf-8 -*-
"""
上下文管理：环境注入、压缩与卸载。

1. InjectionConfig 把时间、任务、上下文用量等信息注入 HintBlock。
2. ContextConfig 在超阈值时压缩旧消息。
3. Offloader 把被压缩的内容持久化，模型以后仍可用 Read/Grep 回查。

运行:
    python agentscope-demo/c06_01_context_management.py
"""

import asyncio
import shutil
import tempfile
from pathlib import Path

from agentscope.agent import Agent, ContextConfig, InjectionConfig
from agentscope.event import HintBlockEvent, TextBlockDeltaEvent
from agentscope.tool import Toolkit
from agentscope.workspace import LocalWorkspace

from common import get_model, make_user


async def environment_injection() -> None:
    print("=== 1. Runtime State 注入 ===")
    agent = Agent(
        name="ContextAware",
        system_prompt="你是能感知运行环境的助手。",
        model=get_model(stream=True),
        toolkit=Toolkit(),
        injection_config=InjectionConfig(
            timezone="Asia/Shanghai",
            extra_fields={
                "project": "ai-learning",
                "environment": "demo",
            },
        ),
    )

    async for event in agent.reply_stream(
        make_user("请告诉我当前项目名和你所处的时区。")
    ):
        if isinstance(event, HintBlockEvent):
            print("[HintBlockEvent]")
            if isinstance(event.hint, str):
                print(event.hint)
            else:
                for block in event.hint:
                    print(f"  {block}")
        elif isinstance(event, TextBlockDeltaEvent):
            print(event.delta, end="", flush=True)
    print("\n")


async def compression_and_offload() -> None:
    print("=== 2. Context Compression + Offloader ===")
    workdir = Path(tempfile.mkdtemp(prefix="agentscope-context-"))
    long_history = (
        "这是一个需要被摘要的历史细节。用户正在开发 AI 学习项目；"
        "项目使用 Python、AgentScope 和 DeepSeek；希望示例足够详细。"
        + "重要细节：用户偏好简洁中文、先给代码再给解释。"
        * 12
    )

    try:
        async with LocalWorkspace(workdir=str(workdir)) as workspace:
            agent = Agent(
                name="LongTaskAgent",
                system_prompt="你是能总结上下文的助手。",
                # 故意使用较小窗口，方便触发压缩逻辑。
                model=get_model(stream=False, context_size=500),
                toolkit=Toolkit(),
                offloader=workspace,
                context_config=ContextConfig(
                    trigger_ratio=0.8,
                    reserve_ratio=0.2,
                    compression_tool_enabled=True,
                ),
            )

            await agent.observe(make_user(long_history))
            print(f"压缩前 context 消息数: {len(agent.state.context)}")
            await agent.compress_context()

            print(f"压缩后 context 消息数: {len(agent.state.context)}")
            print(f"summary 长度: {len(str(agent.state.summary))}")
            print(f"summary 内容:\n{agent.state.summary}\n")

            offload_files = list((workdir / "sessions").rglob("*"))
            print(f"Offloader 文件数: {len(offload_files)}")
            for file in offload_files:
                print(f"  - {file.relative_to(workdir)}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


async def main() -> None:
    await environment_injection()
    await compression_and_offload()


if __name__ == "__main__":
    asyncio.run(main())
