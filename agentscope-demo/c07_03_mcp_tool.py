# -*- coding: utf-8 -*-
"""
通过 MCPClient 接入本地 STDIO MCP Server。

MCP 工具会被命名空间化为 mcp__{server_name}__{tool_name}。
本示例使用同目录的 c07_02_mcp_server.py，不依赖任何外部服务。

运行:
    python agentscope-demo/c07_03_mcp_tool.py
"""

import asyncio
import sys
from pathlib import Path

from agentscope.agent import Agent
from agentscope.mcp import MCPClient, StdioMCPConfig
from agentscope.permission import PermissionMode
from agentscope.state import AgentState
from agentscope.tool import Toolkit

from common import get_model, make_user


async def main() -> None:
    server_path = Path(__file__).with_name("c07_02_mcp_server.py")
    client = MCPClient(
        name="local_demo",
        is_stateful=True,
        mcp_config=StdioMCPConfig(
            command=sys.executable,
            args=[str(server_path)],
        ),
    )

    await client.connect()
    try:
        mcp_tools = await client.list_tools()
        print("=== MCP 发现到的工具 ===")
        for tool in mcp_tools:
            print(f"  {tool.name}: {tool.description.splitlines()[0]}")

        state = AgentState()
        state.permission_context.mode = PermissionMode.BYPASS
        agent = Agent(
            name="McpAgent",
            system_prompt="你可以调用 MCP 提供的本地工具完成任务。",
            model=get_model(stream=False, thinking_enable=True),
            toolkit=Toolkit(mcps=[client]),
            state=state,
        )

        reply = await agent.reply(
            make_user(
                "请查询当前时间，并把“AgentScope MCP”重复 3 次。"
            )
        )
        print("\n=== 最终回答 ===")
        print(reply.get_text_content())
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
