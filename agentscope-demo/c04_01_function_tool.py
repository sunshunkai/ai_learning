# -*- coding: utf-8 -*-
"""
用 FunctionTool 装配 Agent。

Agent 会自动循环：模型请求工具 -> 权限检查 -> 执行 -> 回传结果 ->
继续推理，直到模型不再调用工具。

运行:
    python agentscope-demo/c04_01_function_tool.py
"""

import asyncio

from agentscope.agent import Agent
from agentscope.message import ToolCallBlock, ToolResultBlock
from agentscope.permission import PermissionBehavior, PermissionDecision
from agentscope.tool import FunctionTool, Toolkit

from common import get_model, make_user


def get_weather(city: str) -> str:
    """Query weather for a city.

    Args:
        city: A Chinese city name.

    Returns:
        Weather text.
    """
    data = {
        "杭州": "晴，26°C，适合散步",
        "北京": "晴，18°C，西北风 3 级",
        "上海": "多云，28°C，湿度较高",
    }
    return data.get(city, f"暂无 {city} 天气")


def multiply(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
        a: First number.
        b: Second number.

    Returns:
        Product.
    """
    return a * b


async def main() -> None:
    allow = PermissionDecision(
        behavior=PermissionBehavior.ALLOW,
        message="Demo tool allowed.",
    )
    weather = FunctionTool(get_weather, permission=allow)
    calculator = FunctionTool(multiply, permission=allow)

    agent = Agent(
        name="ToolAgent",
        system_prompt=(
            "当需要实时信息或精确计算时，使用提供的工具。"
            "最后用一句中文汇总结果。"
        ),
        model=get_model(stream=False, thinking_enable=True),
        toolkit=Toolkit(tools=[weather, calculator]),
    )

    reply = await agent.reply(
        make_user("杭州天气如何？另外请计算 123 * 45。")
    )
    print("=== 最终回答 ===")
    print(reply.get_text_content())

    print("\n=== Agent 在上下文中记录的工具轨迹 ===")
    for msg in agent.state.context:
        for block in msg.content:
            if isinstance(block, ToolCallBlock):
                print(f"  [call] {block.name}({block.input})")
            elif isinstance(block, ToolResultBlock):
                print(f"  [result] {block.name}: {block.output}")


if __name__ == "__main__":
    asyncio.run(main())
