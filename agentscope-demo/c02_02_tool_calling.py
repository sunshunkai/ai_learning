# -*- coding: utf-8 -*-
"""
模型层工具调用：声明 schema -> 模型返回 ToolCallBlock -> 执行 -> 回传结果。

这一章先不引入 Agent 的 ReAct 循环，而是手动演示最底层的一次工具往返，
便于理解 Agent 内部自动做了哪些事情。

运行:
    python agentscope-demo/c02_02_tool_calling.py
"""

import asyncio
import json

from agentscope.message import (
    Msg,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultState,
)
from agentscope.tool import FunctionTool, Toolkit, ToolChoice

from common import get_model, make_user, print_usage, stream_and_collect, text_from_response


def get_weather(city: str) -> str:
    """Get the current weather for a Chinese city.

    Args:
        city: The city name, for example Hangzhou.

    Returns:
        A short weather description.
    """
    data = {
        "杭州": "晴，26°C，湿度 55%",
        "北京": "晴，18°C，西北风 3 级",
        "上海": "多云，28°C，湿度 70%",
        "深圳": "阵雨，30°C，湿度 85%",
    }
    return data.get(city, f"暂时没有 {city} 的实时数据")


def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression.

    Args:
        expression: A Python expression containing only digits and + - * / ( ).

    Returns:
        The calculated result.
    """
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        raise ValueError("表达式包含不支持的字符")
    return str(eval(expression))  # noqa: S307 - 输入已做字符白名单限制


async def main() -> None:
    print("=== 1. 生成工具 schema ===")
    toolkit = Toolkit(
        tools=[
            FunctionTool(get_weather),
            FunctionTool(calculate),
        ]
    )
    schemas = await toolkit.get_tool_schemas()
    print(json.dumps(schemas, ensure_ascii=False, indent=2))

    model = get_model(stream=True, thinking_enable=True)
    messages = [
        make_user("杭州今天天气如何？顺便计算 (17 + 23) * 4。")
    ]

    print("\n=== 2. 第一轮：模型决定调用哪些工具 ===")
    response = await stream_and_collect(
        model,
        messages,
        tools=schemas,
        tool_choice=ToolChoice(mode="auto"),
    )
    print_usage(response)

    tool_calls = [
        block
        for block in response.content
        if isinstance(block, ToolCallBlock)
    ]
    if not tool_calls:
        print("模型没有请求工具，直接回答:")
        print(text_from_response(response))
        return

    print("\n=== 3. 开发者执行真实函数 ===")
    dispatcher = {
        "get_weather": lambda args: get_weather(**args),
        "calculate": lambda args: calculate(**args),
    }
    results: list[ToolResultBlock] = []
    for tool_call in tool_calls:
        arguments = json.loads(tool_call.input)
        print(f"  调用 {tool_call.name}({arguments})")
        try:
            output = dispatcher[tool_call.name](arguments)
            state = ToolResultState.SUCCESS
        except Exception as exc:
            output = f"执行失败: {exc}"
            state = ToolResultState.ERROR
        print(f"  结果: {output}")
        results.append(
            ToolResultBlock(
                id=tool_call.id,
                name=tool_call.name,
                output=output,
                state=state,
            )
        )

    messages.extend(
        [
            Msg(
                name="assistant",
                role="assistant",
                content=response.content,
            ),
            Msg(
                name="tool",
                role="assistant",
                content=results,
            ),
        ]
    )

    print("\n=== 4. 第二轮：基于工具结果生成最终回答 ===")
    final_response = await stream_and_collect(
        model,
        messages,
        tools=schemas,
        tool_choice=ToolChoice(mode="auto"),
    )
    print_usage(final_response)


if __name__ == "__main__":
    asyncio.run(main())
