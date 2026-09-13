# -*- coding: utf-8 -*-
"""
Tool Middleware：在单个工具执行链上添加洋葱式钩子。

与 Agent Middleware 不同，它只包裹 tool.call()，因此即使不经过 Agent，
直接调用工具时也会生效。

运行:
    python agentscope-demo/c04_02_tool_middleware.py
"""

import asyncio
import time
from typing import Any, AsyncGenerator

from agentscope.tool import FunctionTool, ToolChunk, ToolMiddlewareBase


class AuditMiddleware(ToolMiddlewareBase):
    """打印参数、耗时，并观察执行结果。"""

    async def on_tool_call(
        self,
        tool,
        input_kwargs: dict[str, Any],
        next_handler,
    ) -> AsyncGenerator[ToolChunk, None]:
        print(f"[audit:before] {tool.name} args={input_kwargs}")
        started = time.perf_counter()
        async for chunk in next_handler(**input_kwargs):
            yield chunk
        elapsed = time.perf_counter() - started
        print(f"[audit:after]  {tool.name} elapsed={elapsed:.4f}s")


class UpperCaseInputMiddleware(ToolMiddlewareBase):
    """在进入内层前改写输入，演示中间件控制参数。"""

    async def on_tool_call(
        self,
        tool,
        input_kwargs: dict[str, Any],
        next_handler,
    ) -> AsyncGenerator[ToolChunk, None]:
        if "name" in input_kwargs:
            input_kwargs["name"] = input_kwargs["name"].upper()
        async for chunk in next_handler(**input_kwargs):
            yield chunk


def greet(name: str) -> str:
    """Return a greeting.

    Args:
        name: User name.
    """
    return f"Hello, {name}!"


async def main() -> None:
    tool = FunctionTool(
        greet,
        middlewares=[
            AuditMiddleware(),
            UpperCaseInputMiddleware(),
        ],
    )

    print("=== 1. 直接调用带两层中间件的工具 ===")
    result = await tool(name="nathan")
    async for chunk in result:
        print(f"[tool result] {chunk.content[0].text}")

    print("\n=== 2. 说明执行顺序 ===")
    print("Audit pre -> UpperCaseInput pre -> tool.call -> Upper post -> Audit post")


if __name__ == "__main__":
    asyncio.run(main())
