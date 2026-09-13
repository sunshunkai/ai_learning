# -*- coding: utf-8 -*-
"""c05_01：结构化输出与工具调用预热。"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from common import banner, build_model, parser_for, print_json


class RouteSelection(BaseModel):
    route: Literal["math", "general"] = Field(description="用户问题的处理路由")


@tool
def add(a: float, b: float) -> float:
    """计算两个数字之和。"""

    return a + b


def main() -> None:
    parser = parser_for(__doc__ or "Structured output and tools")
    args = parser.parse_args()

    model = build_model(
        offline=args.offline,
        model=args.model,
        structured_responses={RouteSelection: RouteSelection(route="math")},
        tool_responses=[
            {"name": "add", "args": {"a": 1.5, "b": 2.5}},
        ],
    )

    structured = model.with_structured_output(RouteSelection).invoke(
        "请计算 1.5 + 2.5"
    )
    tool_request = model.bind_tools([add]).invoke("请调用 add 工具")

    banner("c05_01 结构化输出与工具调用")
    print_json(
        {
            "structured_route": structured.route,
            "tool_calls": tool_request.tool_calls,
        }
    )
    print("\n知识点：模型生成工具名和参数，真正的 Python 函数仍由程序执行。")


if __name__ == "__main__":
    main()

