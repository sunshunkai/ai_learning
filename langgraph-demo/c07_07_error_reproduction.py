# -*- coding: utf-8 -*-
"""c07_07：复现 LangGraph 常见配置错误并说明修复方式。"""

from __future__ import annotations

import argparse
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from common import banner, parser_for


class NumberState(TypedDict):
    value: Annotated[int, operator.add]


def recursion_case() -> str:
    def loop_node(state: NumberState) -> dict[str, int]:
        return {"value": 1}

    graph = StateGraph(NumberState)
    graph.add_node("loop", loop_node)
    graph.add_edge(START, "loop")
    graph.add_edge("loop", "loop")
    try:
        graph.compile().invoke({"value": 0}, {"recursion_limit": 4})
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return "未触发预期错误"


def missing_checkpointer_case() -> str:
    def ask_node(state: NumberState) -> dict[str, int]:
        interrupt("需要人工输入")
        return {"value": 1}

    graph = StateGraph(NumberState)
    graph.add_node("ask", ask_node)
    graph.add_edge(START, "ask")
    graph.add_edge("ask", END)
    try:
        graph.compile().invoke(
            {"value": 0},
            {"configurable": {"thread_id": "missing"}},
        )
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return "未触发预期错误"


def invalid_return_case() -> str:
    def invalid_node(state: NumberState):
        return 1

    graph = StateGraph(NumberState)
    graph.add_node("invalid", invalid_node)
    graph.add_edge(START, "invalid")
    graph.add_edge("invalid", END)
    try:
        graph.compile().invoke({"value": 0})
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return "未触发预期错误"


def concurrent_update_case() -> str:
    def branch_node(state: NumberState) -> int:
        return 1

    graph = StateGraph(NumberState)
    graph.add_node("left", branch_node)
    graph.add_node("right", branch_node)
    graph.add_edge(START, "left")
    graph.add_edge(START, "right")
    graph.add_edge("left", END)
    graph.add_edge("right", END)
    try:
        graph.compile().invoke({"value": 0})
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return "未触发预期错误"


def documented_case(name: str, explanation: str) -> str:
    return f"SKIP：{name} 与具体模型/部署有关。{explanation}"


CASES = {
    "recursion": recursion_case,
    "missing-checkpointer": missing_checkpointer_case,
    "invalid-return": invalid_return_case,
    "concurrent-update": concurrent_update_case,
    "bad-history": lambda: documented_case(
        "INVALID_CHAT_HISTORY",
        "使用支持消息校验的模型时，非法角色或消息顺序会触发；DeepSeek 端到端测试会记录实际行为。",
    ),
    "multiple-subgraphs": lambda: documented_case(
        "MULTIPLE_SUBGRAPHS",
        "通常与部署入口暴露多个图有关，本地 StateGraph 示例不强行制造该错误。",
    ),
}


def main() -> None:
    parser = parser_for(__doc__ or "Error reproduction")
    parser.add_argument("--case", choices=CASES, default="recursion")
    args = parser.parse_args()

    banner(f"c07_07 错误复现：{args.case}")
    print(CASES[args.case]())
    print("\n知识点：先读懂错误类型和触发条件，再修改图结构、状态或运行配置。")


if __name__ == "__main__":
    main()

