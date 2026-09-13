# -*- coding: utf-8 -*-
"""c03_01：并行 fan-out/fan-in 与 Reducer 聚合。"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from common import banner, parser_for, print_json


class BranchState(TypedDict):
    branches: Annotated[list[str], operator.add]


def left_node(state: BranchState) -> dict[str, list[str]]:
    return {"branches": ["left"]}


def right_node(state: BranchState) -> dict[str, list[str]]:
    return {"branches": ["right"]}


def build_branch_graph():
    graph = StateGraph(BranchState)
    graph.add_node("left", left_node)
    graph.add_node("right", right_node)
    graph.add_edge(START, "left")
    graph.add_edge(START, "right")
    graph.add_edge("left", END)
    graph.add_edge("right", END)
    return graph.compile()


def main() -> None:
    parser_for(__doc__ or "Parallel branches")
    banner("c03_01 并行分支与状态聚合")
    result = build_branch_graph().invoke({"branches": []})
    print_json(result)
    print("\n知识点：两个节点并行写入同一字段时，必须使用兼容的 Reducer。")


if __name__ == "__main__":
    main()

