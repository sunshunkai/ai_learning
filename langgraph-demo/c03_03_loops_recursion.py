# -*- coding: utf-8 -*-
"""c03_03：循环、终止条件和递归限制。

循环如果没有正确终止，LangGraph 会触发 GRAPH_RECURSION_LIMIT。生产中应
同时设计业务终止条件和合理的 recursion_limit。
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from common import banner, parser_for, print_json


class LoopState(TypedDict):
    count: int
    limit: int


def increment_node(state: LoopState) -> dict[str, int]:
    return {"count": state["count"] + 1}


def should_continue(state: LoopState) -> str:
    return "increment" if state["count"] < state["limit"] else END


def build_loop_graph():
    graph = StateGraph(LoopState)
    graph.add_node("increment", increment_node)
    graph.add_edge(START, "increment")
    graph.add_conditional_edges("increment", should_continue)
    return graph.compile()


def main() -> None:
    parser = parser_for(__doc__ or "Loop recursion")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    banner("c03_03 循环与递归限制")
    result = build_loop_graph().invoke({"count": 0, "limit": args.limit})
    print_json(result)
    print("\n知识点：业务终止条件优先；recursion_limit 是最后一道保护。")


if __name__ == "__main__":
    main()

