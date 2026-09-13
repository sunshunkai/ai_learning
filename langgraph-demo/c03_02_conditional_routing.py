# -*- coding: utf-8 -*-
"""c03_02：根据状态条件动态选择后续节点。"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from common import banner, parser_for, print_json


class RouteState(TypedDict):
    value: int
    route: Literal["small", "large"]
    result: str


def classify_node(state: RouteState) -> dict[str, str]:
    return {"route": "large" if state["value"] >= 5 else "small"}


def small_node(state: RouteState) -> dict[str, str]:
    return {"result": "small"}


def large_node(state: RouteState) -> dict[str, str]:
    return {"result": "large"}


def route_selector(state: RouteState) -> str:
    return state["route"]


def build_conditional_graph():
    graph = StateGraph(RouteState)
    graph.add_node("classify", classify_node)
    graph.add_node("small", small_node)
    graph.add_node("large", large_node)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_selector,
        {"small": "small", "large": "large"},
    )
    graph.add_edge("small", END)
    graph.add_edge("large", END)
    return graph.compile()


def main() -> None:
    parser = parser_for(__doc__ or "Conditional routing")
    parser.add_argument("--value", type=int, default=8)
    args = parser.parse_args()

    banner("c03_02 条件路由")
    print_json(build_conditional_graph().invoke({"value": args.value}))
    print("\n知识点：路由函数只选择路径，状态修改仍由节点负责。")


if __name__ == "__main__":
    main()

