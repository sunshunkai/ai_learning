# -*- coding: utf-8 -*-
"""c02_04：普通边、条件边、START 和 END。"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from common import banner, parser_for, print_json


class ReviewState(TypedDict):
    text: str
    category: Literal["short", "long"]
    result: str


def classify_node(state: ReviewState) -> dict[str, str]:
    return {"category": "long" if len(state["text"]) > 10 else "short"}


def short_node(state: ReviewState) -> dict[str, str]:
    return {"result": "短文本流程"}


def long_node(state: ReviewState) -> dict[str, str]:
    return {"result": "长文本流程"}


def route_by_length(state: ReviewState) -> str:
    return state["category"]


def build_review_graph():
    graph = StateGraph(ReviewState)
    graph.add_node("classify", classify_node)
    graph.add_node("short", short_node)
    graph.add_node("long", long_node)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_by_length,
        {"short": "short", "long": "long"},
    )
    graph.add_edge("short", END)
    graph.add_edge("long", END)
    return graph.compile()


def main() -> None:
    parser = parser_for(__doc__ or "Nodes and edges")
    parser.add_argument("--text", default="这是一段超过十个字符的示例文本")
    args = parser.parse_args()

    banner("c02_04 Nodes 与 Edges")
    result = build_review_graph().invoke({"text": args.text})
    print_json(result)
    print("\n知识点：条件边通过路由函数选择分支，节点只负责处理状态。")


if __name__ == "__main__":
    main()

