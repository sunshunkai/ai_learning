# -*- coding: utf-8 -*-
"""c09_01：把编译后的子图直接注册为父图节点。"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from common import banner, parser_for, print_json


class SharedState(TypedDict):
    text: str
    result: str


def build_child_graph():
    def uppercase_node(state: SharedState) -> dict[str, str]:
        return {"result": state["text"].upper()}

    graph = StateGraph(SharedState)
    graph.add_node("uppercase", uppercase_node)
    graph.add_edge(START, "uppercase")
    graph.add_edge("uppercase", END)
    return graph.compile()


def build_parent_graph():
    graph = StateGraph(SharedState)
    graph.add_node("child_graph", build_child_graph())
    graph.add_edge(START, "child_graph")
    graph.add_edge("child_graph", END)
    return graph.compile()


def main() -> None:
    parser_for(__doc__ or "Subgraph as node")
    banner("c09_01 编译子图作为节点")
    print_json(build_parent_graph().invoke({"text": "hello"}))
    print("\n知识点：父子图使用兼容 State 时，可以直接把子图当节点组合。")


if __name__ == "__main__":
    main()

