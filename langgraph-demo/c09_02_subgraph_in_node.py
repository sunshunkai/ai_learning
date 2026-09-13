# -*- coding: utf-8 -*-
"""c09_02：在节点函数中调用子图并显式转换状态。"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from common import banner, parser_for, print_json


class ParentState(TypedDict):
    text: str
    child_output: str


class ChildState(TypedDict):
    text: str
    private_child_note: str
    result: str


def build_child_graph():
    def child_node(state: ChildState) -> dict[str, str]:
        return {
            "private_child_note": "子图内部信息",
            "result": state["text"].upper(),
        }

    graph = StateGraph(ChildState)
    graph.add_node("child", child_node)
    graph.add_edge(START, "child")
    graph.add_edge("child", END)
    return graph.compile()


def build_parent_graph():
    child = build_child_graph()

    def call_child_node(state: ParentState) -> dict[str, str]:
        child_result = child.invoke({"text": state["text"]})
        return {"child_output": child_result["result"]}

    graph = StateGraph(ParentState)
    graph.add_node("call_child", call_child_node)
    graph.add_edge(START, "call_child")
    graph.add_edge("call_child", END)
    return graph.compile()


def main() -> None:
    parser_for(__doc__ or "Subgraph in node")
    banner("c09_02 节点内调用子图")
    print_json(build_parent_graph().invoke({"text": "hello"}))
    print("\n知识点：显式调用适合需要转换输入输出、隔离私有状态的场景。")


if __name__ == "__main__":
    main()

