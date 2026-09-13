# -*- coding: utf-8 -*-
"""c08_04：多流模式同时观察父图和子图。"""

from __future__ import annotations

from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from common import banner, parser_for


class SharedState(TypedDict):
    text: str
    child_result: str


def build_child_graph():
    def child_node(state: SharedState) -> dict[str, str]:
        get_stream_writer()({"stage": "child", "text": state["text"]})
        return {"child_result": state["text"].upper()}

    graph = StateGraph(SharedState)
    graph.add_node("child", child_node)
    graph.add_edge(START, "child")
    graph.add_edge("child", END)
    return graph.compile()


def build_parent_graph():
    def finalize_node(state: SharedState) -> dict[str, str]:
        get_stream_writer()({"stage": "parent", "result": state["child_result"]})
        return {"text": f"{state['child_result']}-DONE"}

    graph = StateGraph(SharedState)
    graph.add_node("child_graph", build_child_graph())
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "child_graph")
    graph.add_edge("child_graph", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def main() -> None:
    parser_for(__doc__ or "Multiple stream modes subgraph")
    banner("c08_04 多模式流与子图")

    for part in build_parent_graph().stream(
        {"text": "hello"},
        stream_mode=["updates", "custom"],
        version="v2",
    ):
        print(part)
    print("\n知识点：namespace 可以区分父图与子图产生的数据。")


if __name__ == "__main__":
    main()

