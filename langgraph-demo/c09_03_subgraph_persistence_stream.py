# -*- coding: utf-8 -*-
"""c09_03：子图持久化继承和子图 custom 流。"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from common import banner, parser_for


class State(TypedDict):
    text: str
    result: str


def build_child_graph(checkpointer=None):
    def child_node(state: State) -> dict[str, str]:
        get_stream_writer()({"stage": "child", "message": "开始子图处理"})
        return {"result": state["text"].upper()}

    graph = StateGraph(State)
    graph.add_node("child", child_node)
    graph.add_edge(START, "child")
    graph.add_edge("child", END)
    return graph.compile(checkpointer=checkpointer)


def build_parent_graph():
    checkpointer = InMemorySaver()
    child = build_child_graph(checkpointer=checkpointer)

    def parent_node(state: State) -> dict[str, str]:
        get_stream_writer()({"stage": "parent", "result": state["result"]})
        return {"text": f"{state['result']}-PARENT"}

    graph = StateGraph(State)
    graph.add_node("child_graph", child)
    graph.add_node("parent", parent_node)
    graph.add_edge(START, "child_graph")
    graph.add_edge("child_graph", "parent")
    graph.add_edge("parent", END)
    return graph.compile(checkpointer=checkpointer)


def main() -> None:
    parser_for(__doc__ or "Subgraph persistence stream")
    banner("c09_03 子图持久化与流式输出")

    config = {"configurable": {"thread_id": "subgraph-stream"}}
    for part in build_parent_graph().stream(
        {"text": "hello"},
        config=config,
        stream_mode="custom",
        subgraphs=True,
        version="v2",
    ):
        print(part)
    print("\n知识点：持久化子图可以继承父 Checkpointer，并保留独立 namespace。")


if __name__ == "__main__":
    main()
