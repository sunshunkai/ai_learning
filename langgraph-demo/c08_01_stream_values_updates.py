# -*- coding: utf-8 -*-
"""c08_01：对比 values 与 updates 两种状态流。"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from common import banner, parser_for, print_json


class StreamState(TypedDict):
    value: int


def first_node(state: StreamState) -> dict[str, int]:
    return {"value": state["value"] + 1}


def second_node(state: StreamState) -> dict[str, int]:
    return {"value": state["value"] + 1}


def build_stream_graph():
    graph = StateGraph(StreamState)
    graph.add_node("first", first_node)
    graph.add_node("second", second_node)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)
    return graph.compile()


def collect_modes(graph, initial_state: StreamState) -> list[dict]:
    return list(
        graph.stream(
            initial_state,
            stream_mode=["values", "updates"],
            version="v2",
        )
    )


def main() -> None:
    parser_for(__doc__ or "Stream values and updates")
    banner("c08_01 values 与 updates 流")
    print_json(collect_modes(build_stream_graph(), {"value": 0}))
    print("\n知识点：values 是完整状态，updates 是每个节点产生的状态增量。")


if __name__ == "__main__":
    main()

