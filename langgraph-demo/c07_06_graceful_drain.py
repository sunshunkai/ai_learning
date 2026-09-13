# -*- coding: utf-8 -*-
"""c07_06：RunControl 在当前 superstep 结束后安全排空。"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphDrained
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import RunControl, Runtime

from common import banner, parser_for, print_json


class DrainState(TypedDict):
    count: int


def build_drain_graph(checkpointer=None):
    def first_node(
        state: DrainState,
        runtime: Runtime,
    ) -> dict[str, int]:
        runtime.control.request_drain("sigterm")
        return {"count": state["count"] + 1}

    def second_node(state: DrainState) -> dict[str, int]:
        return {"count": state["count"] + 1}

    graph = StateGraph(DrainState)
    graph.add_node("first", first_node)
    graph.add_node("second", second_node)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)
    return graph.compile(checkpointer=checkpointer)


def run_drain_demo(checkpointer=None):
    graph = build_drain_graph(checkpointer or InMemorySaver())
    config = {"configurable": {"thread_id": "drain-demo"}}
    control = RunControl()

    try:
        graph.invoke({"count": 0}, config=config, control=control)
        paused_count = graph.get_state(config).values["count"]
    except GraphDrained:
        paused_count = graph.get_state(config).values["count"]

    resumed = graph.invoke(None, config=config)
    return paused_count, resumed


def main() -> None:
    parser_for(__doc__ or "Graceful drain")
    banner("c07_06 Graceful Drain")
    paused, resumed = run_drain_demo()
    print_json({"paused_at": paused, "resumed": resumed})
    print("\n知识点：Drain 不会中断正在运行的节点，只会在 superstep 边界停止。")


if __name__ == "__main__":
    main()
