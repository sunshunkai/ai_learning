# -*- coding: utf-8 -*-
"""c03_04：使用 Send 动态创建 Map-Reduce 工作项。"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from common import banner, parser_for, print_json


class MapState(TypedDict):
    items: list[str]
    results: Annotated[list[str], operator.add]


class WorkerState(TypedDict):
    item: str


def fan_out(state: MapState) -> list[Send]:
    return [Send("worker", {"item": item}) for item in state["items"]]


def worker_node(state: WorkerState) -> dict[str, list[str]]:
    return {"results": [state["item"].upper()]}


def build_send_graph():
    graph = StateGraph(MapState)
    graph.add_node("worker", worker_node)
    graph.add_conditional_edges(START, fan_out, ["worker"])
    graph.add_edge("worker", END)
    return graph.compile()


def main() -> None:
    parser_for(__doc__ or "Send map reduce")
    banner("c03_04 Send 与 Map-Reduce")
    result = build_send_graph().invoke({"items": ["alpha", "beta", "gamma"]})
    print_json(result)
    print("\n知识点：Send 在运行时创建未知数量的 worker 任务。")


if __name__ == "__main__":
    main()

