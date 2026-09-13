# -*- coding: utf-8 -*-
"""c08_05：使用 v3 event streaming 观察底层协议事件。"""

from __future__ import annotations

import warnings
from typing import TypedDict

from langchain_core._api import LangChainBetaWarning
from langgraph.graph import END, START, StateGraph

from common import banner, parser_for


class EventState(TypedDict):
    value: int


def build_event_graph():
    def event_node(state: EventState) -> dict[str, int]:
        return {"value": state["value"] + 1}

    graph = StateGraph(EventState)
    graph.add_node("event", event_node)
    graph.add_edge(START, "event")
    graph.add_edge("event", END)
    return graph.compile()


def collect_event_methods(graph) -> list[str]:
    warnings.filterwarnings("ignore", category=LangChainBetaWarning)
    with graph.stream_events({"value": 0}, version="v3") as stream:
        methods = []
        for event in stream:
            method = event.get("method")
            if method:
                methods.append(method)
        return methods


def main() -> None:
    parser_for(__doc__ or "Event streaming")
    banner("c08_05 Event Streaming")
    print({"methods": collect_event_methods(build_event_graph())})
    print("\n知识点：事件流暴露更底层、可组合的执行协议，适合构建前端投影。")


if __name__ == "__main__":
    main()

