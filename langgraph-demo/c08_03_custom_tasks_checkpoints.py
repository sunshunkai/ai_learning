# -*- coding: utf-8 -*-
"""c08_03：custom、tasks、debug 和 checkpoints 流模式。"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from common import banner, parser_for


class CustomState(TypedDict):
    value: int


def build_custom_graph(checkpointer=None):
    def progress_node(state: CustomState) -> dict[str, int]:
        writer = get_stream_writer()
        writer({"progress": 50, "message": "已完成一半"})
        return {"value": state["value"] + 1}

    graph = StateGraph(CustomState)
    graph.add_node("progress", progress_node)
    graph.add_edge(START, "progress")
    graph.add_edge("progress", END)
    return graph.compile(checkpointer=checkpointer)


def main() -> None:
    parser_for(__doc__ or "Custom stream modes")
    banner("c08_03 custom、tasks、debug、checkpoints")

    graph = build_custom_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "stream-modes"}}
    for part in graph.stream(
        {"value": 0},
        config=config,
        stream_mode=["custom", "tasks", "debug", "checkpoints"],
        version="v2",
    ):
        mode = part["type"]
        data = part["data"]
        if mode == "tasks":
            summary = {
                "name": data.get("name"),
                "done": "result" in data,
            }
        elif mode == "debug":
            summary = {
                "step": data.get("step"),
                "event": data.get("type"),
            }
        else:
            summary = data
        print(mode, summary)
    print("\n知识点：开发期使用 debug/checkpoints，业务 UI 优先 custom。")


if __name__ == "__main__":
    main()

