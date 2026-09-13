# -*- coding: utf-8 -*-
"""c06_02：查看 Checkpoint 历史，并从历史状态重新执行。"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from common import banner, parser_for, print_json


class HistoryState(TypedDict):
    count: Annotated[int, operator.add]


def build_history_graph(checkpointer=None):
    def increment_node(state: HistoryState) -> dict[str, int]:
        return {"count": 1}

    graph = StateGraph(HistoryState)
    graph.add_node("increment", increment_node)
    graph.add_edge(START, "increment")
    graph.add_edge("increment", END)
    return graph.compile(checkpointer=checkpointer)


def replay_snapshot(graph, snapshot):
    """从指定快照重新执行；重放可能再次执行后续节点。"""

    return graph.invoke(None, config=snapshot.config)


def main() -> None:
    parser_for(__doc__ or "State history replay")
    banner("c06_02 State History 与 Replay")

    graph = build_history_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "history-demo"}}
    graph.invoke({"count": 0}, config=config)
    graph.invoke({"count": 10}, config=config)

    history = list(graph.get_state_history(config))
    print_json(
        {
            "snapshots": len(history),
            "latest_count": history[0].values["count"],
            "next_nodes": list(history[0].next),
        }
    )
    if len(history) > 1:
        replay = replay_snapshot(graph, history[1])
        print_json({"replay_result": replay})
    print("\n知识点：Replay 从历史 Checkpoint 继续，不会修改更早的记录。")


if __name__ == "__main__":
    main()

