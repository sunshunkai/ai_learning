# -*- coding: utf-8 -*-
"""c06_03：修改历史状态并通过 Fork 进入新的执行分支。"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from common import banner, parser_for, print_json


class TimeTravelState(TypedDict):
    topic: str
    prepared: str
    result: str


def prepare_node(state: TimeTravelState) -> dict[str, str]:
    return {"prepared": state["topic"]}


def write_node(state: TimeTravelState) -> dict[str, str]:
    return {"result": f"{state['prepared']}-draft"}


def build_time_travel_graph(checkpointer=None):
    graph = StateGraph(TimeTravelState)
    graph.add_node("prepare", prepare_node)
    graph.add_node("write", write_node)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "write")
    graph.add_edge("write", END)
    return graph.compile(checkpointer=checkpointer)


def fork_before_write(graph, config, new_topic: str):
    """找到 write 执行前的快照，修改数据并返回新的 fork config。"""

    for snapshot in graph.get_state_history(config):
        if "write" in snapshot.next:
            return graph.update_state(
                snapshot.config,
                {"topic": new_topic, "prepared": new_topic},
                as_node="prepare",
            )
    raise RuntimeError("没有找到 write 之前的 Checkpoint")


def main() -> None:
    parser_for(__doc__ or "Fork time travel")
    banner("c06_03 Fork 与 Time Travel")

    graph = build_time_travel_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "time-travel"}}
    original = graph.invoke({"topic": "A"}, config=config)
    original_snapshot = graph.get_state(config)
    fork_config = fork_before_write(graph, config, "B")
    forked = graph.invoke(None, config=fork_config)

    print_json(
        {
            "original": original["result"],
            "forked": forked["result"],
            "original_branch": graph.get_state(
                original_snapshot.config
            ).values["result"],
        }
    )
    print("\n知识点：Fork 会创建新分支，原 thread 的当前状态保持不变。")


if __name__ == "__main__":
    main()
