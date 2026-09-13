# -*- coding: utf-8 -*-
"""c07_04：顺序处理一个执行中的多个中断。"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from common import banner, parser_for, print_json


class MultiInterruptState(TypedDict):
    answers: Annotated[list[str], operator.add]


def ask_name_node(state: MultiInterruptState) -> dict[str, list[str]]:
    answer = interrupt("工具一：请输入名称")
    return {"answers": [f"name={answer}"]}


def ask_count_node(state: MultiInterruptState) -> dict[str, list[str]]:
    answer = interrupt("工具二：请输入数量")
    return {"answers": [f"count={answer}"]}


def build_multi_interrupt_graph(checkpointer=None):
    graph = StateGraph(MultiInterruptState)
    graph.add_node("ask_name", ask_name_node)
    graph.add_node("ask_count", ask_count_node)
    graph.add_edge(START, "ask_name")
    graph.add_edge("ask_name", "ask_count")
    graph.add_edge("ask_count", END)
    return graph.compile(checkpointer=checkpointer)


def main() -> None:
    parser_for(__doc__ or "Multiple interrupts")
    banner("c07_04 一个执行中的多个中断")

    graph = build_multi_interrupt_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "multi-interrupt"}}

    first = graph.invoke({"answers": []}, config=config)
    second = graph.invoke(Command(resume="Alice"), config=config)
    final = graph.invoke(Command(resume=3), config=config)

    print_json(
        {
            "first_interrupt": first.get("__interrupt__"),
            "second_interrupt": second.get("__interrupt__"),
            "final": final,
        }
    )
    print("\n知识点：一次执行可以多次暂停；并行中断时应使用 ID 映射恢复。")


if __name__ == "__main__":
    main()

