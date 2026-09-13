# -*- coding: utf-8 -*-
"""c07_01：interrupt() 暂停执行，Command(resume=...) 恢复执行。"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from common import banner, parser_for, print_json


class AskState(TypedDict):
    value: str


def ask_node(state: AskState) -> dict[str, str]:
    answer = interrupt("请输入你的名字：")
    return {"value": str(answer)}


def build_interrupt_graph(checkpointer=None):
    graph = StateGraph(AskState)
    graph.add_node("ask", ask_node)
    graph.add_edge(START, "ask")
    graph.add_edge("ask", END)
    return graph.compile(checkpointer=checkpointer)


def main() -> None:
    parser = parser_for(__doc__ or "Interrupt resume")
    parser.add_argument("--answer", default="Alice")
    args = parser.parse_args()

    banner("c07_01 interrupt 与 resume")
    graph = build_interrupt_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "interrupt-demo"}}

    first = graph.invoke({"value": "start"}, config=config)
    print_json({"paused": first.get("__interrupt__")})

    resumed = graph.invoke(Command(resume=args.answer), config=config)
    print_json({"final": resumed})
    print("\n知识点：resume 值会成为 interrupt() 在节点内的返回值。")


if __name__ == "__main__":
    main()

