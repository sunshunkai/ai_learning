# -*- coding: utf-8 -*-
"""c03_05：用 Command 同时更新状态和决定跳转。"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from common import banner, parser_for, print_json


class CommandState(TypedDict):
    value: Annotated[int, operator.add]
    path: Annotated[list[str], operator.add]


def first_node(state: CommandState) -> Command:
    return Command(
        update={"value": 1, "path": ["first"]},
        goto="second",
    )


def second_node(state: CommandState) -> Command:
    return Command(
        update={"value": 1, "path": ["second"]},
        goto=END,
    )


def build_command_graph():
    graph = StateGraph(CommandState)
    graph.add_node("first", first_node)
    graph.add_node("second", second_node)
    graph.add_edge(START, "first")
    return graph.compile()


def main() -> None:
    parser_for(__doc__ or "Command routing")
    banner("c03_05 Command 更新与跳转")
    result = build_command_graph().invoke({"value": 1, "path": []})
    print_json(result)
    print("\n知识点：Command 适合工具或节点需要同时修改状态并控制下一跳的场景。")


if __name__ == "__main__":
    main()

