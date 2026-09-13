# -*- coding: utf-8 -*-
"""c02_03：区分图输入、图输出和节点间私有状态。"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from common import banner, parser_for, print_json


class InputState(TypedDict):
    question: str


class OutputState(TypedDict):
    answer: str


class OverallState(TypedDict):
    question: str
    answer: str
    draft: str


class PrivateState(TypedDict):
    normalized: str


def prepare_node(state: InputState) -> PrivateState:
    """生成只供节点间传递的私有字段。"""

    return {"normalized": state["question"].strip().upper()}


def answer_node(state: PrivateState) -> dict[str, str]:
    return {"answer": state["normalized"], "draft": "内部草稿"}


def build_private_graph():
    graph = StateGraph(
        OverallState,
        input_schema=InputState,
        output_schema=OutputState,
    )
    graph.add_node("prepare", prepare_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def main() -> None:
    parser_for(__doc__ or "Input output private state")
    banner("c02_03 输入、输出与私有状态")
    result = build_private_graph().invoke({"question": " hello graph "})
    print_json(result)
    print("\n知识点：output_schema 只暴露 answer，draft 不会进入公开结果。")


if __name__ == "__main__":
    main()

