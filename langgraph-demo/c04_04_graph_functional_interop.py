# -*- coding: utf-8 -*-
"""c04_04：Functional API 调用 Graph，以及 Graph 调用 Functional API。"""

from __future__ import annotations

from typing import TypedDict

from langgraph.func import entrypoint, task
from langgraph.graph import END, START, StateGraph

from common import banner, parser_for, print_json


class TextState(TypedDict):
    text: str
    result: str


def normalize_node(state: TextState) -> dict[str, str]:
    return {"text": state["text"].strip().upper()}


def decorate_node(state: TextState) -> dict[str, str]:
    return {"result": f"{state['text']}-GRAPH"}


def build_uppercase_graph():
    graph = StateGraph(TextState)
    graph.add_node("normalize", normalize_node)
    graph.add_node("decorate", decorate_node)
    graph.add_edge(START, "normalize")
    graph.add_edge("normalize", "decorate")
    graph.add_edge("decorate", END)
    return graph.compile()


@task
def run_graph_task(text: str) -> str:
    result = build_uppercase_graph().invoke({"text": text})
    return result["result"]


@entrypoint()
def interop_workflow(text: str) -> str:
    return run_graph_task(text).result()


def graph_calls_functional_node(state: TextState) -> dict[str, str]:
    return {"result": interop_workflow.invoke(state["text"])}


def build_graph_calling_functional():
    graph = StateGraph(TextState)
    graph.add_node("functional", graph_calls_functional_node)
    graph.add_edge(START, "functional")
    graph.add_edge("functional", END)
    return graph.compile()


def main() -> None:
    parser = parser_for(__doc__ or "Graph functional interop")
    parser.add_argument("--text", default=" hi ")
    args = parser.parse_args()

    banner("c04_04 Graph 与 Functional API 互调")
    print_json(
        {
            "functional_calls_graph": interop_workflow.invoke(args.text),
            "graph_calls_functional": build_graph_calling_functional().invoke(
                {"text": args.text}
            )["result"],
        }
    )
    print("\n知识点：两种 API 共享任务、持久化和 Runtime 概念，可以组合使用。")


if __name__ == "__main__":
    main()

