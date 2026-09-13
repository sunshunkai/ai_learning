# -*- coding: utf-8 -*-
"""c01_03：对比 Graph API 与 Functional API。

两个版本做同一件事：把文本转成大写，再拼接固定后缀。Graph API 适合需要
可视化、并行和共享状态的流程；Functional API 更适合改造已有 Python 代码。
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.func import entrypoint, task
from langgraph.graph import END, START, StateGraph

from common import banner, parser_for, print_json


class TextState(TypedDict):
    text: str
    result: str


def graph_normalize(state: TextState) -> dict[str, str]:
    return {"text": state["text"].strip().upper()}


def graph_finish(state: TextState) -> dict[str, str]:
    return {"result": f"{state['text']} [graph]"}


def build_graph_version():
    graph = StateGraph(TextState)
    graph.add_node("normalize", graph_normalize)
    graph.add_node("finish", graph_finish)
    graph.add_edge(START, "normalize")
    graph.add_edge("normalize", "finish")
    graph.add_edge("finish", END)
    return graph.compile()


@task
def normalize_task(text: str) -> str:
    return text.strip().upper()


@entrypoint()
def functional_version(text: str) -> str:
    normalized = normalize_task(text).result()
    return f"{normalized} [functional]"


def main() -> None:
    parser = parser_for(__doc__ or "Graph vs Functional API")
    parser.add_argument("--text", default="  hello langgraph  ")
    args = parser.parse_args()

    graph_result = build_graph_version().invoke({"text": args.text})
    functional_result = functional_version.invoke(args.text)

    banner("c01_03 Graph API 与 Functional API")
    print_json(
        {
            "graph": graph_result["result"],
            "functional": functional_result,
        }
    )
    print("\n选择建议：需要复杂协作和可视化优先 Graph；已有函数流程优先 Functional。")


if __name__ == "__main__":
    main()

