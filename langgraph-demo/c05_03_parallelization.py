# -*- coding: utf-8 -*-
"""c05_03：并行执行多个独立 LLM 任务，再合并结果。"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from common import banner, build_model, parser_for, print_json


class ParallelState(TypedDict):
    question: str
    answers: Annotated[list[str], operator.add]


def _answer(model, system: str, question: str) -> str:
    response = model.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=question),
        ]
    )
    return str(response.content)


def build_parallel_graph(model):
    def short_node(state: ParallelState) -> dict[str, list[str]]:
        return {
            "answers": [
                _answer(model, "用一句话回答。", state["question"])
            ]
        }

    def detailed_node(state: ParallelState) -> dict[str, list[str]]:
        return {
            "answers": [
                _answer(model, "分三点详细回答。", state["question"])
            ]
        }

    graph = StateGraph(ParallelState)
    graph.add_node("short", short_node)
    graph.add_node("detailed", detailed_node)
    graph.add_edge(START, "short")
    graph.add_edge(START, "detailed")
    graph.add_edge("short", END)
    graph.add_edge("detailed", END)
    return graph.compile()


def main() -> None:
    parser = parser_for(__doc__ or "Parallelization")
    parser.add_argument("--question", default="如何开始学习 LangGraph？")
    args = parser.parse_args()

    model = build_model(
        offline=args.offline,
        model=args.model,
        text_responses=[
            "先理解状态、节点和边。",
            "1. 学 State\n2. 学控制流\n3. 学持久化",
        ],
    )

    banner("c05_03 Parallelization")
    result = build_parallel_graph(model).invoke(
        {"question": args.question, "answers": []}
    )
    print_json(result)
    print("\n知识点：并行适合互不依赖的子任务，但会增加合并和错误处理复杂度。")


if __name__ == "__main__":
    main()

