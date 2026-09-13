# -*- coding: utf-8 -*-
"""c07_03：人工审核并修改模型或程序生成的内容。"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from common import banner, parser_for, print_json


class ReviewState(TypedDict):
    draft: str


def review_node(state: ReviewState) -> dict[str, str]:
    payload = {
        "instruction": "请检查并返回修改后的文本",
        "draft": state["draft"],
    }
    edited = interrupt(payload)
    if isinstance(edited, dict):
        edited = edited.get("draft", edited)
    return {"draft": str(edited)}


def build_review_graph(checkpointer=None):
    graph = StateGraph(ReviewState)
    graph.add_node("review", review_node)
    graph.add_edge(START, "review")
    graph.add_edge("review", END)
    return graph.compile(checkpointer=checkpointer)


def main() -> None:
    parser = parser_for(__doc__ or "Review edit state")
    parser.add_argument("--draft", default="这是模型生成的初稿。")
    args = parser.parse_args()

    banner("c07_03 人工审核与状态修改")
    graph = build_review_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "review-demo"}}
    paused = graph.invoke({"draft": args.draft}, config=config)
    print_json({"pending": paused.get("__interrupt__")})

    edited = graph.invoke(
        Command(resume={"draft": "这是人工修改后的内容。"}),
        config=config,
    )
    print_json(edited)
    print("\n知识点：恢复载荷可以是结构化对象，用于审核、修改或补充字段。")


if __name__ == "__main__":
    main()

