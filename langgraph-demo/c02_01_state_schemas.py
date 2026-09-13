# -*- coding: utf-8 -*-
"""c02_01：TypedDict、Pydantic 和 MessagesState 三种状态结构。"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel

from common import banner, parser_for, print_json


class TypedState(TypedDict):
    question: str
    answer: str
    history: Annotated[list[str], operator.add]


class PydanticState(BaseModel):
    number: int
    squared: int | None = None


def build_typed_graph():
    def answer_node(state: TypedState) -> dict:
        return {
            "answer": state["question"].upper(),
            "history": ["typed-node"],
        }

    graph = StateGraph(TypedState)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def build_pydantic_graph():
    def square_node(state: PydanticState) -> dict[str, int]:
        return {"squared": state.number * state.number}

    graph = StateGraph(PydanticState)
    graph.add_node("square", square_node)
    graph.add_edge(START, "square")
    graph.add_edge("square", END)
    return graph.compile()


def build_messages_graph():
    def respond_node(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content="离线消息状态示例")]}

    graph = StateGraph(MessagesState)
    graph.add_node("respond", respond_node)
    graph.add_edge(START, "respond")
    graph.add_edge("respond", END)
    return graph.compile()


def main() -> None:
    parser_for(__doc__ or "State schemas")
    banner("c02_01 三种 State Schema")

    typed_result = build_typed_graph().invoke(
        {"question": "hello", "history": []}
    )
    pydantic_result = build_pydantic_graph().invoke({"number": 6})
    messages_result = build_messages_graph().invoke(
        {"messages": [{"role": "user", "content": "hi"}]}
    )

    print_json(
        {
            "typed_dict": typed_result,
            "pydantic": pydantic_result,
            "messages_count": len(messages_result["messages"]),
        }
    )
    print("\n知识点：TypedDict 轻量，Pydantic 可校验，MessagesState 专用于消息列表。")


if __name__ == "__main__":
    main()

