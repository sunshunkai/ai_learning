# -*- coding: utf-8 -*-
"""c02_02：Reducer 决定并发或多次更新如何合并。

默认行为是“后写覆盖”，而 Annotated[T, reducer] 可以把多个更新合并为列表、
累加数值，或通过 add_messages 正确合并聊天消息。
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, MessagesState, StateGraph

from common import banner, parser_for, print_json


class CounterState(TypedDict):
    count: Annotated[int, operator.add]


def build_message_graph(checkpointer=None):
    """构建只做状态回显的 MessagesState 图。"""

    def echo_node(state: MessagesState) -> dict:
        return {}

    graph = StateGraph(MessagesState)
    graph.add_node("echo", echo_node)
    graph.add_edge(START, "echo")
    graph.add_edge("echo", END)
    return graph.compile(checkpointer=checkpointer)


def build_counter_graph(checkpointer=None):
    """每次执行都把 count 增加 1。"""

    def increment_node(state: CounterState) -> dict[str, int]:
        return {"count": 1}

    graph = StateGraph(CounterState)
    graph.add_node("increment", increment_node)
    graph.add_edge(START, "increment")
    graph.add_edge("increment", END)
    return graph.compile(checkpointer=checkpointer)


def main() -> None:
    parser_for(__doc__ or "Reducers")
    banner("c02_02 Reducer 与消息合并")

    messages = build_message_graph().invoke(
        {
            "messages": [
                {"role": "user", "content": "第一条"},
                {"role": "assistant", "content": "第二条"},
            ]
        }
    )
    counter = build_counter_graph()
    first = counter.invoke({"count": 10})
    second = counter.invoke({"count": 20})

    print_json(
        {
            "message_roles": [message.type for message in messages["messages"]],
            "counter_first": first,
            "counter_second": second,
        }
    )
    print("\n知识点：add_messages 会保留历史；operator.add 会累加。")


if __name__ == "__main__":
    main()

