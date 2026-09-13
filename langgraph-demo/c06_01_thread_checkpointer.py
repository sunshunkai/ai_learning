# -*- coding: utf-8 -*-
"""c06_01：InMemorySaver 按 thread_id 保存短期会话。"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from common import banner, parser_for, print_json


def build_thread_graph(checkpointer=None):
    def echo_node(state: MessagesState) -> dict:
        return {}

    graph = StateGraph(MessagesState)
    graph.add_node("echo", echo_node)
    graph.add_edge(START, "echo")
    graph.add_edge("echo", END)
    return graph.compile(checkpointer=checkpointer)


def main() -> None:
    parser_for(__doc__ or "Thread checkpointer")
    banner("c06_01 thread_id 与短期记忆")

    graph = build_thread_graph(InMemorySaver())
    alice = {"configurable": {"thread_id": "alice"}}
    bob = {"configurable": {"thread_id": "bob"}}

    graph.invoke({"messages": [{"role": "user", "content": "你好"}]}, alice)
    alice_result = graph.invoke(
        {"messages": [{"role": "user", "content": "继续"}]},
        alice,
    )
    bob_result = graph.invoke(
        {"messages": [{"role": "user", "content": "新会话"}]},
        bob,
    )

    print_json(
        {
            "alice_messages": len(alice_result["messages"]),
            "bob_messages": len(bob_result["messages"]),
            "current_alice_state": len(graph.get_state(alice).values["messages"]),
        }
    )
    print("\n知识点：同一个 thread_id 共享历史，不同 thread_id 相互隔离。")


if __name__ == "__main__":
    main()

