# -*- coding: utf-8 -*-
"""c06_06：裁剪消息、删除状态和生成摘要。"""

from __future__ import annotations

from langchain_core.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from common import banner, parser_for, print_json


def build_memory_graph(checkpointer=None):
    def echo_node(state: MessagesState) -> dict:
        return {}

    graph = StateGraph(MessagesState)
    graph.add_node("echo", echo_node)
    graph.add_edge(START, "echo")
    graph.add_edge("echo", END)
    return graph.compile(checkpointer=checkpointer)


def trim_oldest_messages(graph, config, keep: int) -> int:
    """删除旧消息，只保留最后 keep 条，返回删除数量。"""

    messages = graph.get_state(config).values["messages"]
    remove = messages[:-keep] if keep else messages
    if not remove:
        return 0

    graph.update_state(
        config,
        {
            "messages": [
                RemoveMessage(id=message.id, content="")
                for message in remove
            ]
        },
    )
    return len(remove)


def summarize_messages(messages) -> str:
    """用确定性规则生成本地摘要，生产环境可替换为 LLM 摘要。"""

    preview = [str(message.content)[:20] for message in messages]
    return f"共 {len(messages)} 条历史，摘要：{' / '.join(preview)}"


def main() -> None:
    parser_for(__doc__ or "Memory management")
    banner("c06_06 消息裁剪与摘要")

    graph = build_memory_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "memory-management"}}
    graph.invoke(
        {
            "messages": [
                {"role": "user", "content": "问题一"},
                {"role": "assistant", "content": "回答一"},
                {"role": "user", "content": "问题二"},
                {"role": "assistant", "content": "回答二"},
            ]
        },
        config=config,
    )

    before = graph.get_state(config).values["messages"]
    summary = summarize_messages(before)
    removed = trim_oldest_messages(graph, config, keep=2)
    after = graph.get_state(config).values["messages"]

    print_json(
        {
            "summary": summary,
            "removed": removed,
            "remaining": len(after),
        }
    )
    print("\n知识点：长期保留全部消息会放大成本，应按业务价值裁剪或摘要。")


if __name__ == "__main__":
    main()

