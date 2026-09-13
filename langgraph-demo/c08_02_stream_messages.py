# -*- coding: utf-8 -*-
"""c08_02：stream_mode="messages" 观察 LLM 消息输出。"""

from __future__ import annotations

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from common import banner, build_model, parser_for


def build_message_stream_graph(model):
    def chat_node(state: MessagesState) -> dict:
        response = model.invoke(
            [
                SystemMessage(content="用一句话回答。"),
                *state["messages"],
            ]
        )
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile()


def main() -> None:
    parser = parser_for(__doc__ or "Stream messages")
    parser.add_argument("--question", default="什么是 LangGraph？")
    args = parser.parse_args()

    model = build_model(
        offline=args.offline,
        model=args.model,
        text_responses=["LangGraph 是用于构建有状态 Agent 的编排运行时。"],
    )

    banner("c08_02 消息流")
    for part in build_message_stream_graph(model).stream(
        {"messages": [{"role": "user", "content": args.question}]},
        stream_mode="messages",
        version="v2",
    ):
        message, metadata = part["data"]
        print(
            f"[node={metadata['langgraph_node']}] "
            f"{message.content}"
        )
    print("\n知识点：消息流适合聊天 UI，状态流适合业务进度展示。")


if __name__ == "__main__":
    main()

