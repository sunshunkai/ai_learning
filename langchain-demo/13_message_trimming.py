# -*- coding: utf-8 -*-
"""
消息裁剪：用 trim_messages 控制历史上下文长度

运行: python langchain-demo/13_message_trimming.py
"""

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)

from common import build_model


def main():
    system = SystemMessage("你是一个简洁的助手。")
    history = [
        system,
        HumanMessage("我叫小明。"),
        AIMessage("你好，小明。"),
        HumanMessage("我在学习 LangChain。"),
        AIMessage("很好，我们可以从模型、Prompt 和 Chain 开始。"),
        HumanMessage("请给我一个学习顺序。"),
        AIMessage("先基础调用，再学 LCEL、工具、Agent 和 RAG。"),
        HumanMessage("记住我的名字，并告诉我下一步学什么。"),
    ]

    trimmed = trim_messages(
        history,
        max_tokens=60,
        token_counter="approximate",
        strategy="last",
        include_system=True,
        start_on=HumanMessage,
        end_on=HumanMessage,
    )

    print(f"原始消息数: {len(history)}")
    print(f"裁剪后消息数: {len(trimmed)}")
    for message in trimmed:
        print(f"[{message.type}] {message.content}")

    reply = build_model(temperature=0.2).invoke(trimmed)
    print(f"\n模型回复: {reply.content}")


if __name__ == "__main__":
    main()
