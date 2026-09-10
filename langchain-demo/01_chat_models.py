# -*- coding: utf-8 -*-
"""
聊天模型基础：消息对象、invoke、batch、stream

运行: python langchain-demo/01_chat_models.py
"""

from langchain_core.messages import HumanMessage, SystemMessage

from common import build_model


def main():
    model = build_model(temperature=0.3)

    print("=== invoke：带 system 与 human 消息 ===")
    messages = [
        SystemMessage("你是 Java 转 AI 学习的导师，回复保持简洁。"),
        HumanMessage("为什么 LLM 每次调用都要传入完整历史消息？"),
    ]
    response = model.invoke(messages)
    print(f"回复: {response.content}")
    print(f"usage: {response.usage_metadata}\n")

    print("=== batch：一次批量处理多个问题 ===")
    answers = model.batch(
        [
            "什么是 Token？",
            "什么是 embedding？",
        ]
    )
    for i, answer in enumerate(answers, start=1):
        print(f"{i}. {answer.content}")

    print("\n=== stream：流式输出 ===")
    print("流式: ", end="", flush=True)
    for chunk in model.stream("介绍一下 LangChain 的 LCEL。"):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
