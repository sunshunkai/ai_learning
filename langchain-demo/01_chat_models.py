# -*- coding: utf-8 -*-
"""
聊天模型基础：消息对象、invoke、batch、stream

运行: python langchain-demo/01_chat_models.py
"""

from langchain_core.messages import HumanMessage, SystemMessage

from common import build_model


def main():
    # 聊天模型接收消息列表；temperature 越低，输出通常越稳定。
    model = build_model(temperature=0.3)

    print("=== invoke：带 system 与 human 消息 ===")
    messages = [
        # 每条调用都是无状态请求，因此需要显式提供 system 和 human 消息。
        SystemMessage("你是 Java 转 AI 学习的导师，回复保持简洁。"),
        HumanMessage("为什么 LLM 每次调用都要传入完整历史消息？"),
    ]
    response = model.invoke(messages)
    print(f"回复: {response.content}")
    # usage_metadata 包含输入、输出和总 token 数，便于观察调用成本。
    print(f"usage: {response.usage_metadata}\n")

    print("=== batch：一次批量处理多个问题 ===")
    # 每个元素会作为一次独立模型调用，返回顺序与输入顺序一致。
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
        # 首个或结尾 chunk 的 content 可能为空，过滤后界面更干净。
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
