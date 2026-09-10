# -*- coding: utf-8 -*-
"""
LangChain 1.x 快速入门（DeepSeek）

运行: python langchain-demo/quickstart.py
"""

from langchain_core.messages import HumanMessage, SystemMessage

from common import build_model


def main():
    model = build_model()

    print("=== 方式一：invoke 简单问答 ===")
    answer = model.invoke("用一句话说明 LangChain 是做什么的？")
    print(f"回复: {answer.content}\n")

    print("=== 方式二：stream 流式输出 ===")
    print("流式: ", end="", flush=True)
    for chunk in model.stream("请从 1 数到 5，每个数字一行。"):
        print(chunk.content, end="", flush=True)
    print("\n")

    print("=== 方式三：messages 消息列表 ===")
    messages = [
        SystemMessage("你是一个简短回答问题的助手。"),
        HumanMessage("1+1 等于几？"),
    ]
    resp = model.invoke(messages)
    print(f"回复: {resp.content}")


if __name__ == "__main__":
    main()
