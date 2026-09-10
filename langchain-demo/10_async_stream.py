# -*- coding: utf-8 -*-
"""
异步调用：ainvoke、astream、abatch

运行: python langchain-demo/10_async_stream.py
"""

import asyncio

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common import build_model


async def main():
    model = build_model(temperature=0.4)

    print("=== ainvoke ===")
    result = await model.ainvoke("异步编程在 Python 里用什么关键字？")
    print(result.content, "\n")

    print("=== astream ===")
    print("流式: ", end="", flush=True)
    async for chunk in model.astream("用一句话介绍 DeepSeek。"):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n")

    print("=== abatch ===")
    results = await model.abatch(
        [
            "LangChain 的 LCEL 是什么？",
            "什么是 Agent？",
        ]
    )
    for index, item in enumerate(results, start=1):
        print(f"{index}. {item.content}")

    print("\n=== async chain ===")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "用一句话解释：{topic}"),
        ]
    )
    chain = prompt | model | StrOutputParser()
    text = await chain.ainvoke({"topic": "RAG"})
    print(text)


if __name__ == "__main__":
    asyncio.run(main())
