# -*- coding: utf-8 -*-
"""
LCEL：用 | 连接 prompt、model、output parser，并组合并行 chain

运行: python langchain-demo/03_lcel_chains.py
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from common import build_model


def main():
    model = build_model(temperature=0.8)
    # StrOutputParser 把 AIMessage 提取成普通字符串，方便后续打印或处理。
    parser = StrOutputParser()

    print("=== 最简单的 LCEL chain ===")
    joke_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个幽默的工程师。"),
            ("human", "围绕{topic}讲一个程序员笑话。"),
        ]
    )
    # LCEL 链可以像函数一样 invoke；输入字典会依次流过每个组件。
    joke_chain = joke_prompt | model | parser
    print(joke_chain.invoke({"topic": "Python"}), "\n")

    print("=== RunnablePassthrough：透传输入 ===")
    # 直接把字符串传给 Prompt 会缺少变量名，因此先包装成 {"topic": 原输入}。
    # RunnablePassthrough 表示不改数据，只把输入继续向后传递。
    passthrough_chain = (
        {"topic": RunnablePassthrough()}
        | joke_prompt
        | model
        | parser
    )
    print(passthrough_chain.invoke("加班"), "\n")

    print("=== RunnableParallel：同一输入并行两个任务 ===")
    summary_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是技术编辑。"),
            ("human", "用 30 字总结这段内容：{text}"),
        ]
    )
    summary_chain = summary_prompt | model | parser
    # RunnableParallel 用同一个输入并发执行多个分支，最后按 key 汇总结果。
    # 每个分支只读取自己需要的 topic 或 text，不会互相覆盖。
    parallel_chain = RunnableParallel(
        joke=joke_chain,
        summary=summary_chain,
    )
    result = parallel_chain.invoke(
        {
            "topic": "DeepSeek",
            "text": "LangChain 是一个用于构建大模型应用的框架，"
                    "提供模型接入、Prompt、记忆、工具、Agent 和 RAG 等能力。",
        }
    )
    print("joke   :", result["joke"])
    print("summary:", result["summary"])


if __name__ == "__main__":
    main()
