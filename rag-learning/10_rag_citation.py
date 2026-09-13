# -*- coding: utf-8 -*-
"""10 带来源引用的 RAG。"""

import argparse

from langchain_core.prompts import ChatPromptTemplate

from common import build_deepseek_model, load_documents
from factory import RAGConfig, RAGFactory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="RAG 是什么？")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "只根据下面的资料回答，并在答案最后列出你使用过的来源。"
                "如果资料不足，请明确说明。\n\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    config = RAGConfig(
        embedding_provider="hash",
        vectorstore_provider="memory",
        top_k=2,
    )
    factory = RAGFactory(config, load_documents())
    _, _, _, retriever = factory.create_components()

    if args.offline:
        from factory import format_docs

        print("\n=== 检索结果 ===")
        print(format_docs(retriever.invoke(args.question)))
        return

    model = build_deepseek_model()
    chain = factory.create_chain(model, retriever, prompt=prompt)
    print("\n=== 带引用回答 ===")
    print(chain.invoke(args.question))


if __name__ == "__main__":
    main()
