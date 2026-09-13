# -*- coding: utf-8 -*-
"""09 基础 RAG：完整跑通检索增强生成。"""

import argparse

from common import build_deepseek_model, load_documents
from factory import RAGConfig, RAGFactory, format_docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="RAG 是什么？")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    config = RAGConfig(
        embedding_provider="hash",
        vectorstore_provider="memory",
        top_k=2,
    )
    factory = RAGFactory(config, load_documents())
    chunks, _, _, retriever = factory.create_components()
    print(f"chunks={len(chunks)}, provider=hash/memory")

    if args.offline:
        print("\n=== 检索结果 ===")
        print(format_docs(retriever.invoke(args.question)))
        return

    model = build_deepseek_model()
    chain = factory.create_chain(model, retriever)
    print("\n=== RAG 回答 ===")
    print(chain.invoke(args.question))


if __name__ == "__main__":
    main()
