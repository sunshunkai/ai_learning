# -*- coding: utf-8 -*-
"""RAG demo：同一个业务逻辑，通过参数切换 embedding 和 vectorstore。"""

from __future__ import annotations

import argparse

from common import build_deepseek_model, load_documents
from factory import RAGConfig, RAGFactory, format_docs


def parse_args():
    parser = argparse.ArgumentParser(description="LangChain RAG 学习 Demo")
    parser.add_argument("--question", default="RAG 是什么？DeepSeek 是否提供 embedding？")
    parser.add_argument("--embedding", default="hash", choices=["hash", "bge-local", "dashscope"])
    parser.add_argument("--vectorstore", default="memory", choices=["memory", "chroma", "faiss"])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--offline", action="store_true", help="只检索，不调用 LLM")
    parser.add_argument("--show-docs", action="store_true", help="显示检索到的文档")
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = RAGConfig(
        embedding_provider=args.embedding,
        vectorstore_provider=args.vectorstore,
        top_k=args.top_k,
    )
    factory = RAGFactory(config, load_documents())

    print(f"Provider: embedding={args.embedding}, vectorstore={args.vectorstore}")
    chunks, _, _, retriever = factory.create_components()
    print(f"文档切分完成: {len(chunks)} chunks")

    retrieved = retriever.invoke(args.question)
    if args.show_docs or args.offline:
        print("\n=== 检索结果 ===")
        print(format_docs(retrieved))

    if args.offline:
        return

    model = build_deepseek_model(model=args.model)
    chain = factory.create_chain(model, retriever)
    print("\n=== RAG 回答 ===")
    print(chain.invoke(args.question))


if __name__ == "__main__":
    main()
