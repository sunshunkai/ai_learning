# -*- coding: utf-8 -*-
"""11 Provider 切换：同一个业务逻辑只改配置。"""

from common import load_documents
from factory import RAGConfig, RAGFactory


def try_provider(name: str, config: RAGConfig):
    print(f"\n=== {name} ===")
    print(config)
    factory = RAGFactory(config, load_documents())
    try:
        chunks, _, _, retriever = factory.create_components()
        print(f"OK: chunks={len(chunks)}, retriever={type(retriever).__name__}")
    except RuntimeError as exc:
        print(f"SKIP: {exc}")


def main():
    try_provider(
        "默认 hash + memory",
        RAGConfig(embedding_provider="hash", vectorstore_provider="memory"),
    )
    try_provider(
        "本地 BGE + Chroma",
        RAGConfig(embedding_provider="bge-local", vectorstore_provider="chroma"),
    )
    try_provider(
        "阿里云 DashScope + Chroma",
        RAGConfig(embedding_provider="dashscope", vectorstore_provider="chroma"),
    )


if __name__ == "__main__":
    main()
