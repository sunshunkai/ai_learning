# -*- coding: utf-8 -*-
"""04 BGE Embedding：真实中文语义向量。"""

import math

from factory import build_embedding


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm)


def main():
    try:
        embedding = build_embedding("bge-local")
    except RuntimeError as exc:
        print(exc)
        print("请先安装: pip install langchain-community sentence-transformers")
        return

    texts = [
        "LangChain 是构建大模型应用的框架",
        "RAG 是检索增强生成",
        "DeepSeek 不提供 embedding",
    ]
    vectors = embedding.embed_documents(texts)
    query = embedding.embed_query("什么是 RAG")

    print(f"Embedding 维度: {len(vectors[0])}")
    for text, vector in zip(texts, vectors):
        print(f"cosine('{text}') = {cosine(query, vector):.3f}")


if __name__ == "__main__":
    main()
