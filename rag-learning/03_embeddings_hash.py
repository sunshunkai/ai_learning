# -*- coding: utf-8 -*-
"""03 Hash Embedding：理解向量化和相似度计算。"""

import math

from factory import build_embedding


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm)


def main():
    embedding = build_embedding("hash", dimensions=64)
    texts = [
        "LangChain 是构建大模型应用的框架",
        "RAG 是检索增强生成",
        "DeepSeek 不提供 embedding",
    ]
    vectors = embedding.embed_documents(texts)

    for text, vector in zip(texts, vectors):
        nonzero = sum(1 for value in vector if value != 0)
        print(f"文本: {text}")
        print(f"  维度={len(vector)}, 非零维度={nonzero}\n")

    query_vector = embedding.embed_query("RAG 是什么")
    for text, vector in zip(texts, vectors):
        print(f"cosine('{text}') = {cosine(query_vector, vector):.3f}")


if __name__ == "__main__":
    main()
