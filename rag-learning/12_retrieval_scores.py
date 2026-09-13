# -*- coding: utf-8 -*-
"""12 检索评分：查看 similarity search 返回的分数。"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from common import load_documents
from factory import build_embedding, build_vectorstore


def main():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=120,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "],
    )
    chunks = splitter.split_documents(load_documents())
    embedding = build_embedding("hash")
    store = build_vectorstore("memory", chunks, embedding)

    question = "RAG 是什么？"
    results = store.similarity_search_with_score(question, k=3)
    print("注意：不同 vectorstore 的 score 含义可能不同，可能越大越相似，也可能越小越相似。")
    for rank, (doc, score) in enumerate(results, 1):
        print(
            f"[{rank}] score={score:.3f} "
            f"source={doc.metadata.get('source')} "
            f"text={doc.page_content[:40]}..."
        )


if __name__ == "__main__":
    main()
