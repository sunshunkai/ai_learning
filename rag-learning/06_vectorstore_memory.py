# -*- coding: utf-8 -*-
"""06 内存向量库：理解向量写入和相似度检索。"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from common import load_documents
from factory import build_embedding, build_vectorstore, format_docs


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
    results = store.similarity_search(question, k=2)
    print(format_docs(results))


if __name__ == "__main__":
    main()
