# -*- coding: utf-8 -*-
"""07 Chroma：本地持久化向量库。"""

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

    try:
        store = build_vectorstore(
            "chroma",
            chunks,
            embedding,
            persist_directory="./chroma_rag_demo",
        )
    except RuntimeError as exc:
        print(exc)
        print("请先安装: pip install langchain-community chromadb")
        return

    question = "RAG 是什么？"
    results = store.similarity_search(question, k=2)
    print(format_docs(results))


if __name__ == "__main__":
    main()
