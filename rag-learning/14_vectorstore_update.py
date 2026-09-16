# -*- coding: utf-8 -*-
"""14 向量库更新：用稳定 ID 演示 upsert、替换版本和删除旧数据。"""

import argparse

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from factory import build_chunk_id, build_embedding


SOURCE = "membership-policy.md"


def policy_document(version: int, content: str) -> tuple[Document, str]:
    chunk_id = build_chunk_id(SOURCE, version=version, chunk_index=0)
    return (
        Document(
            page_content=content,
            metadata={
                "source": SOURCE,
                "chunk_id": chunk_id,
                "version": version,
            },
        ),
        chunk_id,
    )


def create_store(
    provider: str,
    embedding,
    persist_directory: str,
    collection_name: str,
    reset: bool,
):
    if provider == "memory":
        return InMemoryVectorStore(embedding)

    from langchain_community.vectorstores import Chroma

    if reset:
        store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
        )
        store.delete_collection()

    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=persist_directory,
    )


def print_results(title: str, documents: list[Document]) -> None:
    print(f"\n=== {title} ===")
    if not documents:
        print("无结果")
        return
    for rank, document in enumerate(documents, 1):
        print(
            f"[{rank}] version={document.metadata.get('version')} "
            f"chunk_id={document.metadata.get('chunk_id')} "
            f"text={document.page_content}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="会员退款期限是多少天？")
    parser.add_argument(
        "--vectorstore",
        choices=["memory", "chroma"],
        default="memory",
    )
    parser.add_argument("--persist-directory", default="./chroma_update_demo")
    parser.add_argument("--collection-name", default="rag-update-demo")
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    embedding = build_embedding("hash")

    try:
        store = create_store(
            provider=args.vectorstore,
            embedding=embedding,
            persist_directory=args.persist_directory,
            collection_name=args.collection_name,
            reset=args.reset,
        )
    except ImportError as exc:
        print(exc)
        print("使用 Chroma 前请安装: pip install chromadb")
        return

    old_document, old_id = policy_document(
        version=1,
        content="会员退款政策：购买后 7 天内可以申请退款。",
    )
    store.add_documents([old_document], ids=[old_id])
    print_results("写入版本 1", store.similarity_search(args.question, k=3))

    new_document, new_id = policy_document(
        version=2,
        content="会员退款政策：购买后 30 天内可以申请退款。",
    )
    store.add_documents([new_document], ids=[new_id])
    store.delete([old_id])

    print(
        f"\n版本更新: upsert {new_id}, "
        f"delete {old_id}, vectorstore={args.vectorstore}"
    )
    print_results("替换后查询", store.similarity_search(args.question, k=3))


if __name__ == "__main__":
    main()
