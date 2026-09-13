# -*- coding: utf-8 -*-
"""02 文本分块：观察 chunk_size 和 chunk_overlap 的影响。"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from common import load_documents


def show_split(config_name: str, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "],
    )
    chunks = splitter.split_documents(load_documents())
    print(f"\n=== {config_name}: chunk_size={chunk_size}, overlap={chunk_overlap} ===")
    print(f"chunk 数量: {len(chunks)}")
    for index, chunk in enumerate(chunks, 1):
        print(f"[{index}] {chunk.page_content[:80]}...")


def main():
    show_split("小窗口", chunk_size=60, chunk_overlap=10)
    show_split("大窗口", chunk_size=160, chunk_overlap=30)


if __name__ == "__main__":
    main()
