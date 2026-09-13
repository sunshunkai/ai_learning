# -*- coding: utf-8 -*-
"""01 文档加载：认识 Document 和 metadata。"""

from langchain_core.documents import Document

from common import load_documents


def main():
    documents = load_documents()
    for index, doc in enumerate(documents, 1):
        print(f"[{index}] source={doc.metadata.get('source')}")
        print(f"    topic={doc.metadata.get('topic')}")
        print(f"    text={doc.page_content}\n")

    custom_doc = Document(
        page_content="metadata 可以保存来源、页码、作者等信息。",
        metadata={"source": "manual.md", "author": "jingyun"},
    )
    print("手动创建 Document:")
    print(custom_doc)


if __name__ == "__main__":
    main()
