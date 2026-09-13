# -*- coding: utf-8 -*-
"""c06_05：Store 命名空间与跨线程长期记忆。"""

from __future__ import annotations

from langgraph.store.memory import InMemoryStore

from common import banner, parser_for, print_json


def local_embed(texts: list[str]) -> list[list[float]]:
    """使用字符统计生成本地确定性向量，避免依赖 DeepSeek Embedding。"""

    vectors: list[list[float]] = []
    for text in texts:
        vectors.append(
            [
                float(len(text)),
                float(sum(ord(char) for char in text) % 97),
                float(text.count("喜欢")),
            ]
        )
    return vectors


def build_memory_store() -> InMemoryStore:
    return InMemoryStore(
        index={
            "embed": local_embed,
            "dims": 3,
            "fields": ["text"],
        }
    )


def main() -> None:
    parser_for(__doc__ or "Long term memory store")
    banner("c06_05 Store 长期记忆")

    store = build_memory_store()
    namespace = ("user", "alice", "memories")
    store.put(namespace, "food", {"text": "喜欢面食，不吃香菜"})
    store.put(namespace, "travel", {"text": "喜欢海边旅行"})

    item = store.get(namespace, "food")
    search = store.search(namespace, query="面食", limit=2)

    print_json(
        {
            "get": item.value if item else None,
            "search_keys": [result.key for result in search],
        }
    )
    print("\n知识点：Store 按命名空间隔离长期记忆，Checkpointer 管线程内状态。")


if __name__ == "__main__":
    main()

