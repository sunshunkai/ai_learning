from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAG_ROOT = PROJECT_ROOT / "rag-learning"


def load_rag_factory():
    original_common = sys.modules.pop("common", None)
    original_factory = sys.modules.pop("factory", None)
    sys.path.insert(0, str(RAG_ROOT))
    try:
        import factory as rag_factory  # noqa: PLC0415

        return rag_factory
    finally:
        sys.path.pop(0)
        rag_common = sys.modules.get("common")
        if rag_common is not None and Path(rag_common.__file__).resolve().parent == RAG_ROOT:
            sys.modules.pop("common", None)
        if original_common is not None:
            sys.modules["common"] = original_common
        sys.modules.pop("factory", None)
        if original_factory is not None:
            sys.modules["factory"] = original_factory


rag_factory = load_rag_factory()
build_chunk_id = rag_factory.build_chunk_id
rerank_documents = rag_factory.rerank_documents

from langchain_core.documents import Document  # noqa: E402


class RerankDocumentsTests(unittest.TestCase):
    def test_orders_by_score_and_keeps_original_rank(self) -> None:
        documents = [
            Document(page_content="first", metadata={"source": "a.md"}),
            Document(page_content="second", metadata={"source": "b.md"}),
            Document(page_content="third", metadata={"source": "c.md"}),
        ]

        def scorer(query: str, texts: list[str]) -> list[tuple[int, float]]:
            self.assertEqual(query, "question")
            self.assertEqual(texts, ["first", "second", "third"])
            return [(0, 0.2), (1, 0.9), (2, 0.5)]

        results = rerank_documents(
            query="question",
            documents=documents,
            scorer=scorer,
            top_k=2,
        )

        self.assertEqual([item.document.page_content for item in results], ["second", "third"])
        self.assertEqual([item.score for item in results], [0.9, 0.5])
        self.assertEqual([item.original_rank for item in results], [2, 3])

    def test_empty_candidates_return_empty_list(self) -> None:
        def scorer(query: str, texts: list[str]) -> list[tuple[int, float]]:
            raise AssertionError("scorer should not be called")

        results = rerank_documents(
            query="question",
            documents=[],
            scorer=scorer,
            top_k=3,
        )

        self.assertEqual(results, [])

    def test_rejects_out_of_range_index(self) -> None:
        documents = [Document(page_content="only")]

        def scorer(query: str, texts: list[str]) -> list[tuple[int, float]]:
            return [(1, 0.9)]

        with self.assertRaises(ValueError):
            rerank_documents(
                query="question",
                documents=documents,
                scorer=scorer,
                top_k=1,
            )


class BuildChunkIdTests(unittest.TestCase):
    def test_builds_stable_versioned_id(self) -> None:
        self.assertEqual(
            build_chunk_id("policy.md", version=2, chunk_index=3),
            "policy.md:v2:c3",
        )

    def test_rejects_invalid_version_or_chunk_index(self) -> None:
        with self.assertRaises(ValueError):
            build_chunk_id("policy.md", version=0, chunk_index=0)
        with self.assertRaises(ValueError):
            build_chunk_id("policy.md", version=1, chunk_index=-1)


if __name__ == "__main__":
    unittest.main()
