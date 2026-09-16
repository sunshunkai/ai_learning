# -*- coding: utf-8 -*-
"""13 DashScope Rerank：先召回候选，再按相关性重新排序。"""

import argparse
import os

from common import load_documents
from factory import RAGConfig, RAGFactory, rerank_documents


def build_dashscope_scorer(model: str):
    def score(query: str, documents: list[str]) -> list[tuple[int, float]]:
        import dashscope

        try:
            response = dashscope.TextReRank.call(
                model=model,
                query=query,
                documents=documents,
                return_documents=False,
                top_n=len(documents),
                api_key=os.environ.get("DASHSCOPE_API_KEY"),
            )
        except Exception as exc:
            raise RuntimeError(f"DashScope Rerank 请求异常: {exc}") from exc
        if response.status_code != 200:
            raise RuntimeError(
                "DashScope Rerank 调用失败: "
                f"code={response.code}, message={response.message}, "
                f"request_id={response.request_id}"
            )
        return [
            (result.index, result.relevance_score)
            for result in response.output.results
        ]

    return score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="DeepSeek 有 embedding API 吗？")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--embedding-provider",
        choices=["hash", "bge-local", "dashscope"],
        default="dashscope",
    )
    parser.add_argument(
        "--vectorstore",
        choices=["memory", "chroma"],
        default="memory",
    )
    parser.add_argument("--persist-directory", default="./chroma_rerank_demo")
    parser.add_argument("--collection-name", default="rag-rerank-demo")
    parser.add_argument(
        "--model",
        default=os.environ.get("DASHSCOPE_RERANK_MODEL", "gte-rerank-v2"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.candidate_k <= 0 or args.top_k <= 0:
        raise ValueError("candidate-k 和 top-k 必须大于 0")
    if args.top_k > args.candidate_k:
        raise ValueError("top-k 不能大于 candidate-k")

    vectorstore_kwargs = {}
    if args.vectorstore == "chroma":
        vectorstore_kwargs = {
            "persist_directory": args.persist_directory,
            "collection_name": args.collection_name,
        }

    config = RAGConfig(
        embedding_provider=args.embedding_provider,
        vectorstore_provider=args.vectorstore,
        top_k=args.candidate_k,
        chunk_size=60,
        chunk_overlap=10,
        vectorstore_kwargs=vectorstore_kwargs,
    )

    try:
        factory = RAGFactory(config, load_documents())
        chunks, _, _, retriever = factory.create_components()
    except (RuntimeError, ImportError) as exc:
        print(exc)
        if args.embedding_provider == "dashscope":
            print("请先设置 DASHSCOPE_API_KEY。")
        if args.vectorstore == "chroma":
            print("使用 Chroma 前请安装: pip install chromadb")
        return

    candidates = retriever.invoke(args.question)
    print(
        f"chunks={len(chunks)}, "
        f"embedding={args.embedding_provider}, "
        f"vectorstore={args.vectorstore}, "
        f"candidates={len(candidates)}"
    )

    print("\n=== 向量召回（重排前） ===")
    for rank, document in enumerate(candidates, 1):
        source = document.metadata.get("source", "unknown")
        print(f"[{rank}] source={source} text={document.page_content[:60]}...")

    scorer = build_dashscope_scorer(args.model)
    try:
        ranked = rerank_documents(
            query=args.question,
            documents=candidates,
            scorer=scorer,
            top_k=args.top_k,
        )
    except RuntimeError as exc:
        print(exc)
        return

    print(f"\n=== DashScope Rerank（{args.model}） ===")
    for rank, item in enumerate(ranked, 1):
        source = item.document.metadata.get("source", "unknown")
        print(
            f"[{rank}] rerank={item.score:.6f} "
            f"original_rank={item.original_rank} "
            f"source={source} "
            f"text={item.document.page_content[:60]}..."
        )


if __name__ == "__main__":
    main()
