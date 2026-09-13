# -*- coding: utf-8 -*-
"""c10_02：带检索评分、问题重写和回答循环的 Agentic RAG。"""

from __future__ import annotations

import re
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from common import banner, build_model, parser_for, print_json
from fixtures.documents import load_documents


class RagState(TypedDict):
    question: str
    documents: list[dict[str, str]]
    relevant: bool
    attempts: int
    answer: str


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9\u4e00-\u9fff]+", text.lower()))


def retrieve_documents(
    query: str,
    documents: list[dict[str, str]],
    limit: int = 2,
) -> list[dict[str, str]]:
    query_tokens = _tokens(query)
    ranked = sorted(
        documents,
        key=lambda document: len(
            query_tokens & _tokens(document["text"])
        ),
        reverse=True,
    )
    return ranked[:limit]


def build_rag_graph(model, documents: list[dict[str, str]] | None = None):
    corpus = documents or load_documents()

    def retrieve_node(state: RagState) -> dict:
        found = retrieve_documents(state["question"], corpus)
        relevant = bool(_tokens(state["question"]) & _tokens(found[0]["text"]))
        return {"documents": found, "relevant": relevant}

    def rewrite_node(state: RagState) -> dict[str, str | int]:
        response = model.invoke(
            [
                SystemMessage(content="把问题改写为更利于检索的短查询。"),
                HumanMessage(content=state["question"]),
            ]
        )
        return {
            "question": str(response.content),
            "attempts": state.get("attempts", 0) + 1,
        }

    def answer_node(state: RagState) -> dict[str, str]:
        context = "\n".join(
            document["text"] for document in state["documents"]
        )
        response = model.invoke(
            [
                SystemMessage(content="只根据资料回答，资料不足时明确说明。"),
                HumanMessage(content=f"资料：{context}\n问题：{state['question']}"),
            ]
        )
        return {"answer": str(response.content)}

    def route_after_retrieve(state: RagState) -> str:
        if state["relevant"]:
            return "answer"
        if state.get("attempts", 0) >= 2:
            return END
        return "rewrite"

    graph = StateGraph(RagState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"rewrite": "rewrite", "answer": "answer"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("answer", END)
    return graph.compile()


def main() -> None:
    parser = parser_for(__doc__ or "Agentic RAG")
    parser.add_argument("--question", default="state graph reducers")
    args = parser.parse_args()

    model = build_model(
        offline=args.offline,
        model=args.model,
        text_responses=[
            "state graph reducers",
            "LangGraph 通过 State、节点、边和 Reducer 组织有状态流程。",
        ],
    )

    banner("c10_02 Agentic RAG")
    result = build_rag_graph(model).invoke(
        {
            "question": args.question,
            "attempts": 0,
        }
    )
    print_json(
        {
            "documents": [doc["id"] for doc in result["documents"]],
            "relevant": result["relevant"],
            "answer": result.get("answer", ""),
        }
    )
    print("\n知识点：检索评分和问题重写让 RAG 能处理第一次检索失败的情况。")


if __name__ == "__main__":
    main()

