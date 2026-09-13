"""Agentic RAG 示例使用的本地知识库。"""

from __future__ import annotations


DOCUMENTS = [
    {
        "id": "langgraph-state",
        "text": (
            "LangGraph StateGraph uses state, nodes and edges. "
            "Reducers control how state updates are merged."
        ),
    },
    {
        "id": "langgraph-persistence",
        "text": (
            "Checkpointers persist graph state by thread_id. "
            "Stores keep long-term memory across threads."
        ),
    },
    {
        "id": "langgraph-hitl",
        "text": (
            "Interrupts pause graph execution for human review. "
            "Command(resume=...) continues a checkpointed run."
        ),
    },
]


def load_documents() -> list[dict[str, str]]:
    return [dict(document) for document in DOCUMENTS]

