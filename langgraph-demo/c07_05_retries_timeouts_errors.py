# -*- coding: utf-8 -*-
"""c07_05：RetryPolicy、Timeout 和 NodeError 错误处理。"""

from __future__ import annotations

from typing import TypedDict

from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from common import banner, parser_for, print_json


class RetryState(TypedDict):
    value: int
    attempts: int


class ErrorState(TypedDict):
    status: str


def build_retry_graph():
    attempts = {"count": 0}

    def flaky_node(state: RetryState) -> dict[str, int]:
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise ConnectionError("模拟第一次连接失败")
        return {
            "value": state["value"] * 2,
            "attempts": attempts["count"],
        }

    graph = StateGraph(RetryState)
    graph.add_node(
        "flaky",
        flaky_node,
        retry_policy=RetryPolicy(
            max_attempts=2,
            retry_on=ConnectionError,
            jitter=False,
            initial_interval=0.001,
        ),
    )
    graph.add_edge(START, "flaky")
    graph.add_edge("flaky", END)
    return graph.compile()


def build_error_handler_graph():
    def risky_node(state: ErrorState) -> dict[str, str]:
        raise RuntimeError("模拟节点失败")

    def recover_node(state: ErrorState, error: NodeError) -> dict[str, str]:
        return {"status": "recovered"}

    graph = StateGraph(ErrorState)
    graph.add_node("risky", risky_node, error_handler=recover_node)
    graph.add_edge(START, "risky")
    graph.add_edge("risky", END)
    return graph.compile()


def main() -> None:
    parser_for(__doc__ or "Retries timeouts errors")
    banner("c07_05 Retry 与 NodeError")

    retry_result = build_retry_graph().invoke({"value": 3})
    handler_result = build_error_handler_graph().invoke(
        {"status": "running"}
    )
    print_json(
        {
            "retry": retry_result,
            "error_handler": handler_result,
        }
    )
    print("知识点：Timeout 仅支持 async 节点；错误处理器在重试耗尽后运行。")


if __name__ == "__main__":
    main()
