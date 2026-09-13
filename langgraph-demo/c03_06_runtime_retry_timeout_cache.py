# -*- coding: utf-8 -*-
"""c03_06：Runtime Context、RetryPolicy、Timeout 和 CachePolicy。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TypedDict

from langgraph.cache.memory import InMemoryCache
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import CachePolicy, RetryPolicy

from common import banner, parser_for, print_json


@dataclass
class DemoContext:
    multiplier: int = 2


class ReliabilityState(TypedDict):
    value: int
    attempts: int
    result: int


def build_reliability_graph():
    """第一次调用模拟网络失败，重试后成功。"""

    attempts = {"count": 0}

    async def reliable_node(
        state: ReliabilityState,
        runtime: Runtime[DemoContext],
    ) -> dict[str, int]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ConnectionError("离线模拟的第一次网络失败")

        multiplier = runtime.context.multiplier
        return {
            "attempts": attempts["count"],
            "result": state["value"] * multiplier,
        }

    graph = StateGraph(ReliabilityState, context_schema=DemoContext)
    graph.add_node(
        "reliable",
        reliable_node,
        retry_policy=RetryPolicy(
            max_attempts=2,
            retry_on=ConnectionError,
            jitter=False,
            initial_interval=0.001,
        ),
        cache_policy=CachePolicy(ttl=60),
        timeout=1.0,
    )
    graph.add_edge(START, "reliable")
    graph.add_edge("reliable", END)
    return graph.compile(cache=InMemoryCache())


def main() -> None:
    parser_for(__doc__ or "Runtime retry timeout cache")
    banner("c03_06 Runtime、Retry、Timeout 与 Cache")
    result = asyncio.run(
        build_reliability_graph().ainvoke(
            {"value": 21},
            context=DemoContext(multiplier=2),
        )
    )
    print_json(result)
    print("\n知识点：Runtime Context 注入运行配置，RetryPolicy 只重试失败节点。")


if __name__ == "__main__":
    main()
