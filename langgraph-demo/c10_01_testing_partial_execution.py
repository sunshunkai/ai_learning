# -*- coding: utf-8 -*-
"""c10_01：节点单元测试和从中间节点继续执行。"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from common import banner, parser_for, print_json


class PipelineState(TypedDict):
    value: int


def step_one(state: PipelineState) -> dict[str, int]:
    return {"value": state["value"] + 1}


def step_two(state: PipelineState) -> dict[str, int]:
    return {"value": state["value"] * 2}


def build_pipeline_graph(checkpointer=None):
    graph = StateGraph(PipelineState)
    graph.add_node("step_one", step_one)
    graph.add_node("step_two", step_two)
    graph.add_edge(START, "step_one")
    graph.add_edge("step_one", "step_two")
    graph.add_edge("step_two", END)
    return graph.compile(checkpointer=checkpointer)


def test_node_contract() -> None:
    assert step_one({"value": 1}) == {"value": 2}
    assert step_two({"value": 3}) == {"value": 6}


def main() -> None:
    parser_for(__doc__ or "Testing partial execution")
    banner("c10_01 测试与局部执行")

    test_node_contract()
    graph = build_pipeline_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "partial-execution"}}
    paused = graph.invoke(
        {"value": 1},
        config=config,
        interrupt_after=["step_one"],
    )
    resumed = graph.invoke(None, config=config)

    print_json({"paused": paused, "resumed": resumed})
    print("\n知识点：先测试节点输入输出，再用局部执行减少调试成本和模型调用。")


if __name__ == "__main__":
    main()

