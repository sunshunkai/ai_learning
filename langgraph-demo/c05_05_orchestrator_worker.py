# -*- coding: utf-8 -*-
"""c05_05：Orchestrator-Worker 动态拆分和并行处理。"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from common import banner, build_model, parser_for, print_json


class Plan(BaseModel):
    subtasks: list[str] = Field(description="需要并行处理的子任务")


class OrchestratorState(TypedDict):
    topic: str
    subtasks: Annotated[list[str], operator.add]
    drafts: Annotated[list[str], operator.add]


class WorkerInput(TypedDict):
    topic: str
    subtask: str


def build_orchestrator_graph(planner_model, worker_model):
    planner = planner_model.with_structured_output(Plan)

    def plan_node(state: OrchestratorState) -> dict[str, list[str]]:
        plan = planner.invoke(
            [
                SystemMessage(content="把主题拆成 2 到 4 个独立子任务。"),
                HumanMessage(content=state["topic"]),
            ]
        )
        return {"subtasks": plan.subtasks}

    def fan_out(state: OrchestratorState) -> list[Send]:
        return [
            Send(
                "worker",
                {
                    "topic": state["topic"],
                    "subtask": subtask,
                },
            )
            for subtask in state["subtasks"]
        ]

    def worker_node(state: WorkerInput) -> dict[str, list[str]]:
        response = worker_model.invoke(
            [
                SystemMessage(content="完成指定子任务，输出简短草稿。"),
                HumanMessage(
                    content=f"主题：{state['topic']}\n子任务：{state['subtask']}"
                ),
            ]
        )
        return {"drafts": [str(response.content)]}

    graph = StateGraph(OrchestratorState)
    graph.add_node("plan", plan_node)
    graph.add_node("worker", worker_node)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", fan_out, ["worker"])
    graph.add_edge("worker", END)
    return graph.compile()


def main() -> None:
    parser = parser_for(__doc__ or "Orchestrator worker")
    parser.add_argument("--topic", default="LangGraph 快速入门报告")
    args = parser.parse_args()

    planner = build_model(
        offline=args.offline,
        model=args.model,
        structured_responses={
            Plan: Plan(subtasks=["核心概念", "示例结构", "常见误区"]),
        },
    )
    worker = build_model(
        offline=args.offline,
        model=args.model,
        text_responses=["概念草稿", "示例草稿", "误区草稿"],
    )

    banner("c05_05 Orchestrator-Worker")
    result = build_orchestrator_graph(planner, worker).invoke(
        {"topic": args.topic, "subtasks": [], "drafts": []}
    )
    print_json(result)
    print("\n知识点：Orchestrator 动态计划，Worker 通过 Send 并行执行。")


if __name__ == "__main__":
    main()

