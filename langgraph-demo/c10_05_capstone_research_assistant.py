# -*- coding: utf-8 -*-
"""c10_05：综合研究助手。

整合结构化路由、工具调用、人工审批、Checkpoint 和最终生成。
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel

from common import banner, build_model, parser_for, print_json


class RouteDecision(BaseModel):
    route: Literal["research", "general"]


class CapstoneState(TypedDict):
    question: str
    route: Literal["research", "general"]
    tool_result: str
    approved: bool
    answer: str


@tool
def research_topic(topic: str) -> str:
    """返回主题的本地演示资料。"""

    return (
        f"{topic}：LangGraph 通过状态、节点、边、持久化和 HITL "
        "构建可控的有状态工作流。"
    )


def build_capstone_graph(model, checkpointer=None):
    selector = model.with_structured_output(RouteDecision)

    def route_node(state: CapstoneState) -> dict[str, str]:
        decision = selector.invoke(
            [
                SystemMessage(content="需要资料时选择 research，否则 general。"),
                HumanMessage(content=state["question"]),
            ]
        )
        return {"route": decision.route}

    def research_node(state: CapstoneState) -> dict[str, str]:
        return {
            "tool_result": research_topic.invoke(
                {"topic": state["question"]}
            )
        }

    def approval_node(state: CapstoneState) -> dict[str, bool]:
        approved = interrupt(
            {
                "question": "是否批准将研究资料用于生成回答？",
                "tool_result": state["tool_result"],
            }
        )
        return {"approved": bool(approved)}

    def answer_node(state: CapstoneState) -> dict[str, str]:
        if not state["approved"]:
            return {"answer": "用户拒绝使用研究资料。"}
        response = model.invoke(
            [
                SystemMessage(content="根据资料生成简洁答案。"),
                HumanMessage(
                    content=f"问题：{state['question']}\n资料：{state['tool_result']}"
                ),
            ]
        )
        return {"answer": str(response.content)}

    graph = StateGraph(CapstoneState)
    graph.add_node("route", route_node)
    graph.add_node("research", research_node)
    graph.add_node("approval", approval_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "route")
    graph.add_edge("route", "research")
    graph.add_edge("research", "approval")
    graph.add_edge("approval", "answer")
    graph.add_edge("answer", END)
    return graph.compile(checkpointer=checkpointer)


def main() -> None:
    parser = parser_for(__doc__ or "Capstone research assistant")
    parser.add_argument("--question", default="介绍 LangGraph")
    args = parser.parse_args()

    model = build_model(
        offline=args.offline,
        model=args.model,
        structured_responses={
            RouteDecision: RouteDecision(route="research"),
        },
        text_responses=[
            "LangGraph 是用于构建可控、有状态 Agent 工作流的编排运行时。"
        ],
    )

    banner("c10_05 综合项目：研究助手")
    graph = build_capstone_graph(model, InMemorySaver())
    config = {"configurable": {"thread_id": "capstone"}}
    paused = graph.invoke({"question": args.question}, config=config)
    print_json({"paused": paused.get("__interrupt__")})

    from langgraph.types import Command

    final = graph.invoke(Command(resume=True), config=config)
    print_json(final)
    print("\n知识点：综合项目把前面章节的能力组合成一条可恢复业务流程。")


if __name__ == "__main__":
    main()

