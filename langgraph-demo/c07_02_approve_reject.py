# -*- coding: utf-8 -*-
"""c07_02：人工批准和拒绝走不同执行分支。"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from common import banner, parser_for, print_json


class ApprovalState(TypedDict):
    action: str
    approved: bool
    result: str


def approval_node(state: ApprovalState) -> dict[str, bool]:
    approved = interrupt(
        {
            "question": f"是否批准执行：{state['action']}？",
            "risk": "演示操作可能修改外部状态",
        }
    )
    return {"approved": bool(approved)}


def approved_node(state: ApprovalState) -> dict[str, str]:
    return {"result": f"approved:{state['action']}"}


def rejected_node(state: ApprovalState) -> dict[str, str]:
    return {"result": f"rejected:{state['action']}"}


def approval_route(state: ApprovalState) -> str:
    return "approved" if state["approved"] else "rejected"


def build_approval_graph(checkpointer=None):
    graph = StateGraph(ApprovalState)
    graph.add_node("approval", approval_node)
    graph.add_node("approved", approved_node)
    graph.add_node("rejected", rejected_node)
    graph.add_edge(START, "approval")
    graph.add_conditional_edges(
        "approval",
        approval_route,
        {"approved": "approved", "rejected": "rejected"},
    )
    graph.add_edge("approved", END)
    graph.add_edge("rejected", END)
    return graph.compile(checkpointer=checkpointer)


def _run(decision: bool) -> ApprovalState:
    graph = build_approval_graph(InMemorySaver())
    thread = "approved" if decision else "rejected"
    config = {"configurable": {"thread_id": thread}}
    graph.invoke({"action": "部署新版本"}, config=config)
    return graph.invoke(Command(resume=decision), config=config)


def main() -> None:
    parser = parser_for(__doc__ or "Approve or reject")
    parser.add_argument(
        "--decision",
        choices=["approve", "reject"],
        default="approve",
    )
    args = parser.parse_args()

    banner("c07_02 人工批准与拒绝")
    result = _run(args.decision == "approve")
    print_json(result)
    print("\n知识点：中断后的恢复值可以驱动条件边，形成可审计的 HITL 流程。")


if __name__ == "__main__":
    main()

