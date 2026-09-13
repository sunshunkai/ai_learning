# -*- coding: utf-8 -*-
"""c05_04：使用结构化输出进行任务路由。"""

from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from common import banner, build_model, parser_for, print_json


class RouteSelection(BaseModel):
    route: Literal["math", "general"]


class RoutingState(TypedDict):
    question: str
    route: Literal["math", "general"]
    result: str


def build_routing_graph(model):
    selector = model.with_structured_output(RouteSelection)

    def classify_node(state: RoutingState) -> dict[str, str]:
        decision = selector.invoke(
            [
                SystemMessage(
                    content="判断问题属于 math 还是 general，只返回结构。"
                ),
                HumanMessage(content=state["question"]),
            ]
        )
        return {"route": decision.route}

    def math_node(state: RoutingState) -> dict[str, str]:
        return {"result": f"数学处理链：{state['question']}"}

    def general_node(state: RoutingState) -> dict[str, str]:
        return {"result": f"通用处理链：{state['question']}"}

    def route_selector(state: RoutingState) -> str:
        return state["route"]

    graph = StateGraph(RoutingState)
    graph.add_node("classify", classify_node)
    graph.add_node("math", math_node)
    graph.add_node("general", general_node)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_selector,
        {"math": "math", "general": "general"},
    )
    graph.add_edge("math", END)
    graph.add_edge("general", END)
    return graph.compile()


def main() -> None:
    parser = parser_for(__doc__ or "Routing")
    parser.add_argument("--question", default="请计算 12 * 8")
    args = parser.parse_args()

    model = build_model(
        offline=args.offline,
        model=args.model,
        structured_responses={
            RouteSelection: RouteSelection(route="math"),
        },
    )

    banner("c05_04 Routing")
    print_json(build_routing_graph(model).invoke({"question": args.question}))
    print("\n知识点：路由只负责分类，具体任务仍由独立处理链完成。")


if __name__ == "__main__":
    main()

