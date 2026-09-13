# -*- coding: utf-8 -*-
"""c05_07：ToolNode 和条件回边组成最小 ReAct Agent。"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from common import banner, build_model, parser_for, print_json


@tool
def calculate_shipping(weight_kg: float, distance_km: float) -> float:
    """按重量和距离计算演示运费。"""

    return round(weight_kg * 10 + distance_km, 2)


def build_react_agent(model):
    tools = [calculate_shipping]
    model_with_tools = model.bind_tools(tools)

    def agent_node(state: MessagesState) -> dict:
        response = model_with_tools.invoke(
            [
                SystemMessage(
                    content="需要计算运费时调用 calculate_shipping 工具。"
                ),
                *state["messages"],
            ]
        )
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


def main() -> None:
    parser = parser_for(__doc__ or "ReAct tool agent")
    args = parser.parse_args()

    model = build_model(
        offline=args.offline,
        model=args.model,
        tool_responses=[
            {
                "name": "calculate_shipping",
                "args": {"weight_kg": 2, "distance_km": 10},
            }
        ],
        text_responses=["根据工具结果，演示运费为 30 元。"],
    )

    banner("c05_07 ReAct Tool Agent")
    result = build_react_agent(model).invoke(
        {
            "messages": [
                HumanMessage(content="计算 2kg、10km 的运费。")
            ]
        }
    )
    print_json(
        {
            "message_types": [
                message.type for message in result["messages"]
            ],
            "final": result["messages"][-1].content,
        }
    )
    print("\n知识点：Agent 循环必须能回到终点，工具结果以 ToolMessage 回传。")


if __name__ == "__main__":
    main()

