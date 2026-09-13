# -*- coding: utf-8 -*-
"""c09_04：Supervisor、Handoff、Agent-as-Tool 和并行 Agent。"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

from common import banner, build_model, parser_for, print_json


class WorkerChoice(BaseModel):
    worker: Literal["researcher", "writer"]


class SupervisorState(TypedDict):
    topic: str
    worker: Literal["researcher", "writer"]
    result: str


def build_supervisor_graph(model):
    selector = model.with_structured_output(
        WorkerChoice,
        method="function_calling",
    )

    def supervisor_node(state: SupervisorState) -> dict[str, str]:
        choice = selector.invoke(
            [
                SystemMessage(
                    content="把研究任务交给 researcher，把写作任务交给 writer。"
                ),
                HumanMessage(content=state["topic"]),
            ]
        )
        return {"worker": choice.worker}

    def researcher_node(state: SupervisorState) -> dict[str, str]:
        return {"result": f"研究结果：{state['topic']}"}

    def writer_node(state: SupervisorState) -> dict[str, str]:
        return {"result": f"写作结果：{state['topic']}"}

    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["worker"],
        {"researcher": "researcher", "writer": "writer"},
    )
    graph.add_edge("researcher", END)
    graph.add_edge("writer", END)
    return graph.compile()


def build_handoff_graph():
    class HandoffState(TypedDict):
        topic: str
        research: str
        article: str

    def researcher_node(state: HandoffState) -> dict[str, str]:
        return {"research": f"{state['topic']} 的资料摘要"}

    def writer_node(state: HandoffState) -> dict[str, str]:
        return {"article": f"根据资料完成：{state['research']}"}

    graph = StateGraph(HandoffState)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", END)
    return graph.compile()


@tool
def research_tool(topic: str) -> str:
    """返回指定主题的演示研究结果。"""

    return f"{topic} 的研究摘要"


def build_agent_as_tool_graph(model):
    model_with_tools = model.bind_tools([research_tool])

    def agent_node(state: MessagesState) -> dict:
        response = model_with_tools.invoke(
            [
                SystemMessage(content="需要资料时调用 research_tool。"),
                *state["messages"],
            ]
        )
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode([research_tool]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


def build_parallel_agents_graph():
    class ParallelState(TypedDict):
        topic: str
        outputs: Annotated[list[str], operator.add]

    def researcher_node(state: ParallelState) -> dict[str, list[str]]:
        return {"outputs": [f"研究:{state['topic']}"]}

    def critic_node(state: ParallelState) -> dict[str, list[str]]:
        return {"outputs": [f"评审:{state['topic']}"]}

    graph = StateGraph(ParallelState)
    graph.add_node("researcher", researcher_node)
    graph.add_node("critic", critic_node)
    graph.add_edge(START, "researcher")
    graph.add_edge(START, "critic")
    graph.add_edge("researcher", END)
    graph.add_edge("critic", END)
    return graph.compile()


def main() -> None:
    parser = parser_for(__doc__ or "Multi-agent patterns")
    parser.add_argument(
        "--pattern",
        choices=["supervisor", "handoff", "agent-tool", "parallel"],
        default="supervisor",
    )
    parser.add_argument("--topic", default="LangGraph")
    args = parser.parse_args()

    banner(f"c09_04 多 Agent 模式：{args.pattern}")
    if args.pattern == "supervisor":
        model = build_model(
            offline=args.offline,
            model=args.model,
            structured_responses={
                WorkerChoice: WorkerChoice(worker="researcher"),
            },
        )
        result = build_supervisor_graph(model).invoke({"topic": args.topic})
    elif args.pattern == "handoff":
        result = build_handoff_graph().invoke({"topic": args.topic})
    elif args.pattern == "agent-tool":
        model = build_model(
            offline=args.offline,
            model=args.model,
            tool_responses=[
                {"name": "research_tool", "args": {"topic": args.topic}}
            ],
            text_responses=["已完成研究工具调用。"],
        )
        result = build_agent_as_tool_graph(model).invoke(
            {"messages": [{"role": "user", "content": args.topic}]}
        )
    else:
        result = build_parallel_agents_graph().invoke(
            {"topic": args.topic, "outputs": []}
        )
    print_json(result)
    print("\n知识点：只有状态隔离和并行专业化真正需要时，才引入多 Agent。")


if __name__ == "__main__":
    main()
