# -*- coding: utf-8 -*-
"""c05_06：Evaluator-Optimizer 生成、评估和有限次优化。"""

from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from common import banner, build_model, parser_for, print_json


class OptimizerState(TypedDict):
    task: str
    draft: str
    feedback: Literal["PASS", "REVISE"]
    rounds: int
    max_rounds: int


def build_evaluator_graph(generator_model, evaluator_model):
    def generate_node(state: OptimizerState) -> dict[str, str | int]:
        previous = (
            f"\n上一版：{state['draft']}\n反馈：{state['feedback']}"
            if state.get("draft")
            else ""
        )
        response = generator_model.invoke(
            [
                SystemMessage(content="根据任务和反馈优化答案。"),
                HumanMessage(content=f"{state['task']}{previous}"),
            ]
        )
        return {
            "draft": str(response.content),
            "rounds": state.get("rounds", 0) + 1,
        }

    def evaluate_node(state: OptimizerState) -> dict[str, str]:
        response = evaluator_model.invoke(
            [
                SystemMessage(content="只回答 PASS 或 REVISE。"),
                HumanMessage(content=state["draft"]),
            ]
        )
        verdict = str(response.content).strip().upper()
        return {"feedback": "PASS" if "PASS" in verdict else "REVISE"}

    def should_continue(state: OptimizerState) -> str:
        if state["feedback"] == "PASS":
            return END
        if state["rounds"] >= state.get("max_rounds", 2):
            return END
        return "generate"

    graph = StateGraph(OptimizerState)
    graph.add_node("generate", generate_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges("evaluate", should_continue)
    return graph.compile()


def main() -> None:
    parser = parser_for(__doc__ or "Evaluator optimizer")
    parser.add_argument("--task", default="用一句话介绍 LangGraph")
    args = parser.parse_args()

    generator = build_model(
        offline=args.offline,
        model=args.model,
        text_responses=["初稿", "优化稿"],
    )
    evaluator = build_model(
        offline=args.offline,
        model=args.model,
        text_responses=["REVISE", "PASS"],
    )

    banner("c05_06 Evaluator-Optimizer")
    result = build_evaluator_graph(generator, evaluator).invoke(
        {
            "task": args.task,
            "draft": "",
            "feedback": "REVISE",
            "rounds": 0,
            "max_rounds": 2,
        }
    )
    print_json(result)
    print("\n知识点：优化循环必须有最大轮数，避免模型无限互评。")


if __name__ == "__main__":
    main()
