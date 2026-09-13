# -*- coding: utf-8 -*-
"""c05_02：Prompt Chaining，将复杂任务拆成可检查的顺序步骤。"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from common import banner, build_model, parser_for, print_json


class ChainState(TypedDict):
    topic: str
    outline: str
    draft: str
    final: str


def _invoke_text(model, system: str, prompt: str) -> str:
    response = model.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ]
    )
    return str(response.content)


def build_prompt_chain_graph(model):
    def outline_node(state: ChainState) -> dict[str, str]:
        return {
            "outline": _invoke_text(
                model,
                "你是课程设计助手。",
                f"为 {state['topic']} 生成三点提纲。",
            )
        }

    def draft_node(state: ChainState) -> dict[str, str]:
        return {
            "draft": _invoke_text(
                model,
                "根据提纲写作，不要添加新章节。",
                state["outline"],
            )
        }

    def polish_node(state: ChainState) -> dict[str, str]:
        return {
            "final": _invoke_text(
                model,
                "润色文本，保持事实不变。",
                state["draft"],
            )
        }

    graph = StateGraph(ChainState)
    graph.add_node("outline", outline_node)
    graph.add_node("draft", draft_node)
    graph.add_node("polish", polish_node)
    graph.add_edge(START, "outline")
    graph.add_edge("outline", "draft")
    graph.add_edge("draft", "polish")
    graph.add_edge("polish", END)
    return graph.compile()


def main() -> None:
    parser = parser_for(__doc__ or "Prompt chaining")
    parser.add_argument("--topic", default="LangGraph")
    args = parser.parse_args()

    model = build_model(
        offline=args.offline,
        model=args.model,
        text_responses=[
            "1. 核心概念\n2. 状态图\n3. 典型应用",
            "这是根据提纲生成的初稿。",
            "这是经过润色后的最终文本。",
        ],
    )

    banner("c05_02 Prompt Chaining")
    print_json(build_prompt_chain_graph(model).invoke({"topic": args.topic}))
    print("\n知识点：每一步都有清晰输入输出契约，便于验证和定位错误。")


if __name__ == "__main__":
    main()

