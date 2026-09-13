# -*- coding: utf-8 -*-
"""c02_05：编译图并输出 Mermaid 结构；可选生成 PNG。"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from common import banner, parser_for


class State(TypedDict):
    value: int


def increment_node(state: State) -> dict[str, int]:
    return {"value": state["value"] + 1}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("increment", increment_node)
    graph.add_edge(START, "increment")
    graph.add_edge("increment", END)
    return graph.compile()


def main() -> None:
    parser = parser_for(__doc__ or "Compile and visualize")
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()

    banner("c02_05 编译与可视化")
    graph = build_graph()
    mermaid = graph.get_graph().draw_mermaid()
    print(mermaid)
    print("\n执行结果：", graph.invoke({"value": 41}))

    if args.png:
        output = Path(__file__).with_name("c02_05_graph.png")
        try:
            output.write_bytes(graph.get_graph().draw_mermaid_png())
            print(f"PNG 已写入：{output}")
        except Exception as exc:
            print(f"PNG 生成失败（不影响 Mermaid）：{exc}")


if __name__ == "__main__":
    main()

