# -*- coding: utf-8 -*-
"""c01_02：最小 StateGraph。

LangGraph 的核心执行单元是节点，状态在节点之间流动，边决定下一个节点。
这个例子不调用模型，先建立最小的 State -> Node -> Edge -> Compile 认识。
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from common import banner, parser_for, print_json


class HelloState(TypedDict):
    name: str
    message: str


def greet_node(state: HelloState) -> dict[str, str]:
    """按照输入名称生成消息，体现节点只返回状态增量。"""

    return {"message": f"hello {state['name']}"}


def build_hello_graph():
    """构建并编译最小图。"""

    graph = StateGraph(HelloState)
    graph.add_node("greet", greet_node)
    graph.add_edge(START, "greet")
    graph.add_edge("greet", END)
    return graph.compile()


def run_hello_graph(name: str) -> HelloState:
    """供命令行和测试复用的执行入口。"""

    return build_hello_graph().invoke({"name": name})


def main() -> None:
    parser = parser_for(__doc__ or "Hello StateGraph")
    parser.add_argument("--name", default="LangGraph")
    args = parser.parse_args()

    banner("c01_02 最小 StateGraph")
    result = run_hello_graph(args.name)
    print_json(result)

    print("\n执行路径：START -> greet -> END")
    print("知识点：State 是共享数据，节点返回增量，图编译后通过 invoke 执行。")


if __name__ == "__main__":
    main()

