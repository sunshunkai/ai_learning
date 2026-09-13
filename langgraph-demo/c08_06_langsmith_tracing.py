# -*- coding: utf-8 -*-
"""c08_06：检查 LangSmith tracing 配置，不强制要求远程服务。"""

from __future__ import annotations

import os
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from common import banner, parser_for, print_json


class TraceState(TypedDict):
    value: int


def tracing_status() -> dict[str, Any]:
    enabled = (
        os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
        or os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    )
    return {
        "enabled": enabled,
        "project": os.environ.get("LANGSMITH_PROJECT", "default"),
        "api_key": (
            "configured" if os.environ.get("LANGSMITH_API_KEY") else "missing"
        ),
    }


def build_trace_graph():
    def trace_node(state: TraceState) -> dict[str, int]:
        return {"value": state["value"] + 1}

    graph = StateGraph(TraceState)
    graph.add_node("trace", trace_node)
    graph.add_edge(START, "trace")
    graph.add_edge("trace", END)
    return graph.compile()


def main() -> None:
    parser_for(__doc__ or "LangSmith tracing")
    banner("c08_06 LangSmith Observability")

    status = tracing_status()
    result = build_trace_graph().invoke({"value": 1})
    print_json({"tracing": status, "graph_result": result})
    if not status["enabled"]:
        print("提示：设置 LANGSMITH_TRACING=true 和 LANGSMITH_API_KEY 后可上传追踪。")
    print("\n知识点：追踪记录节点、模型和工具调用，生产环境必须脱敏。")


if __name__ == "__main__":
    main()

