# -*- coding: utf-8 -*-
"""c06_04：使用 SQLite Checkpointer 跨进程持久化。

SQLite 集成位于可选包 ``langgraph-checkpoint-sqlite``。未安装时本示例输出
SKIP，并给出安装命令，而不是伪造持久化结果。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from common import banner, parser_for, print_json


class State(TypedDict):
    value: int


def increment_node(state: State) -> dict[str, int]:
    return {"value": state["value"] + 1}


def build_graph(checkpointer):
    graph = StateGraph(State)
    graph.add_node("increment", increment_node)
    graph.add_edge(START, "increment")
    graph.add_edge("increment", END)
    return graph.compile(checkpointer=checkpointer)


def main() -> None:
    parser_for(__doc__ or "SQLite checkpointer")
    banner("c06_04 SQLite Checkpointer")

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ModuleNotFoundError:
        print("SKIP：缺少 langgraph-checkpoint-sqlite。")
        print("安装：venv/bin/pip install 'langgraph-checkpoint-sqlite>=3,<4'")
        return

    with tempfile.TemporaryDirectory(prefix="langgraph-sqlite-") as temp_dir:
        database = Path(temp_dir) / "checkpoints.sqlite"
        with SqliteSaver.from_conn_string(str(database)) as saver:
            graph = build_graph(saver)
            config = {"configurable": {"thread_id": "sqlite-demo"}}
            first = graph.invoke({"value": 0}, config=config)
            second = graph.invoke({"value": 10}, config=config)
            state = graph.get_state(config)

        print_json(
            {
                "first": first,
                "second": second,
                "persisted_value": state.values["value"],
                "database_exists": database.exists(),
            }
        )
    print("\n知识点：SQLite 文件让状态在进程退出后继续保留。")


if __name__ == "__main__":
    main()

