# -*- coding: utf-8 -*-
"""c10_03：只读 SQL Agent 与查询执行前人工审核。"""

from __future__ import annotations

import sqlite3
import tempfile
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from common import banner, parser_for, print_json
from fixtures.sqlite_schema import create_database


class SqlState(TypedDict):
    question: str
    sql: str
    approved: bool
    rows: list[dict]
    error: str


def validate_readonly_sql(sql: str) -> str:
    normalized = sql.strip().rstrip(";")
    if not normalized.lower().startswith("select "):
        raise ValueError("只允许执行 SELECT 查询")
    forbidden = ("insert ", "update ", "delete ", "drop ", "alter ")
    if any(word in normalized.lower() for word in forbidden):
        raise ValueError("查询包含写操作关键字")
    return normalized


def plan_sql(question: str) -> str:
    if "客户" in question or "订单" in question:
        return "SELECT customer, SUM(amount) AS total FROM orders GROUP BY customer"
    return "SELECT * FROM orders"


def build_sql_graph(database_path, checkpointer=None):
    def plan_node(state: SqlState) -> dict[str, str]:
        return {"sql": plan_sql(state["question"])}

    def review_node(state: SqlState) -> dict[str, bool]:
        approved = interrupt(
            {
                "question": "是否批准执行该只读查询？",
                "sql": state["sql"],
            }
        )
        return {"approved": bool(approved)}

    def execute_node(state: SqlState) -> dict[str, list[dict]]:
        sql = validate_readonly_sql(state["sql"])
        with sqlite3.connect(database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = [dict(row) for row in connection.execute(sql)]
        return {"rows": rows}

    def denied_node(state: SqlState) -> dict[str, str]:
        return {"error": "human_rejected"}

    def route(state: SqlState) -> str:
        return "execute" if state["approved"] else "denied"

    graph = StateGraph(SqlState)
    graph.add_node("plan", plan_node)
    graph.add_node("review", review_node)
    graph.add_node("execute", execute_node)
    graph.add_node("denied", denied_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "review")
    graph.add_conditional_edges(
        "review",
        route,
        {"execute": "execute", "denied": "denied"},
    )
    graph.add_edge("execute", END)
    graph.add_edge("denied", END)
    return graph.compile(checkpointer=checkpointer)


def main() -> None:
    parser = parser_for(__doc__ or "SQL agent HITL")
    parser.add_argument("--question", default="各客户订单总额是多少？")
    args = parser.parse_args()

    banner("c10_03 SQL Agent 与人工审核")
    with tempfile.TemporaryDirectory(prefix="langgraph-sql-") as temp_dir:
        database = create_database(f"{temp_dir}/demo.sqlite")
        graph = build_sql_graph(database, InMemorySaver())
        config = {"configurable": {"thread_id": "sql-review"}}
        paused = graph.invoke(
            {"question": args.question},
            config=config,
        )
        print_json({"pending": paused.get("__interrupt__")})
        final = graph.invoke(Command(resume=True), config=config)
        print_json(final)
    print("\n知识点：只读校验和人工审核共同降低模型生成 SQL 的风险。")


if __name__ == "__main__":
    main()

