# -*- coding: utf-8 -*-
"""c04_01：Functional API 的 @entrypoint 与 @task。"""

from __future__ import annotations

from langgraph.func import entrypoint, task

from common import banner, parser_for, print_json


@task
def double_task(value: int) -> int:
    return value * 2


@entrypoint()
def double_workflow(value: int) -> int:
    """任务返回 Future，调用 result() 取得可重放结果。"""

    return double_task(value).result()


def main() -> None:
    parser = parser_for(__doc__ or "Entrypoint and task")
    parser.add_argument("--value", type=int, default=21)
    args = parser.parse_args()

    banner("c04_01 Functional API 基础")
    print_json({"input": args.value, "output": double_workflow.invoke(args.value)})
    print("\n知识点：@entrypoint 是工作流边界，@task 是可持久化和重试的执行单元。")


if __name__ == "__main__":
    main()

