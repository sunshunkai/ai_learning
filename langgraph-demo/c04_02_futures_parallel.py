# -*- coding: utf-8 -*-
"""c04_02：启动多个 Future 并聚合结果。"""

from __future__ import annotations

from langgraph.func import entrypoint, task

from common import banner, parser_for, print_json


@task
def square(value: int) -> int:
    return value * value


@task
def double(value: int) -> int:
    return value * 2


@entrypoint()
def parallel_square_workflow(value: int) -> int:
    square_future = square(value)
    double_future = double(value)

    # 先启动两个任务，再读取结果，使两个 Future 可以并行推进。
    return square_future.result() + double_future.result()


def main() -> None:
    parser = parser_for(__doc__ or "Futures parallel")
    parser.add_argument("--value", type=int, default=4)
    args = parser.parse_args()

    banner("c04_02 Functional API 并行")
    print_json(parallel_square_workflow.invoke(args.value))
    print("\n知识点：先创建 Future，后调用 result()，可以减少等待时间。")


if __name__ == "__main__":
    main()

