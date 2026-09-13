# -*- coding: utf-8 -*-
"""c04_03：Task 重试、缓存策略和错误后的恢复。"""

from __future__ import annotations

from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy

from common import banner, parser_for, print_json


_attempts = {"count": 0}


@task(
    retry_policy=RetryPolicy(
        max_attempts=2,
        retry_on=ValueError,
        jitter=False,
        initial_interval=0.001,
    )
)
def flaky_double(value: int) -> int:
    _attempts["count"] += 1
    if _attempts["count"] == 1:
        raise ValueError("离线模拟的第一次业务失败")
    return value * 2


@entrypoint()
def retry_workflow(value: int) -> dict[str, int]:
    doubled = flaky_double(value).result()
    return {"value": doubled, "attempts": _attempts["count"]}


def run_retry_workflow(value: int) -> dict[str, int]:
    _attempts["count"] = 0
    return retry_workflow.invoke(value)


def main() -> None:
    parser = parser_for(__doc__ or "Retry and resume")
    parser.add_argument("--value", type=int, default=7)
    args = parser.parse_args()

    banner("c04_03 Functional API 重试与恢复")
    print_json(run_retry_workflow(args.value))
    print("\n知识点：失败任务可按 RetryPolicy 重试；副作用必须保持幂等。")


if __name__ == "__main__":
    main()

