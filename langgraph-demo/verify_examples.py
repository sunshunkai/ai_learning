# -*- coding: utf-8 -*-
"""批量运行 LangGraph 教学示例。

默认离线运行 53 个示例，不访问网络；使用 ``--live`` 时让模型相关示例走
DeepSeek。单个示例失败不会阻止其余示例运行，但最终会返回非零退出码。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent

EXAMPLES = {
    "c01": [
        "c01_01_env_check.py",
        "c01_02_hello_state_graph.py",
        "c01_03_graph_vs_functional.py",
    ],
    "c02": [
        "c02_01_state_schemas.py",
        "c02_02_reducers_messages.py",
        "c02_03_input_output_private_state.py",
        "c02_04_nodes_edges.py",
        "c02_05_compile_visualize.py",
    ],
    "c03": [
        "c03_01_branch_parallel.py",
        "c03_02_conditional_routing.py",
        "c03_03_loops_recursion.py",
        "c03_04_send_map_reduce.py",
        "c03_05_command.py",
        "c03_06_runtime_retry_timeout_cache.py",
    ],
    "c04": [
        "c04_01_entrypoint_task.py",
        "c04_02_futures_parallel.py",
        "c04_03_retry_resume.py",
        "c04_04_graph_functional_interop.py",
    ],
    "c05": [
        "c05_01_structured_tools_warmup.py",
        "c05_02_prompt_chaining.py",
        "c05_03_parallelization.py",
        "c05_04_routing.py",
        "c05_05_orchestrator_worker.py",
        "c05_06_evaluator_optimizer.py",
        "c05_07_react_tool_agent.py",
    ],
    "c06": [
        "c06_01_thread_checkpointer.py",
        "c06_02_state_history_replay.py",
        "c06_03_fork_time_travel.py",
        "c06_04_sqlite_checkpointer.py",
        "c06_05_store_long_term_memory.py",
        "c06_06_memory_management.py",
    ],
    "c07": [
        "c07_01_interrupt_resume.py",
        "c07_02_approve_reject.py",
        "c07_03_review_edit_state.py",
        "c07_04_multiple_tool_interrupts.py",
        "c07_05_retries_timeouts_errors.py",
        "c07_06_graceful_drain.py",
        "c07_07_error_reproduction.py",
    ],
    "c08": [
        "c08_01_stream_values_updates.py",
        "c08_02_stream_messages.py",
        "c08_03_custom_tasks_checkpoints.py",
        "c08_04_multiple_modes_subgraphs.py",
        "c08_05_event_streaming.py",
        "c08_06_langsmith_tracing.py",
    ],
    "c09": [
        "c09_01_subgraph_as_node.py",
        "c09_02_subgraph_in_node.py",
        "c09_03_subgraph_persistence_stream.py",
        "c09_04_multi_agent_patterns.py",
    ],
    "c10": [
        "c10_01_testing_partial_execution.py",
        "c10_02_agentic_rag.py",
        "c10_03_sql_agent_hitl.py",
        "c10_04_application_local_server.py",
        "c10_05_capstone_research_assistant.py",
    ],
}


@dataclass
class Result:
    script: str
    status: str
    output: str


def all_examples() -> list[tuple[str, str]]:
    return [
        (group, script)
        for group, scripts in EXAMPLES.items()
        for script in scripts
    ]


def run_example(
    script: str,
    *,
    offline: bool,
    model: str,
    timeout: int,
) -> Result:
    command = [sys.executable, str(ROOT / script)]
    if offline:
        command.append("--offline")
    else:
        command.extend(["--model", model])

    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return Result(script, "TIMEOUT", str(exc))

    output = (completed.stdout + "\n" + completed.stderr).strip()
    if completed.returncode != 0:
        return Result(script, "FAIL", output)
    if "SKIP：" in completed.stdout:
        return Result(script, "SKIP", output)
    return Result(script, "PASS", output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--offline",
        action="store_true",
        help="使用 FakeModel 离线运行，默认选项。",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="使用 DeepSeek 在线运行。",
    )
    parser.add_argument("--group", choices=EXAMPLES)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [
        (group, script)
        for group, script in all_examples()
        if args.group is None or group == args.group
    ]

    if args.list:
        for group, script in selected:
            print(f"{group}  {script}")
        print(f"\n共 {len(selected)} 个示例")
        return

    offline = not args.live
    mode = "offline" if offline else f"live:{args.model}"
    print(f"运行模式：{mode}")
    results: list[Result] = []
    for group, script in selected:
        result = run_example(
            script,
            offline=offline,
            model=args.model,
            timeout=args.timeout,
        )
        results.append(result)
        print(f"[{result.status}] {group}/{script}")
        if result.status in {"FAIL", "TIMEOUT"}:
            for line in result.output.splitlines()[-30:]:
                print(f"       {line}")

    print("\n=== 验证汇总 ===")
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    for status in ("PASS", "SKIP", "FAIL", "TIMEOUT"):
        if status in counts:
            print(f"{status}: {counts[status]}")

    failed = [
        result.script
        for result in results
        if result.status in {"FAIL", "TIMEOUT"}
    ]
    if failed:
        raise SystemExit(f"失败示例：{', '.join(failed)}")


if __name__ == "__main__":
    main()

