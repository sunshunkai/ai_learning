# -*- coding: utf-8 -*-
"""
逐个验证 AgentScope 示例。

普通示例直接运行；MCP 通过客户端拉起 STDIO Server；A2A 与 Agent
Service 自动启动服务、等待就绪、运行客户端后关闭。

运行:
    python agentscope-demo/verify_examples.py
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

STANDALONE = [
    "c01_01_env_check.py",
    "c01_02_basic_model.py",
    "c01_03_message_event.py",
    "c02_01_thinking_chat.py",
    "c02_02_tool_calling.py",
    "c02_03_structured_output.py",
    "c02_04_multi_agent_formatter.py",
    "c03_01_basic_agent.py",
    "c03_02_agent_structured_output.py",
    "c03_03_multi_turn.py",
    "c03_04_streaming_events.py",
    "c03_05_state_persistence.py",
    "c03_06_interrupt_agent.py",
    "c04_01_function_tool.py",
    "c04_02_tool_middleware.py",
    "c04_03_builtin_tools.py",
    "c04_04_permission_hitl.py",
    "c04_05_ask_user.py",
    "c04_06_tool_groups.py",
    "c05_01_plan_mode.py",
    "c05_02_custom_middleware.py",
    "c05_03_goal_pipeline.py",
    "c06_01_context_management.py",
    "c06_02_rag.py",
    "c06_03_long_term_memory.py",
    "c07_01_workspace_skill.py",
    "c07_03_mcp_tool.py",
]


def run_script(
    script: str,
    *,
    env: dict[str, str] | None = None,
    args: list[str] | None = None,
    timeout: int = 180,
) -> subprocess.CompletedProcess:
    merged_env = os.environ.copy()
    merged_env.update(env or {})
    return subprocess.run(
        [PYTHON, str(ROOT / script), *(args or [])],
        cwd=ROOT,
        env=merged_env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def wait_for_http(url: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=1.0)
            if response.status_code < 500:
                return
        except Exception:
            pass
        time.sleep(0.25)
    raise TimeoutError(f"服务未在 {timeout:.1f}s 内就绪: {url}")


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def check_result(
    name: str,
    result: subprocess.CompletedProcess | None = None,
    error: Exception | None = None,
) -> bool:
    if error is None and result is not None and result.returncode == 0:
        print(f"[PASS] {name}")
        return True

    print(f"[FAIL] {name}")
    if error is not None:
        print(f"       {type(error).__name__}: {error}")
    if result is not None:
        print(f"       returncode={result.returncode}")
        tail = (result.stdout + "\n" + result.stderr).strip().splitlines()
        for line in tail[-20:]:
            print(f"       {line}")
    return False


def verify_a2a() -> bool:
    process = subprocess.Popen(
        [
            PYTHON,
            str(ROOT / "c07_04_a2a_server.py"),
            "--port",
            "19999",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_for_http("http://127.0.0.1:19999/.well-known/agent-card.json")
        result = run_script(
            "c07_05_a2a_client.py",
            args=["--url", "http://127.0.0.1:19999"],
        )
        return check_result("A2A server + client", result)
    except Exception as exc:
        return check_result("A2A server + client", error=exc)
    finally:
        stop_process(process)


def verify_service() -> bool:
    with tempfile.TemporaryDirectory(prefix="agentscope-service-") as service_dir:
        env = {
            "AGENTSCOPE_DEMO_SERVICE_DIR": service_dir,
        }
        process = subprocess.Popen(
            [
                PYTHON,
                str(ROOT / "c08_01_agent_service.py"),
                "--port",
                "18000",
            ],
            cwd=ROOT,
            env={**os.environ, **env},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_for_http("http://127.0.0.1:18000/health")
            result = run_script(
                "c08_02_service_smoke.py",
                args=[
                    "--url",
                    "http://127.0.0.1:18000",
                    "--user-id",
                    f"verify-{int(time.time())}",
                ],
            )
            return check_result("Agent Service + HTTP smoke", result)
        except Exception as exc:
            return check_result("Agent Service + HTTP smoke", error=exc)
        finally:
            stop_process(process)


def main() -> None:
    results: dict[str, bool] = {}

    for script in STANDALONE:
        env = {}
        if script == "c04_04_permission_hitl.py":
            env["AGENTSCOPE_APPROVAL"] = "yes"
        if script == "c07_06_console_chat.py":
            env["AGENTSCOPE_MESSAGE"] = "用一句话介绍 AgentScope。"
        try:
            result = run_script(script, env=env)
            results[script] = check_result(script, result)
        except Exception as exc:
            results[script] = check_result(script, error=exc)

    results["c07_06_console_chat.py"] = check_result(
        "c07_06_console_chat.py",
        run_script(
            "c07_06_console_chat.py",
            env={"AGENTSCOPE_MESSAGE": "用一句话介绍 AgentScope。"},
        ),
    )
    results["c07_04_a2a_server.py + c07_05_a2a_client.py"] = verify_a2a()
    results["c08_01_agent_service.py + c08_02_service_smoke.py"] = (
        verify_service()
    )

    print("\n=== 验证汇总 ===")
    for name, passed in results.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    failed = [name for name, passed in results.items() if not passed]
    if failed:
        raise SystemExit(f"\n失败项: {', '.join(failed)}")
    print(f"\n全部通过，共 {len(results)} 项。")


if __name__ == "__main__":
    main()
