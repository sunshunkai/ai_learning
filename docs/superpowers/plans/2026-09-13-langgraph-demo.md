# LangGraph Demo Learning Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build `langgraph-demo/` as a 53-example Chinese learning module covering LangGraph 1.x with DeepSeek and deterministic offline modes.

**Architecture:** Shared helpers provide environment loading, CLI parsing, DeepSeek/FakeModel construction, and script-friendly output. Ten chapter groups contain focused examples. A verification runner executes examples offline or live by chapter, while unit tests validate state and control-flow contracts.

**Tech Stack:** Python 3.13, LangGraph 1.2.11, LangGraph Checkpoint 4.2.0, LangChain 1.4, LangChain Core 1.6, langchain-openai 1.6, Pydantic 2, unittest, SQLite

---

## File Map

### Shared Infrastructure

- Create: `langgraph-demo/common.py`
- Create: `langgraph-demo/requirements.txt`
- Create: `langgraph-demo/fixtures/__init__.py`
- Create: `langgraph-demo/fixtures/fake_models.py`
- Create: `langgraph-demo/fixtures/documents.py`
- Create: `langgraph-demo/fixtures/sqlite_schema.py`
- Create: `langgraph-demo/verify_examples.py`
- Create: `tests/test_langgraph_demo.py`

### Chapter 1

- Create: `langgraph-demo/c01_01_env_check.py`
- Create: `langgraph-demo/c01_02_hello_state_graph.py`
- Create: `langgraph-demo/c01_03_graph_vs_functional.py`

### Chapter 2

- Create: `langgraph-demo/c02_01_state_schemas.py`
- Create: `langgraph-demo/c02_02_reducers_messages.py`
- Create: `langgraph-demo/c02_03_input_output_private_state.py`
- Create: `langgraph-demo/c02_04_nodes_edges.py`
- Create: `langgraph-demo/c02_05_compile_visualize.py`

### Chapter 3

- Create: `langgraph-demo/c03_01_branch_parallel.py`
- Create: `langgraph-demo/c03_02_conditional_routing.py`
- Create: `langgraph-demo/c03_03_loops_recursion.py`
- Create: `langgraph-demo/c03_04_send_map_reduce.py`
- Create: `langgraph-demo/c03_05_command.py`
- Create: `langgraph-demo/c03_06_runtime_retry_timeout_cache.py`

### Chapter 4

- Create: `langgraph-demo/c04_01_entrypoint_task.py`
- Create: `langgraph-demo/c04_02_futures_parallel.py`
- Create: `langgraph-demo/c04_03_retry_resume.py`
- Create: `langgraph-demo/c04_04_graph_functional_interop.py`

### Chapter 5

- Create: `langgraph-demo/c05_01_structured_tools_warmup.py`
- Create: `langgraph-demo/c05_02_prompt_chaining.py`
- Create: `langgraph-demo/c05_03_parallelization.py`
- Create: `langgraph-demo/c05_04_routing.py`
- Create: `langgraph-demo/c05_05_orchestrator_worker.py`
- Create: `langgraph-demo/c05_06_evaluator_optimizer.py`
- Create: `langgraph-demo/c05_07_react_tool_agent.py`

### Chapter 6

- Create: `langgraph-demo/c06_01_thread_checkpointer.py`
- Create: `langgraph-demo/c06_02_state_history_replay.py`
- Create: `langgraph-demo/c06_03_fork_time_travel.py`
- Create: `langgraph-demo/c06_04_sqlite_checkpointer.py`
- Create: `langgraph-demo/c06_05_store_long_term_memory.py`
- Create: `langgraph-demo/c06_06_memory_management.py`

### Chapter 7

- Create: `langgraph-demo/c07_01_interrupt_resume.py`
- Create: `langgraph-demo/c07_02_approve_reject.py`
- Create: `langgraph-demo/c07_03_review_edit_state.py`
- Create: `langgraph-demo/c07_04_multiple_tool_interrupts.py`
- Create: `langgraph-demo/c07_05_retries_timeouts_errors.py`
- Create: `langgraph-demo/c07_06_graceful_drain.py`
- Create: `langgraph-demo/c07_07_error_reproduction.py`

### Chapter 8

- Create: `langgraph-demo/c08_01_stream_values_updates.py`
- Create: `langgraph-demo/c08_02_stream_messages.py`
- Create: `langgraph-demo/c08_03_custom_tasks_checkpoints.py`
- Create: `langgraph-demo/c08_04_multiple_modes_subgraphs.py`
- Create: `langgraph-demo/c08_05_event_streaming.py`
- Create: `langgraph-demo/c08_06_langsmith_tracing.py`

### Chapter 9

- Create: `langgraph-demo/c09_01_subgraph_as_node.py`
- Create: `langgraph-demo/c09_02_subgraph_in_node.py`
- Create: `langgraph-demo/c09_03_subgraph_persistence_stream.py`
- Create: `langgraph-demo/c09_04_multi_agent_patterns.py`

### Chapter 10

- Create: `langgraph-demo/c10_01_testing_partial_execution.py`
- Create: `langgraph-demo/c10_02_agentic_rag.py`
- Create: `langgraph-demo/c10_03_sql_agent_hitl.py`
- Create: `langgraph-demo/c10_04_application_local_server.py`
- Create: `langgraph-demo/c10_05_capstone_research_assistant.py`

### Documentation and Navigation

- Create: `langgraph-demo/README.md`
- Create: `langgraph-demo/KNOWLEDGE_GUIDE.md`
- Modify: `README.md`

## Task 1: Shared Runtime and Fake Model Contract

**Files:**
- Create: `langgraph-demo/common.py`
- Create: `langgraph-demo/fixtures/__init__.py`
- Create: `langgraph-demo/fixtures/fake_models.py`
- Create: `langgraph-demo/requirements.txt`
- Create: `tests/test_langgraph_demo.py`

- [x] **Step 1: Write failing tests for CLI parsing and fake structured output**

Create `tests/test_langgraph_demo.py` with tests that:

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LANGGRAPH_DEMO_ROOT = PROJECT_ROOT / "langgraph-demo"
if str(LANGGRAPH_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(LANGGRAPH_DEMO_ROOT))

from common import add_common_args, build_model, parser_for  # noqa: E402
from fixtures.fake_models import DeterministicFakeChatModel  # noqa: E402


class Route(BaseModel):
    route: Literal["math", "general"]


class CommonHelperTests(unittest.TestCase):
    def test_common_parser_accepts_offline_model_and_verbose(self) -> None:
        parser = parser_for("demo")
        args = parser.parse_args(
            ["--offline", "--model", "deepseek-reasoner", "--verbose"]
        )
        self.assertTrue(args.offline)
        self.assertEqual(args.model, "deepseek-reasoner")
        self.assertTrue(args.verbose)

    def test_offline_model_uses_structured_override(self) -> None:
        model = build_model(
            offline=True,
            structured_responses={Route: Route(route="math")},
        )
        result = model.with_structured_output(Route).invoke("2+2")
        self.assertEqual(result.route, "math")


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run the tests to verify RED**

Run:

```bash
venv/bin/python -m unittest tests.test_langgraph_demo -v
```

Expected: FAIL because `common.py` and the fixture package do not exist.

- [x] **Step 3: Implement the shared helper and fake model**

`common.py` exports:

```python
def parser_for(description: str) -> argparse.ArgumentParser: ...
def add_common_args(parser: argparse.ArgumentParser) -> None: ...
def build_model(
    *,
    offline: bool,
    model: str = "deepseek-chat",
    temperature: float = 0.0,
    structured_responses: dict[type, object] | None = None,
    tool_responses: list[dict] | None = None,
    text_responses: list[str] | None = None,
    **kwargs,
): ...
def banner(title: str) -> None: ...
def print_json(value: object) -> None: ...
def ensure_project_root_on_path() -> None: ...
```

`fixtures/fake_models.py` provides `DeterministicFakeChatModel` with:

- a queued `text_responses`;
- a queued `tool_responses`;
- a Pydantic structured response map;
- `invoke`, `stream`, `bind_tools`, `with_structured_output`;
- deterministic fallback text when no response is queued.

`requirements.txt` contains compatible minimum versions for:

```text
langgraph>=1.2,<2
langgraph-checkpoint>=4.2,<5
langchain>=1.4,<2
langchain-openai>=1.6,<2
pydantic>=2.7
python-dotenv>=1.0
```

- [x] **Step 4: Run focused tests to verify GREEN**

Run:

```bash
venv/bin/python -m unittest tests.test_langgraph_demo -v
```

Expected: PASS.

## Task 2: Chapters 1 and 2

**Files:**
- Create all `langgraph-demo/c01_*.py`
- Create all `langgraph-demo/c02_*.py`
- Modify: `tests/test_langgraph_demo.py`

- [x] **Step 1: Add failing tests for state and reducer behavior**

Test these exact contracts:

- `c02_01_state_schemas.py` exports `build_typed_graph()` and
  `build_pydantic_graph()`;
- `c02_02_reducers_messages.py` exports `build_message_graph()` and
  `build_counter_graph()`;
- a message graph invoked twice with the same `thread_id` returns two AI/user
  pairs when a checkpointer is supplied;
- a counter graph receiving `{"count": 1}` and then `{"count": 2}` returns 3;
- private-state graph output does not include `draft`.

Run:

```bash
venv/bin/python -m unittest tests.test_langgraph_demo -v
```

Expected: FAIL with missing graph builders.

- [x] **Step 2: Implement chapter 1**

Each script uses `parser_for`, `add_common_args`, prints a Chinese banner, and
contains a `main()` entry point.

- `c01_01_env_check.py` checks versions, key presence without printing the key,
  base URL, model name, and optional offline/live readiness.
- `c01_02_hello_state_graph.py` reproduces the official hello-world graph,
  then prints state before and after execution.
- `c01_03_graph_vs_functional.py` builds equivalent two-step graph and
  functional workflows and prints both results and selection guidance.

- [x] **Step 3: Implement chapter 2**

- `c02_01_state_schemas.py` contrasts TypedDict, Pydantic, and messages state.
- `c02_02_reducers_messages.py` contrasts default overwrite with `add`,
  `operator.add`, and `add_messages`.
- `c02_03_input_output_private_state.py` uses explicit input/output schemas and
  a private node-to-node field.
- `c02_04_nodes_edges.py` demonstrates a normal edge, conditional edge, shared
  node reuse, and `START`/`END`.
- `c02_05_compile_visualize.py` prints Mermaid text and writes PNG only when
  the optional renderer is available; PNG failure is reported, not fatal.

- [x] **Step 4: Run chapter tests and offline scripts**

Run:

```bash
venv/bin/python -m unittest tests.test_langgraph_demo -v
venv/bin/python langgraph-demo/c01_01_env_check.py --offline
venv/bin/python langgraph-demo/c01_02_hello_state_graph.py --offline
venv/bin/python langgraph-demo/c02_02_reducers_messages.py --offline
```

Expected: all commands exit 0.

## Task 3: Chapter 3 Advanced Graph Control Flow

**Files:**
- Create all `langgraph-demo/c03_*.py`
- Modify: `tests/test_langgraph_demo.py`

- [x] **Step 1: Add focused tests**

Test:

- parallel branches merge through a list reducer;
- conditional routing selects the expected node;
- a loop stops at three iterations;
- `Send` processes every requested item;
- `Command` updates state and transitions to the requested node.

Run:

```bash
venv/bin/python -m unittest tests.test_langgraph_demo -v
```

Expected: FAIL on missing builders.

- [x] **Step 2: Implement the six examples**

Expose pure graph builders so tests do not need subprocesses:

```python
def build_branch_parallel_graph(): ...
def build_conditional_graph(): ...
def build_loop_graph(): ...
def build_send_graph(): ...
def build_command_graph(): ...
def build_reliability_graph(): ...
```

The reliability example uses local deterministic functions, a retry policy, a
node timeout, a cache policy, and runtime context. It must not require network
access in offline mode.

- [x] **Step 3: Run focused tests and scripts**

Run:

```bash
venv/bin/python -m unittest tests.test_langgraph_demo -v
venv/bin/python langgraph-demo/c03_01_branch_parallel.py --offline
venv/bin/python langgraph-demo/c03_04_send_map_reduce.py --offline
venv/bin/python langgraph-demo/c03_05_command.py --offline
venv/bin/python langgraph-demo/c03_06_runtime_retry_timeout_cache.py --offline
```

Expected: all commands exit 0.

## Task 4: Chapter 4 Functional API

**Files:**
- Create all `langgraph-demo/c04_*.py`
- Modify: `tests/test_langgraph_demo.py`

- [x] **Step 1: Add focused tests**

Test:

- `@task` returns the expected value;
- two futures are both collected;
- a task with retry succeeds after two simulated failures;
- a functional workflow can invoke a compiled graph and return its state.

Run:

```bash
venv/bin/python -m unittest tests.test_langgraph_demo -v
```

Expected: FAIL.

- [x] **Step 2: Implement the four examples**

Use `langgraph.func.entrypoint` and `langgraph.func.task`. Keep API calls in
module-level functions so tests can import them. Explain determinism,
serialization, idempotency, and side effects in the docstring and
`KNOWLEDGE_GUIDE.md`.

- [x] **Step 3: Run focused tests and scripts**

Run:

```bash
venv/bin/python -m unittest tests.test_langgraph_demo -v
venv/bin/python langgraph-demo/c04_01_entrypoint_task.py --offline
venv/bin/python langgraph-demo/c04_02_futures_parallel.py --offline
venv/bin/python langgraph-demo/c04_04_graph_functional_interop.py --offline
```

Expected: all commands exit 0.

## Task 5: Chapter 5 Workflows and Agents

**Files:**
- Create all `langgraph-demo/c05_*.py`
- Modify: `tests/test_langgraph_demo.py`

- [x] **Step 1: Add focused fake-model tests**

Test:

- prompt chaining calls stages in order;
- parallelization produces one result per branch;
- routing uses the structured route value;
- evaluator-optimizer stops after the configured bound;
- ReAct loop executes a tool and returns a final `AIMessage`.

Run:

```bash
venv/bin/python -m unittest tests.test_langgraph_demo -v
```

Expected: FAIL.

- [x] **Step 2: Implement warm-up and deterministic patterns**

`c05_01` demonstrates structured output and tool calls without a graph.
`c05_02` through `c05_06` implement the five official workflow patterns with
small graph builders and deterministic offline responses.

- [x] **Step 3: Implement the ReAct agent**

`c05_07_react_tool_agent.py` uses `ToolNode`, `tools_condition`, and a
conditional back-edge. The tool performs a safe local calculation such as
shipping-cost calculation. Offline mode scripts one tool call followed by a
final answer.

- [x] **Step 4: Run focused tests and scripts**

Run:

```bash
venv/bin/python -m unittest tests.test_langgraph_demo -v
venv/bin/python langgraph-demo/c05_01_structured_tools_warmup.py --offline
venv/bin/python langgraph-demo/c05_03_parallelization.py --offline
venv/bin/python langgraph-demo/c05_07_react_tool_agent.py --offline
```

Expected: all commands exit 0.

## Task 6: Chapter 6 Persistence, Memory, and Time Travel

**Files:**
- Create all `langgraph-demo/c06_*.py`
- Modify: `tests/test_langgraph_demo.py`

- [x] **Step 1: Add focused tests**

Test:

- separate thread IDs do not share state;
- state history contains multiple checkpoints;
- replay starts at the selected checkpoint;
- forked state changes do not mutate the original history;
- SQLite state is visible after reopening the connection;
- Store namespaces isolate user memories.

Run:

```bash
venv/bin/python -m unittest tests.test_langgraph_demo -v
```

Expected: FAIL.

- [x] **Step 2: Implement memory examples**

Use `InMemorySaver` for `c06_01-c06_03`. Use only the standard library plus
`langgraph-checkpoint` for `c06_04`; if `langgraph-checkpoint-sqlite` is not
installed, print the exact installation command and exit successfully in a
documented `SKIP` mode. The optional dependency is listed in README rather than
the default requirements.

Use `InMemoryStore` for `c06_05` and a deterministic local embedding object.
`c06_06` demonstrates trimming, deletion, and summary-based replacement
without mutating the original checkpoint history.

- [x] **Step 3: Run focused tests and scripts**

Run:

```bash
venv/bin/python -m unittest tests.test_langgraph_demo -v
venv/bin/python langgraph-demo/c06_01_thread_checkpointer.py --offline
venv/bin/python langgraph-demo/c06_02_state_history_replay.py --offline
venv/bin/python langgraph-demo/c06_03_fork_time_travel.py --offline
venv/bin/python langgraph-demo/c06_05_store_long_term_memory.py --offline
venv/bin/python langgraph-demo/c06_06_memory_management.py --offline
```

Expected: all commands exit 0; SQLite script exits 0 with PASS or SKIP.

## Task 7: Chapter 7 HITL, Durability, and Errors

**Files:**
- Create all `langgraph-demo/c07_*.py`
- Modify: `tests/test_langgraph_demo.py`

- [x] **Step 1: Add focused interrupt and error tests**

Test:

- a graph pauses at `interrupt()`;
- `Command(resume=...)` returns the human payload;
- approve and reject branches produce different states;
- state editing changes the resumed value;
- a retry node succeeds after a transient failure;
- timeout and node-error routing produce explicit outcomes.

Run:

```bash
venv/bin/python -m unittest tests.test_langgraph_demo -v
```

Expected: FAIL.

- [x] **Step 2: Implement interrupt examples**

Every interrupt example compiles with `InMemorySaver` and a stable test thread
ID. Print the first pause state, resume command, and final state. Do not wrap
`interrupt()` in `try/except`.

- [x] **Step 3: Implement reliability and error reproduction**

Use deterministic functions for retry and timeout. `c07_07` accepts
`--case recursion|missing-checkpointer|invalid-return|concurrent-update|bad-history|multiple-subgraphs`
and catches the documented exception to print its cause and fix. Each case
must exit 0 after successfully reproducing the expected error.

- [x] **Step 4: Run focused tests and scripts**

Run:

```bash
venv/bin/python -m unittest tests.test_langgraph_demo -v
venv/bin/python langgraph-demo/c07_01_interrupt_resume.py --offline
venv/bin/python langgraph-demo/c07_02_approve_reject.py --offline
venv/bin/python langgraph-demo/c07_05_retries_timeouts_errors.py --offline
for case in recursion missing-checkpointer invalid-return concurrent-update bad-history multiple-subgraphs; do venv/bin/python langgraph-demo/c07_07_error_reproduction.py --case "$case"; done
```

Expected: all commands exit 0.

## Task 8: Chapter 8 Streaming, Events, and Observability

**Files:**
- Create all `langgraph-demo/c08_*.py`
- Modify: `tests/test_langgraph_demo.py`

- [x] **Step 1: Add focused stream tests**

Test:

- `values` yields at least the initial and final states;
- `updates` includes each executed node name;
- `messages` includes an AI token or message;
- multiple modes yield tagged `(mode, payload)` pairs;
- a subgraph stream includes child events.

Run:

```bash
venv/bin/python -m unittest tests.test_langgraph_demo -v
```

Expected: FAIL.

- [x] **Step 2: Implement state, message, custom, and event streaming examples**

Use `stream_mode` and `version="v2"` where supported. Offline model streaming
must produce deterministic chunks. `c08_05` demonstrates event lifecycle and
one small custom projection. `c08_06` prints whether tracing is enabled; it
does not require a LangSmith key.

- [x] **Step 3: Run focused tests and scripts**

Run:

```bash
venv/bin/python -m unittest tests.test_langgraph_demo -v
venv/bin/python langgraph-demo/c08_01_stream_values_updates.py --offline
venv/bin/python langgraph-demo/c08_02_stream_messages.py --offline
venv/bin/python langgraph-demo/c08_03_custom_tasks_checkpoints.py --offline
venv/bin/python langgraph-demo/c08_04_multiple_modes_subgraphs.py --offline
venv/bin/python langgraph-demo/c08_05_event_streaming.py --offline
```

Expected: all commands exit 0.

## Task 9: Chapter 9 Subgraphs and Multi-Agent Systems

**Files:**
- Create all `langgraph-demo/c09_*.py`
- Modify: `tests/test_langgraph_demo.py`

- [x] **Step 1: Add focused tests**

Test:

- a compiled subgraph works as a parent node;
- calling a subgraph inside a node supports state translation;
- child state does not leak private fields;
- supervisor routing selects the expected worker;
- agent-as-tool returns a bounded final response.

Run:

```bash
venv/bin/python -m unittest tests.test_langgraph_demo -v
```

Expected: FAIL.

- [x] **Step 2: Implement subgraph examples**

`c09_01` and `c09_02` compare the two invocation forms. `c09_03` shows
persistence inheritance and streamed child output. `c09_04` uses a CLI
`--pattern {supervisor,handoff,agent-tool,parallel}` switch to run four small
patterns from one teaching script.

- [x] **Step 3: Run focused tests and scripts**

Run:

```bash
venv/bin/python -m unittest tests.test_langgraph_demo -v
venv/bin/python langgraph-demo/c09_01_subgraph_as_node.py --offline
venv/bin/python langgraph-demo/c09_02_subgraph_in_node.py --offline
venv/bin/python langgraph-demo/c09_03_subgraph_persistence_stream.py --offline
venv/bin/python langgraph-demo/c09_04_multi_agent_patterns.py --offline --pattern supervisor
```

Expected: all commands exit 0.

## Task 10: Chapter 10 Testing, Applications, and Capstone

**Files:**
- Create all `langgraph-demo/c10_*.py`
- Create: `langgraph-demo/fixtures/documents.py`
- Create: `langgraph-demo/fixtures/sqlite_schema.py`
- Modify: `tests/test_langgraph_demo.py`

- [x] **Step 1: Add application-level tests**

Test:

- partial execution starts at a selected node and skips upstream work;
- RAG retrieval returns a relevant document id;
- SQL guard rejects write statements;
- capstone offline mode covers routing, tool use, approval, and final answer.

Run:

```bash
venv/bin/python -m unittest tests.test_langgraph_demo -v
```

Expected: FAIL.

- [x] **Step 2: Implement testing and data-agent examples**

`c10_01` tests a node directly and then runs a graph from a selected node.
`c10_02` implements retrieve-grade-rewrite-answer routing with local documents.
`c10_03` uses an in-memory or temporary SQLite database, read-only SQL
validation, and an interrupt before query execution.

- [x] **Step 3: Implement application structure and capstone**

`c10_04` creates a small `app/` layout example in printed form and provides a
network-free FastAPI/Studio setup check. It must not start a server during
offline verification. `c10_05` integrates routing, tool calls, subgraphs, HITL,
checkpointing, and streaming with deterministic offline responses.

- [x] **Step 4: Run focused tests and scripts**

Run:

```bash
venv/bin/python -m unittest tests.test_langgraph_demo -v
venv/bin/python langgraph-demo/c10_01_testing_partial_execution.py --offline
venv/bin/python langgraph-demo/c10_02_agentic_rag.py --offline
venv/bin/python langgraph-demo/c10_03_sql_agent_hitl.py --offline
venv/bin/python langgraph-demo/c10_05_capstone_research_assistant.py --offline
```

Expected: all commands exit 0.

## Task 11: Verification Runner, Documentation, and Navigation

**Files:**
- Create: `langgraph-demo/verify_examples.py`
- Create: `langgraph-demo/README.md`
- Create: `langgraph-demo/KNOWLEDGE_GUIDE.md`
- Modify: `README.md`

- [x] **Step 1: Implement the verification runner**

The runner supports:

```text
--offline
--live
--group c01
--timeout 180
--list
```

It discovers the exact 53 scripts, passes the selected mode to each script,
prints `PASS`, `FAIL`, or documented `SKIP`, captures the final 30 output
lines on failure, and exits non-zero if a required example fails.

- [x] **Step 2: Write the Chinese README**

Include:

- module purpose and architecture;
- DeepSeek and offline setup;
- the ten chapter tables with all 53 commands;
- study order;
- version matrix;
- verification commands;
- DeepSeek limitations;
- safety notes.

- [x] **Step 3: Write the Chinese knowledge guide**

For every example, include:

1. learning goal;
2. official concept;
3. state/data flow;
4. key APIs;
5. expected result;
6. common mistakes;
7. exercises;
8. official documentation links.

- [x] **Step 4: Update the root README**

Add `langgraph-demo/` to the layout, installation commands, learning route,
run examples, and project description.

- [x] **Step 5: Run the complete verification set**

Run:

```bash
venv/bin/python -m compileall langgraph-demo tests
venv/bin/python -m unittest discover -s tests -v
venv/bin/python langgraph-demo/verify_examples.py --offline
venv/bin/python langgraph-demo/verify_examples.py --list
```

Expected: compile succeeds, all tests pass, all required offline examples
pass, and the list contains 53 examples.

## Task 12: DeepSeek Live Smoke Tests

**Files:**
- Modify only if a provider-specific limitation is discovered:
  `langgraph-demo/README.md`

- [x] **Step 1: Check credentials without printing secrets**

Run:

```bash
if rg -q '^DEEPSEEK_API_KEY=.+' .env; then echo configured; else echo missing; fi
```

Expected: `configured` or a documented skip.

- [x] **Step 2: Run bounded live groups**

Run:

```bash
venv/bin/python langgraph-demo/verify_examples.py --live --group c01
venv/bin/python langgraph-demo/verify_examples.py --live --group c05
venv/bin/python langgraph-demo/verify_examples.py --live --group c10
```

Expected: examples pass, or a provider limitation is documented with the
observed error and a working alternative path.

- [x] **Step 3: Re-run the complete offline suite after any live fixes**

Run:

```bash
venv/bin/python -m unittest discover -s tests -v
venv/bin/python langgraph-demo/verify_examples.py --offline
```

Expected: PASS.

## Self-Review Checklist

- [x] Every spec example appears in the exact file map.
- [x] Every model-powered example supports `--offline`.
- [x] No script prints or commits a secret.
- [x] Tests assert state and control flow, not exact LLM prose.
- [x] Optional dependencies produce documented SKIP behavior.
- [x] `verify_examples.py` reports 53 examples.
- [x] Root and module documentation point to official sources.
- [x] Existing LangChain and AgentScope behavior remains unchanged.
