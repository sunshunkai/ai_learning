# LangGraph Demo Learning Module Design

## Goal

Add a standalone `langgraph-demo/` learning module that turns the current
official LangGraph Python documentation and repository examples into a
progressive Chinese-language curriculum. The module must teach both the Graph
API and Functional API, include practical workflow and agent patterns, and
cover persistence, human-in-the-loop, streaming, fault tolerance, subgraphs,
testing, and small end-to-end applications.

All model-powered examples use DeepSeek through its OpenAI-compatible API by
default. Every example that needs a model also supports a deterministic,
network-free fake-model mode so the module can be verified without spending
API credits.

## Baseline

The design is based on:

- `https://docs.langchain.com/oss/python/langgraph/overview`
- `https://docs.langchain.com/oss/python/langgraph/llms.txt`, which lists 43
  LangGraph Python documentation pages
- `https://github.com/langchain-ai/langgraph`

The local implementation baseline is:

- Python 3.13
- LangGraph 1.2.x
- langgraph-checkpoint 4.x
- LangChain 1.x
- `langchain-openai` with `ChatOpenAI`
- DeepSeek's OpenAI-compatible endpoint

The exact installed versions and any DeepSeek capability differences will be
recorded in `langgraph-demo/README.md`.

## Existing Repository Context

The repository uses one directory per learning topic:

- `langchain-demo/` teaches LangChain and already includes introductory
  LangGraph checkpointing and agent examples.
- `agentscope-demo/` provides an example of a chapter-numbered, heavily
  documented DeepSeek learning module.
- `tools/load_env.py` loads the project-level `.env` file.
- Tests live under `tests/`.

The new module is named `langgraph-demo/` so it cannot shadow the installed
`langgraph` package. It will cross-link to `langchain-demo/` where concepts
overlap, but it will not refactor or replace the existing module.

## Approaches Considered

### Documentation-First

Mirror the 43 official pages in order. This maximizes API coverage but
fragments the learning path and provides little end-to-end context.

### Project-First

Build only large agent applications. This is motivating but hides the state,
reducer, control-flow, persistence, and functional-programming foundations
that make LangGraph predictable.

### Hybrid Progressive Curriculum

Start with the execution model, teach the Graph and Functional APIs, apply
them to workflows and agents, then cover production capabilities and finish
with integrated projects. This is the selected approach.

## Architecture

The module has four layers:

1. **Core helpers**: environment loading, DeepSeek model creation, common CLI
   arguments, deterministic fake models, and small output helpers.
2. **Focused examples**: one independently runnable script per concept,
   grouped into ten chapters.
3. **Verification**: offline examples, unit tests for stateful behavior, and
   optional live DeepSeek smoke tests.
4. **Documentation**: a module README, a per-example Chinese knowledge guide,
   official-document links, and learning exercises.

### Proposed File Structure

```text
langgraph-demo/
├── README.md
├── KNOWLEDGE_GUIDE.md
├── common.py
├── requirements.txt
├── verify_examples.py
├── fixtures/
│   ├── __init__.py
│   ├── documents.py
│   ├── fake_models.py
│   └── sqlite_schema.py
├── c01_01_env_check.py
├── ...
└── c10_05_capstone_research_assistant.py
```

The module will not use an `__init__.py` at its top level because existing
learning modules are not imported by their hyphenated directory name.
Chapter scripts are run directly or through `verify_examples.py`.

## Common Helper Design

`common.py` owns only cross-example concerns:

- load the repository `.env`;
- read `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, and `DEEPSEEK_MODEL`;
- default the model to `deepseek-chat`;
- create a `ChatOpenAI` instance for DeepSeek;
- create a deterministic fake chat model for `--offline`;
- add shared `--offline`, `--model`, and `--verbose` CLI arguments;
- provide small helpers for printing chapter banners and structured results;
- keep provider-specific setup out of individual demos.

The fake model supports:

- ordinary `invoke()` and `stream()` responses;
- deterministic Pydantic structured output;
- deterministic tool calls;
- scripted response sequences for multi-step agents.

The fake model is for verification and teaching, not an attempt to simulate
DeepSeek behavior.

## CLI Convention

Every model-powered script accepts:

```bash
venv/bin/python langgraph-demo/<script>.py --offline
venv/bin/python langgraph-demo/<script>.py
venv/bin/python langgraph-demo/<script>.py --model deepseek-chat
```

The default is live DeepSeek. `--offline` must not read credentials or make
network calls.

`verify_examples.py` accepts:

```bash
venv/bin/python langgraph-demo/verify_examples.py --offline
venv/bin/python langgraph-demo/verify_examples.py --live
venv/bin/python langgraph-demo/verify_examples.py --live --group c03
```

The offline suite is required to pass before completion. The live suite is an
integration check and may be run by chapter to control latency and cost.

## Curriculum and Demo Inventory

The initial implementation contains 53 runnable examples.

### c01: Orientation and Mental Model

| Demo | Chinese teaching focus |
| --- | --- |
| `c01_01_env_check.py` | 检查 Python、LangGraph、LangChain、DeepSeek 配置，并解释各包职责 |
| `c01_02_hello_state_graph.py` | 用最小 StateGraph 理解 State、Node、Edge、Compile、Invoke |
| `c01_03_graph_vs_functional.py` | 对比 Graph API 与 Functional API，解释选择依据 |

Knowledge points:

- LangGraph is an orchestration runtime, not a prompt or agent abstraction.
- Deterministic code and LLM decisions can coexist in one workflow.
- `StateGraph`, node functions, edges, and execution state form the core model.
- Selecting an API depends on collaboration, visualization, parallelism,
  shared state, and how much existing Python code must be preserved.

### c02: State and Graph Fundamentals

| Demo | Chinese teaching focus |
| --- | --- |
| `c02_01_state_schemas.py` | TypedDict、Pydantic 和 messages state 的差异 |
| `c02_02_reducers_messages.py` | Reducer 合并规则、并发更新和 `add_messages` |
| `c02_03_input_output_private_state.py` | 输入、输出和节点间私有状态的边界 |
| `c02_04_nodes_edges.py` | 普通边、条件边、入口、出口和节点复用 |
| `c02_05_compile_visualize.py` | 编译图、Mermaid/PNG 可视化和执行路径检查 |

Knowledge points:

- Every update passes through a channel and may use a reducer.
- Default last-value semantics are not suitable for every field.
- Message history needs explicit merge semantics.
- Private state prevents implementation details from leaking into public
  graph input or output.
- Compilation validates the graph shape and prepares a runnable instance.

### c03: Advanced Graph Control Flow

| Demo | Chinese teaching focus |
| --- | --- |
| `c03_01_branch_parallel.py` | 顺序、分支、并行 fan-out/fan-in 和状态聚合 |
| `c03_02_conditional_routing.py` | 基于状态条件的动态路由 |
| `c03_03_loops_recursion.py` | 循环终止、递归限制和剩余步数 |
| `c03_04_send_map_reduce.py` | 使用 `Send` 动态创建 Map-Reduce 任务 |
| `c03_05_command.py` | 用 `Command` 合并状态更新和节点跳转 |
| `c03_06_runtime_retry_timeout_cache.py` | Runtime Context、重试、超时、缓存和节点默认配置 |

Knowledge points:

- State updates from parallel nodes must be compatible with reducers.
- Conditional routing centralizes decision logic outside nodes when useful.
- Loops require explicit termination and recursion protection.
- `Send` is for dynamic fan-out where the number of work items is not known
  when the graph is compiled.
- `Command` is useful for tool-driven routing and combined updates.
- Retry, timeout, and cache policies belong to node execution configuration.

### c04: Functional API

| Demo | Chinese teaching focus |
| --- | --- |
| `c04_01_entrypoint_task.py` | `@entrypoint`、`@task` 和基本函数式工作流 |
| `c04_02_futures_parallel.py` | Future 并行执行与结果聚合 |
| `c04_03_retry_resume.py` | 重试、缓存、超时和错误后的恢复 |
| `c04_04_graph_functional_interop.py` | Functional API 调用 Graph，以及 Graph 调用 Functional API |

Knowledge points:

- `@task` defines the durable, replayable unit; `@entrypoint` defines the
  workflow boundary.
- Workflow code should remain deterministic while side effects stay in tasks.
- Idempotency matters because task execution can be replayed.
- Both APIs share persistence and runtime concepts and can be combined.

### c05: Workflows and Agents

| Demo | Chinese teaching focus |
| --- | --- |
| `c05_01_structured_tools_warmup.py` | DeepSeek 结构化输出、工具调用和消息往返 |
| `c05_02_prompt_chaining.py` | 顺序 Prompt Chaining 与中间校验 |
| `c05_03_parallelization.py` | 并行 LLM 调用、结果合并和失败处理 |
| `c05_04_routing.py` | 根据任务类型路由到不同处理链 |
| `c05_05_orchestrator_worker.py` | Orchestrator 动态拆分任务，Worker 并行处理 |
| `c05_06_evaluator_optimizer.py` | 生成、评估、反馈、重写的闭环 |
| `c05_07_react_tool_agent.py` | ToolNode、条件回边和 ReAct Agent 循环 |

Knowledge points:

- Start with the simplest deterministic workflow that solves the problem.
- Prompt Chaining improves control when each stage has a clear contract.
- Parallelization trades coordination complexity for latency.
- Routing and Orchestrator-Worker solve different scaling problems.
- Evaluator-Optimizer is a bounded feedback loop, not an unlimited retry loop.
- A ReAct agent is a model-tool loop whose termination must be explicit.

### c06: Persistence, Memory, and Time Travel

| Demo | Chinese teaching focus |
| --- | --- |
| `c06_01_thread_checkpointer.py` | thread_id、短期记忆和 InMemorySaver |
| `c06_02_state_history_replay.py` | 获取历史 Checkpoint 并从指定状态重放 |
| `c06_03_fork_time_travel.py` | 修改历史状态、Fork 执行和分支对比 |
| `c06_04_sqlite_checkpointer.py` | SQLite Checkpointer 与跨进程持久化 |
| `c06_05_store_long_term_memory.py` | Store 命名空间、跨线程长期记忆和语义检索 |
| `c06_06_memory_management.py` | 消息裁剪、删除、摘要和 Checkpoint 管理 |

Knowledge points:

- A Checkpointer persists execution state for a thread.
- A Store persists application-level data across threads.
- Replay re-executes from a checkpoint; Fork changes state before replay.
- Long-term memory needs explicit namespace, retrieval, and update policy.
- Unbounded checkpoints and messages become storage and context problems.

### c07: Human-in-the-Loop, Durability, and Errors

| Demo | Chinese teaching focus |
| --- | --- |
| `c07_01_interrupt_resume.py` | `interrupt()` 暂停、检查点和恢复载荷 |
| `c07_02_approve_reject.py` | 工具调用前的人工批准与拒绝路径 |
| `c07_03_review_edit_state.py` | 审核和修改人类或模型提交的状态 |
| `c07_04_multiple_tool_interrupts.py` | 一个执行中处理多个中断和工具审批 |
| `c07_05_retries_timeouts_errors.py` | 自定义重试、Run/Idle Timeout、NodeError 路由 |
| `c07_06_graceful_drain.py` | 服务关闭时的请求排空和可恢复执行 |
| `c07_07_error_reproduction.py` | 复现递归超限、缺少 Checkpointer、非法返回值等错误 |

Knowledge points:

- Interrupts require persistence for the intended production workflow.
- Human input can approve, reject, edit, or supply missing data.
- Side effects before an interrupt must be idempotent.
- Retry policy, timeout, and error routing are distinct reliability controls.
- Error messages become part of the curriculum, not hidden implementation
  failures.

### c08: Streaming, Events, and Observability

| Demo | Chinese teaching focus |
| --- | --- |
| `c08_01_stream_values_updates.py` | `values` 与 `updates` 两种状态流 |
| `c08_02_stream_messages.py` | LLM Token、消息元数据和 V2 StreamPart |
| `c08_03_custom_tasks_checkpoints.py` | `custom`、`tasks`、`debug`、`checkpoints` 流 |
| `c08_04_multiple_modes_subgraphs.py` | 多模式流、子图输出和事件过滤 |
| `c08_05_event_streaming.py` | 消息、工具、生命周期事件和自定义投影 |
| `c08_06_langsmith_tracing.py` | LangSmith 开关、项目、元数据和脱敏 |

Knowledge points:

- State streaming answers "what changed?"
- Message streaming answers "what tokens or messages were produced?"
- Custom streaming transports application-specific progress.
- Debug and checkpoint streams expose runtime internals for development.
- Tracing records execution across nodes and model calls when enabled.

### c09: Subgraphs and Multi-Agent Systems

| Demo | Chinese teaching focus |
| --- | --- |
| `c09_01_subgraph_as_node.py` | 将编译后的子图直接作为父图节点 |
| `c09_02_subgraph_in_node.py` | 在节点函数中调用子图并转换状态 |
| `c09_03_subgraph_persistence_stream.py` | 父子状态、持久化继承和子图流式输出 |
| `c09_04_multi_agent_patterns.py` | Supervisor、Handoff、Agent-as-Tool 和并行 Agent |

Knowledge points:

- Subgraphs provide state and responsibility boundaries.
- The two invocation forms have different state and streaming behavior.
- Child persistence can inherit from or be isolated from the parent.
- Multi-agent systems should be introduced only when isolation, ownership, or
  parallel specialization justifies them.

### c10: Testing, Applications, and Capstone

| Demo | Chinese teaching focus |
| --- | --- |
| `c10_01_testing_partial_execution.py` | 节点/边单元测试、Graph 局部执行和断言 |
| `c10_02_agentic_rag.py` | 检索、文档评分、问题重写、回答和循环路由 |
| `c10_03_sql_agent_hitl.py` | SQLite Agent、查询工具、人工审核和错误恢复 |
| `c10_04_application_local_server.py` | 应用目录、配置和本地 HTTP/Studio 接入说明 |
| `c10_05_capstone_research_assistant.py` | 综合路由、工具、子图、审批、持久化和流式输出 |

Knowledge points:

- Test node contracts and routing rather than exact LLM prose.
- Partial execution reduces cost while debugging a graph.
- RAG and SQL agents are constrained tool loops with application-specific
  validation.
- Application structure separates graph definitions, configuration, and
  runtime concerns.
- The capstone integrates concepts instead of introducing a new framework.

## Official Documentation Mapping

The module will explicitly map knowledge to official pages:

- Orientation: `overview`, `install`, `quickstart`, `thinking-in-langgraph`,
  `choosing-apis`
- Graph and control flow: `graph-api`, `use-graph-api`, `pregel`
- Functional API: `functional-api`, `use-functional-api`
- Patterns and agents: `workflows-agents`
- Persistence and memory: `persistence`, `checkpointers`, `add-memory`,
  `stores`, `use-time-travel`
- Reliability and HITL: `interrupts`, `fault-tolerance`, and the six listed
  error pages
- Streaming and observability: `streaming`, `event-streaming`, `observability`
- Subgraphs and applications: `use-subgraphs`, `test`,
  `application-structure`, `agentic-rag`, `sql-agent`
- Local development: `local-server`, `studio`, `ui`

Pages such as `case-studies`, `changelog`, `backward-compatibility`, and
hosted deployment are summarized as reading notes only because they do not
produce self-contained local Python demos.

Official code is used as a technical baseline, but examples will be adapted
for DeepSeek, Chinese explanations, offline verification, and the repository's
chapter structure rather than copied verbatim.

## Data Flow

Most examples follow one of these flows:

```text
CLI input
  -> common parser
  -> DeepSeek model or deterministic fake model
  -> graph/function workflow
  -> state, events, checkpoints, or interrupts
  -> Chinese explanation and expected result
```

```text
verify_examples.py
  -> discover chapter scripts
  -> run script subprocess with --offline or live flags
  -> enforce timeout
  -> print PASS/FAIL and failure tail
  -> return non-zero if any required example fails
```

## DeepSeek Compatibility

`common.py` uses `ChatOpenAI` with:

```text
base_url = DEEPSEEK_BASE_URL or https://api.deepseek.com
model = DEEPSEEK_MODEL or deepseek-chat
```

`deepseek-chat` is the default because examples rely on tool calling and
structured output. `deepseek-reasoner` can be selected with `--model` for
comparison examples.

DeepSeek does not provide a public embedding endpoint through this setup.
Semantic Store and RAG examples therefore use deterministic local embeddings
for the learning path. Documentation will explain how to replace them with a
production embedding provider.

The module will document any capability that cannot be exercised with the
configured DeepSeek model instead of silently changing providers.

## Error Handling

- Missing `DEEPSEEK_API_KEY` produces a clear live-mode error that points to
  the repository `.env`.
- `--offline` never requires credentials.
- Interrupt examples use stable thread IDs so resume behavior is repeatable.
- Timeout examples use deterministic local operations when possible.
- File-based examples place temporary SQLite databases under a temporary
  directory and clean them up after execution.
- `verify_examples.py` applies per-example timeouts and prints the last output
  lines for failures.
- Live LLM assertions validate state shape and control flow, not exact prose.

## Testing Strategy

Implementation uses test-driven development for behavior with stable
contracts:

- reducer and message merge behavior;
- conditional routing and `Command` transitions;
- persistence across repeated invocations;
- history replay and fork behavior;
- interrupt and resume behavior;
- retry and timeout configuration;
- streaming event shape;
- subgraph state isolation;
- fake structured output and tool calling;
- offline verification of every runnable example.

Required verification commands:

```bash
venv/bin/python -m compileall langgraph-demo
venv/bin/python -m unittest discover -s tests -v
venv/bin/python langgraph-demo/verify_examples.py --offline
```

Live verification:

```bash
venv/bin/python langgraph-demo/verify_examples.py --live --group c01
venv/bin/python langgraph-demo/verify_examples.py --live --group c05
venv/bin/python langgraph-demo/verify_examples.py --live --group c10
```

Live examples are run by chapter to control cost. A live failure caused by a
specific provider limitation must be documented with the observed error and a
working alternative path.

## Documentation Contract

Each entry in `KNOWLEDGE_GUIDE.md` contains:

1. the learning goal;
2. the official concept being demonstrated;
3. the state and execution flow;
4. the key APIs and exact code locations;
5. the expected offline and live behavior;
6. common misunderstandings;
7. two or three exercises;
8. links to the corresponding official documentation.

`README.md` contains:

- installation and environment setup;
- the DeepSeek configuration matrix;
- recommended study order;
- chapter tables with all example commands;
- offline and live verification commands;
- version baselines and documentation sources;
- security notes for tools, SQL, tracing, and human approval.

The root `README.md` will add `langgraph-demo/` to the directory layout and
learning route.

## Implementation Phases

### Phase 1: Foundation

Create the module skeleton, README outline, requirements, common helpers,
fake-model contract, verification runner, and `c01-c02`.

### Phase 2: Stateful Orchestration

Implement `c03-c05`, covering advanced graph control flow, Functional API,
workflow patterns, and agents.

### Phase 3: Durable State and Human Control

Implement `c06-c07`, including Checkpointers, Store, time travel, interrupts,
retry, timeout, error routing, and graceful drain.

### Phase 4: Runtime Interfaces and Integration

Implement `c08-c10`, including streaming, tracing, subgraphs, multi-agent
patterns, RAG, SQL, testing, local-server guidance, and the capstone.

### Phase 5: Verification and Learning Documentation

Complete `KNOWLEDGE_GUIDE.md`, root navigation, offline verification, focused
unit tests, and available live DeepSeek smoke tests.

## Non-Goals

- Reproducing all 43 official pages verbatim.
- Building a production LangSmith deployment.
- Requiring PostgreSQL, Redis, or external vector services for the core path.
- Implementing production authentication and multi-tenancy for the local
  server example.
- Replacing `langchain-demo/` or refactoring unrelated modules.
- Hiding provider limitations behind mock-only behavior.

## Acceptance Criteria

- The module contains 53 runnable examples grouped into ten chapters.
- Every model-powered example supports `--offline` and DeepSeek live mode.
- Every example is covered in `KNOWLEDGE_GUIDE.md` with Chinese explanations.
- The Graph API, Functional API, workflows, agents, persistence, HITL,
  streaming, fault tolerance, subgraphs, and testing are represented.
- `verify_examples.py --offline` succeeds.
- Focused unit tests and the existing test suite succeed.
- At least the core DeepSeek-backed chapters pass live smoke verification when
  a valid API key is available.
- Documentation identifies every provider limitation and optional dependency.
