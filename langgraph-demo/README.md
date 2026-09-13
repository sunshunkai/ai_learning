# LangGraph 学习模块

本目录基于 LangGraph 1.x 官方 Python 文档整理，使用 53 个可运行示例讲解
Graph API、Functional API、工作流、Agent、持久化、HITL、流式、子图、
多 Agent 和综合应用。

模型默认通过 OpenAI 兼容接口接入 DeepSeek。所有依赖模型的示例都支持
`--offline`，使用确定性本地 FakeModel，不访问网络。

逐文件原理说明见 [`KNOWLEDGE_GUIDE.md`](./KNOWLEDGE_GUIDE.md)。

官方参考：

- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangGraph Python Docs Index](https://docs.langchain.com/oss/python/langgraph/llms.txt)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)

## 版本基线

本文档和代码按以下环境验证：

| 组件 | 版本 |
| --- | --- |
| Python | 3.13 |
| langgraph | 1.2.x |
| langgraph-checkpoint | 4.2.x |
| langchain | 1.4.x |
| langchain-core | 1.6.x |
| langchain-openai | 1.6.x |
| Pydantic | 2.x |

安装依赖：

```bash
venv/bin/pip install -r langgraph-demo/requirements.txt
```

SQLite Checkpointer 是可选能力：

```bash
venv/bin/pip install 'langgraph-checkpoint-sqlite>=3,<4'
```

未安装时，`c06_04_sqlite_checkpointer.py` 会输出 `SKIP`，不会让离线验证失败。

## DeepSeek 配置

项目根目录 `.env`：

```text
DEEPSEEK_API_KEY=你的 DeepSeek Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

如果 `.env` 未配置 `DEEPSEEK_MODEL`，默认模型是 `deepseek-chat`。可通过
`--model deepseek-reasoner` 做推理模型对照。

DeepSeek 当前可能不支持 LangChain 默认使用的 JSON Schema
`response_format`。本模块的结构化输出示例显式使用
`with_structured_output(..., method="function_calling")`，基于 DeepSeek
工具调用返回 Pydantic 对象。

运行在线示例：

```bash
venv/bin/python langgraph-demo/c01_01_env_check.py
venv/bin/python langgraph-demo/c05_07_react_tool_agent.py
```

运行离线示例：

```bash
venv/bin/python langgraph-demo/c01_01_env_check.py --offline
venv/bin/python langgraph-demo/c05_07_react_tool_agent.py --offline
```

`--offline` 不读取 Key，也不访问网络。示例替换的只是模型层，StateGraph、
Reducer、Checkpointer、Interrupt 和 Streaming 结构保持不变。

## 学习路线

建议按 `c01 -> c10` 顺序学习：

1. 先理解 State、Node、Edge 和 Compile。
2. 掌握 Reducer、条件路由、并行、循环、Send 和 Command。
3. 对照 Graph API 学习 Functional API。
4. 学习官方工作流模式和 ReAct Agent。
5. 学习 Checkpointer、Store 和时间旅行。
6. 学习 HITL、重试、超时、错误处理和 Graceful Drain。
7. 学习流式输出、事件协议和 LangSmith。
8. 学习子图和多 Agent。
9. 最后完成测试、RAG、SQL 和综合项目。

目录命名为 `langgraph-demo/`，避免遮蔽 pip 安装的 `langgraph` 包。

## c01 入门与心智模型

| 文件 | 知识点 |
| --- | --- |
| `c01_01_env_check.py` | Python、LangGraph、LangChain、DeepSeek 配置检查 |
| `c01_02_hello_state_graph.py` | 最小 StateGraph、节点、边和编译 |
| `c01_03_graph_vs_functional.py` | Graph API 与 Functional API 对比 |

```bash
venv/bin/python langgraph-demo/c01_02_hello_state_graph.py --offline
```

## c02 State 与 Graph 基础

| 文件 | 知识点 |
| --- | --- |
| `c02_01_state_schemas.py` | TypedDict、Pydantic、MessagesState |
| `c02_02_reducers_messages.py` | 默认覆盖、`operator.add`、`add_messages` |
| `c02_03_input_output_private_state.py` | 输入、输出和节点私有状态 |
| `c02_04_nodes_edges.py` | 普通边、条件边、START、END |
| `c02_05_compile_visualize.py` | Mermaid 和可选 PNG 可视化 |

## c03 高级图控制流

| 文件 | 知识点 |
| --- | --- |
| `c03_01_branch_parallel.py` | 并行 fan-out/fan-in 与 Reducer 聚合 |
| `c03_02_conditional_routing.py` | 条件路由 |
| `c03_03_loops_recursion.py` | 循环与递归限制 |
| `c03_04_send_map_reduce.py` | `Send` 动态 Map-Reduce |
| `c03_05_command.py` | `Command` 更新状态并跳转 |
| `c03_06_runtime_retry_timeout_cache.py` | Runtime、Retry、Timeout、Cache |

`c03_06` 使用 async 节点，因为 LangGraph 的同步节点不支持可取消 Timeout。

## c04 Functional API

| 文件 | 知识点 |
| --- | --- |
| `c04_01_entrypoint_task.py` | `@entrypoint` 与 `@task` |
| `c04_02_futures_parallel.py` | Future 并行和结果聚合 |
| `c04_03_retry_resume.py` | RetryPolicy 与失败恢复 |
| `c04_04_graph_functional_interop.py` | 两种 API 双向互调 |

## c05 工作流与 Agent

| 文件 | 知识点 |
| --- | --- |
| `c05_01_structured_tools_warmup.py` | 结构化输出和工具调用 |
| `c05_02_prompt_chaining.py` | Prompt Chaining |
| `c05_03_parallelization.py` | Parallelization |
| `c05_04_routing.py` | Routing |
| `c05_05_orchestrator_worker.py` | Orchestrator-Worker |
| `c05_06_evaluator_optimizer.py` | Evaluator-Optimizer |
| `c05_07_react_tool_agent.py` | ToolNode 与 ReAct 循环 |

## c06 持久化、记忆与时间旅行

| 文件 | 知识点 |
| --- | --- |
| `c06_01_thread_checkpointer.py` | `thread_id` 与 InMemorySaver |
| `c06_02_state_history_replay.py` | 状态历史和 Replay |
| `c06_03_fork_time_travel.py` | 修改历史状态和 Fork |
| `c06_04_sqlite_checkpointer.py` | SQLite 跨进程持久化 |
| `c06_05_store_long_term_memory.py` | Store 和本地确定性 Embedding |
| `c06_06_memory_management.py` | 消息裁剪、删除和摘要 |

DeepSeek 当前配置不提供 Embedding 接口，因此长期记忆示例使用本地向量，
只演示 Store 接口。生产环境可替换为 OpenAI、BGE 或自建 Embedding 服务。

## c07 HITL、容错与错误复现

| 文件 | 知识点 |
| --- | --- |
| `c07_01_interrupt_resume.py` | `interrupt()` 和 `Command(resume=...)` |
| `c07_02_approve_reject.py` | 批准与拒绝分支 |
| `c07_03_review_edit_state.py` | 人工修改状态 |
| `c07_04_multiple_tool_interrupts.py` | 一个执行中的多个中断 |
| `c07_05_retries_timeouts_errors.py` | RetryPolicy 和 NodeError |
| `c07_06_graceful_drain.py` | GraphDrained 和安全恢复 |
| `c07_07_error_reproduction.py` | 常见错误复现和修复提示 |

运行所有错误复现：

```bash
for case in recursion missing-checkpointer invalid-return concurrent-update bad-history multiple-subgraphs; do
  venv/bin/python langgraph-demo/c07_07_error_reproduction.py --case "$case"
done
```

## c08 Streaming、事件与可观测性

| 文件 | 知识点 |
| --- | --- |
| `c08_01_stream_values_updates.py` | `values` 与 `updates` |
| `c08_02_stream_messages.py` | 消息和 Token 流 |
| `c08_03_custom_tasks_checkpoints.py` | custom/tasks/debug/checkpoints |
| `c08_04_multiple_modes_subgraphs.py` | 多模式流与子图 |
| `c08_05_event_streaming.py` | v3 事件协议 |
| `c08_06_langsmith_tracing.py` | LangSmith 配置与脱敏提醒 |

LangSmith tracing 是可选能力。未配置时示例只检查状态并继续运行。

## c09 子图与多 Agent

| 文件 | 知识点 |
| --- | --- |
| `c09_01_subgraph_as_node.py` | 编译子图直接作为父图节点 |
| `c09_02_subgraph_in_node.py` | 节点中调用子图并转换状态 |
| `c09_03_subgraph_persistence_stream.py` | 子图持久化和 namespace 流 |
| `c09_04_multi_agent_patterns.py` | Supervisor、Handoff、Agent-as-Tool、并行 Agent |

运行多 Agent 的四种结构：

```bash
for pattern in supervisor handoff agent-tool parallel; do
  venv/bin/python langgraph-demo/c09_04_multi_agent_patterns.py \
    --offline --pattern "$pattern"
done
```

## c10 测试、数据 Agent 与综合项目

| 文件 | 知识点 |
| --- | --- |
| `c10_01_testing_partial_execution.py` | 节点测试和局部执行 |
| `c10_02_agentic_rag.py` | 检索评分、问题重写和回答 |
| `c10_03_sql_agent_hitl.py` | 只读 SQL 和人工审批 |
| `c10_04_application_local_server.py` | 应用结构、Server/Studio 检查 |
| `c10_05_capstone_research_assistant.py` | 综合路由、工具、审批和持久化 |

`c10_04` 不会在验证时启动服务，只检查目录建议和可选依赖。

## 批量验证

列出 53 个示例：

```bash
venv/bin/python langgraph-demo/verify_examples.py --list
```

离线验证：

```bash
venv/bin/python langgraph-demo/verify_examples.py --offline
```

在线 DeepSeek 验证，可按章节控制成本：

```bash
venv/bin/python langgraph-demo/verify_examples.py --live --group c01
venv/bin/python langgraph-demo/verify_examples.py --live --group c05
venv/bin/python langgraph-demo/verify_examples.py --live --group c10
```

完整测试：

```bash
venv/bin/python -m compileall langgraph-demo tests
venv/bin/python -m unittest discover -s tests -v
```

## 安全说明

- SQL 示例只允许单条 SELECT，并在执行前暂停等待人工批准。
- 工具示例只调用本地、无副作用的演示函数。
- 不应把真实密钥、用户隐私或敏感上下文发送给未脱敏的 LangSmith 项目。
- `InMemorySaver` 只适合开发和测试，生产环境应使用持久化 Checkpointer。
- 出现不可信模型输出时，应先验证结构、权限和副作用，再执行外部操作。

## 与 LangChain 模块的关系

`langchain-demo/` 重点是模型、Prompt、Tool 和 Agent 的高层组件；
`langgraph-demo/` 重点是状态、执行、持久化和人工控制等底层编排能力。
两个目录可以独立学习，也可以交叉对照。
