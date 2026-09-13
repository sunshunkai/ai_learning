# LangGraph Demo 逐文件知识指南

这份文档按文件解释每个示例的学习目标、执行流程、关键 API、常见误区和
练习方向。代码以 LangGraph 1.x 官方 Python 文档为基线，并针对 DeepSeek
和离线验证做了适配。

官方文档：

- [Overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart)
- [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [Functional API](https://docs.langchain.com/oss/python/langgraph/functional-api)
- [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [Streaming](https://docs.langchain.com/oss/python/langgraph/streaming)
- [Fault Tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance)

## 一、先理解六个基础问题

### 1. State

State 是图在一次执行中共享和演进的数据。节点通常不复制整份状态，而是
返回一个增量字典。LangGraph 根据字段类型和 Reducer 合并这些增量。

### 2. Node

Node 是普通 Python 函数或 Runnable。它负责读取状态、执行工作并返回状态
更新。节点可以有重试、超时、缓存和错误处理器。

### 3. Edge

Edge 决定节点执行顺序。普通边固定下一跳，条件边由路由函数决定下一跳。
`START` 和 `END` 是入口和出口标记。

### 4. Reducer

Reducer 决定多个更新如何合并。默认是后写覆盖；`operator.add` 可累加数值
或拼接列表；`add_messages` 专门处理消息新增、替换和删除。

### 5. Checkpointer 与 Store

Checkpointer 按 `thread_id` 保存一次会话的执行状态。Store 保存跨线程、
跨会话的长期数据，两者解决不同层次的问题。

### 6. DeepSeek 与离线 FakeModel

在线模式使用 `ChatOpenAI` 的 OpenAI 兼容配置访问 DeepSeek。离线模式用
`DeterministicFakeChatModel` 提供固定文本、结构化输出和工具调用；它不会
模拟真实模型质量，只用于验证图结构。

DeepSeek 可能不支持 LangChain 默认的 JSON Schema `response_format`。
结构化输出应显式传入 `method="function_calling"`，否则可能收到
`This response_format type is unavailable now`。

## 二、c01 入门与心智模型

### c01_01_env_check.py

**目标：** 确认 Python、LangGraph、LangChain 和 DeepSeek 配置。

**执行流程：** 读取包版本和环境变量 -> 不打印密钥 -> 根据不同模式给出结论。

**关键 API：** `importlib.metadata.version`、环境变量、`--offline`。

**易错点：** `langgraph` 没有稳定的顶层 `__version__`，应读取发行包元数据；
`--offline` 也不需要 API Key。

**练习：** 增加 `LANGSMITH_TRACING` 状态，并确保仍然不泄露 Key。

### c01_02_hello_state_graph.py

**目标：** 用最小图理解 State -> Node -> Edge -> Compile -> Invoke。

**执行流程：** 输入 `name` -> `greet` 节点返回 `message` -> `END`。

**关键 API：** `StateGraph`、`add_node`、`add_edge`、`compile`、`invoke`。

**易错点：** `compile()` 返回的是可执行图，不是原始 Builder；节点返回的是
状态增量，而不是完整 State。

**练习：** 增加第二个节点，把问候语转换成大写。

### c01_03_graph_vs_functional.py

**目标：** 对比同一任务的 Graph 和 Functional 写法。

**执行流程：** Graph 使用两个节点顺序执行；Functional 使用 `@task` 和
`@entrypoint` 组合函数。

**关键 API：** `entrypoint`、`task`、`.result()`、`invoke()`。

**易错点：** Functional API 不代表“没有状态”，任务仍可能被重放和持久化。

**练习：** 为两个版本分别增加一次失败重试。

## 三、c02 State 与 Graph 基础

### c02_01_state_schemas.py

**目标：** 对比 TypedDict、Pydantic 和 MessagesState。

**执行流程：** 三种 Schema 分别构建一个最小图并输出结果。

**关键 API：** `TypedDict`、`BaseModel`、`MessagesState`、`Annotated`。

**易错点：** Pydantic 提供校验能力，但也增加结构和序列化约束；消息状态应
使用 LangChain 消息类型，而不是随意字典。

**练习：** 为 Pydantic State 增加范围校验并验证非法输入。

### c02_02_reducers_messages.py

**目标：** 理解默认覆盖、数值累加和消息合并。

**执行流程：** 消息图回显历史；计数器图每次向 `count` 写入 1。

**关键 API：** `operator.add`、`Annotated`、`add_messages`。

**易错点：** 两个并行节点写同一个无 Reducer 字段会冲突；消息列表不能只靠
普通列表拼接，因为删除和消息 ID 需要专门处理。

**练习：** 添加删除消息的 `RemoveMessage` 更新。

### c02_03_input_output_private_state.py

**目标：** 限制公开输入输出，并在节点间传递私有字段。

**执行流程：** `InputState` -> `prepare` 生成 `PrivateState` -> `answer`
输出 `OutputState`。

**关键 API：** `input_schema`、`output_schema`、私有 TypedDict。

**易错点：** 公共 `OverallState` 不代表每个字段都必须公开；私有状态不能
被后续不声明该字段的节点读取。

**练习：** 增加审计字段，但不要暴露到最终输出。

### c02_04_nodes_edges.py

**目标：** 掌握普通边和条件边。

**执行流程：** `classify` 生成分类 -> 条件边选择短文本或长文本节点。

**关键 API：** `add_conditional_edges`、路径映射、路由函数。

**易错点：** 路由函数应返回明确的路径值；条件边不是执行任务的节点，尽量
保持它无副作用。

**练习：** 增加第三种路由分支。

### c02_05_compile_visualize.py

**目标：** 输出 Mermaid 图结构，并理解编译验证。

**执行流程：** 构建图 -> 编译 -> `get_graph().draw_mermaid()` -> 执行。

**关键 API：** `draw_mermaid`、可选 `draw_mermaid_png`。

**易错点：** PNG 渲染可能依赖外部服务或系统工具；教学验证不要求 PNG。

**练习：** 为 c03 的并行图输出 Mermaid。

## 四、c03 高级图控制流

### c03_01_branch_parallel.py

**目标：** 理解 fan-out/fan-in 和并发状态合并。

**执行流程：** `START` 同时进入 left/right -> 两边写入 `branches` ->
Reducer 合并列表。

**关键 API：** 多个 `START` 边、`Annotated[list, operator.add]`。

**易错点：** 并行节点不能对普通标量字段无规则地同时写入。

**练习：** 增加第三个并行节点，并在最后添加汇总节点。

### c03_02_conditional_routing.py

**目标：** 根据 State 动态决定后续节点。

**执行流程：** `classify` 设置 route -> 条件边选择 small/large。

**关键 API：** `Literal`、路由函数、路径映射。

**易错点：** 路由值必须覆盖映射键，否则会出现意外分支或运行错误。

**练习：** 改成三档路由并增加测试。

### c03_03_loops_recursion.py

**目标：** 设计有终止条件的循环。

**执行流程：** `increment` 后判断是否达到 limit；未达到则回到自身。

**关键 API：** 条件回边、`END`、`recursion_limit`。

**易错点：** 不能只依赖递归限制终止业务循环；模型状态异常时应提前退出。

**练习：** 增加超时或剩余步数检查。

### c03_04_send_map_reduce.py

**目标：** 在运行时根据输入创建未知数量的 Worker。

**执行流程：** `fan_out` 为每个 item 创建 `Send` -> Worker 并行处理 ->
Reducer 汇总结果。

**关键 API：** `Send`、`add_conditional_edges`、列表 Reducer。

**易错点：** `Send` 适合动态 Map；固定分支使用普通并行边即可。

**练习：** 让 Worker 返回结构化结果并保留 item ID。

### c03_05_command.py

**目标：** 同时更新状态和决定下一跳。

**执行流程：** `first` 返回 `Command(update=..., goto="second")` ->
`second` 跳转 `END`。

**关键 API：** `Command`、`goto`、`update`。

**易错点：** 作为输入的 `Command` 主要使用 `resume`；节点返回的 `Command`
才通常组合 `update` 和 `goto`。

**练习：** 增加一个根据状态选择不同节点的工具。

### c03_06_runtime_retry_timeout_cache.py

**目标：** 组合 Runtime Context、重试、超时和缓存配置。

**执行流程：** async 节点第一次失败 -> RetryPolicy 重试 -> 从 Runtime 读取
乘数 -> 返回结果。

**关键 API：** `Runtime`、`RetryPolicy`、`CachePolicy`、`InMemoryCache`。

**易错点：** Timeout 仅支持 async 节点；同步代码无法安全取消，因此编译
时会报错。

**练习：** 使用 `TimeoutPolicy` 分别设置 run/idle timeout。

## 五、c04 Functional API

### c04_01_entrypoint_task.py

**目标：** 掌握 Functional API 的两个入口装饰器。

**执行流程：** `@entrypoint` 工作流调用 `@task`，通过 Future 取得结果。

**关键 API：** `@entrypoint`、`@task`、`.result()`。

**易错点：** 不是所有普通函数都需要变成 task；只有需要重放、持久化或
并行调度的边界才适合。

**练习：** 增加第二个 task 并顺序组合。

### c04_02_futures_parallel.py

**目标：** 并行运行独立 task。

**执行流程：** 同时创建 square/double Future -> 最后读取两个结果。

**关键 API：** Future、`.result()`。

**易错点：** 如果创建 Future 后立刻 `result()`，就可能退化成顺序执行。

**练习：** 增加一个可能失败的 Future，并配置 fallback。

### c04_03_retry_resume.py

**目标：** 理解 task 重试和恢复。

**执行流程：** 首次任务抛出 ValueError -> RetryPolicy 再执行 -> 返回成功值。

**关键 API：** `RetryPolicy`、`retry_on`、`max_attempts`。

**易错点：** 重放时函数体可能再次执行，所有外部副作用必须幂等。

**练习：** 把失败计数改到外部数据库，并验证重复执行不会重复扣款。

### c04_04_graph_functional_interop.py

**目标：** 在两种 API 之间组合调用。

**执行流程：** Functional Workflow 调用编译图；另一个 Graph 节点调用
Functional Workflow。

**关键 API：** `entrypoint`、`task`、`StateGraph`、`invoke`。

**易错点：** 互调时要注意输入输出 Schema 和序列化边界，不能假设对象总在
同一进程内存中。

**练习：** 在一个 Graph 的多个节点中调用同一个 entrypoint。

## 六、c05 工作流与 Agent

### c05_01_structured_tools_warmup.py

**目标：** 预热结构化输出和工具请求。

**执行流程：** 模型返回 RouteSelection；绑定工具后返回 tool_calls。

**关键 API：** Pydantic、`with_structured_output(method="function_calling")`、
`bind_tools`、`@tool`。

**易错点：** 模型不会直接执行 Python 工具；程序必须验证并执行工具请求。

**练习：** 增加参数校验和未知工具名处理。

### c05_02_prompt_chaining.py

**目标：** 用多个清晰阶段减少单体 Prompt 的不可控性。

**执行流程：** outline -> draft -> polish。

**关键 API：** `StateGraph` 顺序边、消息调用。

**易错点：** 每一步都应有稳定输入输出契约，不能只靠隐式自然语言约定。

**练习：** 在 outline 和 draft 之间增加结构校验。

### c05_03_parallelization.py

**目标：** 并行生成不同视角的答案。

**执行流程：** short/detailed 两个节点并行写列表 -> Reducer 合并。

**关键 API：** 并行边、`operator.add`。

**易错点：** 并发调用成本和速率限制会叠加；输出顺序不保证稳定。

**练习：** 为每个分支增加独立重试和超时。

### c05_04_routing.py

**目标：** 用模型分类而不是让一个 Prompt 处理所有任务。

**执行流程：** 结构化路由 -> 条件边 -> 对应处理链。

**关键 API：** `with_structured_output`、`Literal`、条件边。

**易错点：** 路由分类错误会把任务送到错误处理链；应记录路由决策和置信度。

**练习：** 增加 `other` 路由作为兜底。

### c05_05_orchestrator_worker.py

**目标：** 动态拆分任务并并行派发 Worker。

**执行流程：** Planner -> subtasks -> Send Worker -> drafts 汇总。

**关键 API：** 结构化 Plan、`Send`、列表 Reducer。

**易错点：** Worker 数量不受控会增加成本；需要限制最大子任务数。

**练习：** 限制 subtasks 数量，并记录每个 Worker 的来源 ID。

### c05_06_evaluator_optimizer.py

**目标：** 通过评估反馈有限次优化结果。

**执行流程：** generate -> evaluate -> PASS 结束或有限次数回环。

**关键 API：** 条件回边、最大轮数、Evaluator 模型。

**易错点：** 生成器和评估器如果都过于宽松或严格，会出现无限互评或过早停止。

**练习：** 增加结构化评分和终止阈值。

### c05_07_react_tool_agent.py

**目标：** 理解模型与工具的最小 ReAct 循环。

**执行流程：** agent 模型提出工具调用 -> ToolNode 执行 -> ToolMessage
回传 -> agent 生成最终回答。

**关键 API：** `ToolNode`、`tools_condition`、`bind_tools`。

**易错点：** 必须存在明确终止路径；工具错误也应回传模型或进入错误处理。

**练习：** 增加工具调用次数上限和工具异常演示。

## 七、c06 持久化、记忆与时间旅行

### c06_01_thread_checkpointer.py

**目标：** 理解 `thread_id` 是会话边界。

**执行流程：** 同一 thread 连续调用累积消息；不同 thread 互相隔离。

**关键 API：** `InMemorySaver`、`compile(checkpointer=...)`、config。

**易错点：** InMemorySaver 进程退出后数据消失；模型本身不会自动保存历史。

**练习：** 增加第二个用户并比较消息数量。

### c06_02_state_history_replay.py

**目标：** 查看 Checkpoint 历史并从旧状态重放。

**执行流程：** 多次调用 -> `get_state_history()` -> 从选定 snapshot
`invoke(None)`。

**关键 API：** `get_state`、`get_state_history`、snapshot.config。

**易错点：** Replay 会重新执行后续节点及副作用，必须保证可重放。

**练习：** 比较不同 snapshot 的 `next` 和 metadata。

### c06_03_fork_time_travel.py

**目标：** 从历史点修改状态并创建新分支。

**执行流程：** 找到 write 前 snapshot -> `update_state` -> Fork 后继续。

**关键 API：** `update_state`、`as_node`、snapshot.config。

**易错点：** Fork 后当前线程的最新 pointer 会移动；需要保存原 snapshot
config 才能回看原分支。

**练习：** 从同一 snapshot 创建三个不同主题的分支。

### c06_04_sqlite_checkpointer.py

**目标：** 展示跨进程的持久化 Checkpointer。

**执行流程：** 创建临时 SQLite 文件 -> 执行两次 -> 读取最终状态。

**关键 API：** `SqliteSaver.from_conn_string`。

**易错点：** 该能力来自可选包；未安装时不能把 InMemorySaver 冒充为
SQLite。

**练习：** 进程 A 写入状态，进程 B 使用同一数据库读取状态。

### c06_05_store_long_term_memory.py

**目标：** 理解长期记忆与线程状态的区别。

**执行流程：** 按用户 namespace 写入记忆 -> get 精确读取 -> search 查询。

**关键 API：** `InMemoryStore`、namespace、`put/get/search`、Embedding。

**易错点：** 长期记忆需要明确的写入、检索、更新和删除策略，不能无限累积。

**练习：** 增加“删除记忆”和“更新覆盖”流程。

### c06_06_memory_management.py

**目标：** 控制消息长度和成本。

**执行流程：** 写入多条消息 -> 删除旧消息 -> 保留最后 N 条。

**关键 API：** `RemoveMessage`、`update_state`、消息摘要。

**易错点：** 删除 Checkpoint 状态和删除 Store 记忆是不同操作。

**练习：** 用 LLM 摘要替换旧消息，并验证摘要仍保留关键事实。

## 八、c07 HITL、容错与错误复现

### c07_01_interrupt_resume.py

**目标：** 理解暂停和恢复的最小机制。

**执行流程：** 节点调用 `interrupt()` -> invoke 返回 `__interrupt__` ->
`Command(resume=...)` 恢复。

**关键 API：** `interrupt`、`Command`、Checkpointer。

**易错点：** 没有 Checkpointer 就无法按预期恢复；resume 值和节点内调用
位置必须匹配。

**练习：** 返回结构化问题并校验人工答案。

### c07_02_approve_reject.py

**目标：** 为高风险操作设计批准与拒绝分支。

**执行流程：** approval 中断 -> 布尔恢复值 -> 条件边选择 approve/reject。

**关键 API：** `interrupt`、条件边、`Command(resume=bool)`。

**易错点：** 拒绝也应是明确业务流程，不能只把中断视为异常。

**练习：** 增加“修改参数后重新审批”。

### c07_03_review_edit_state.py

**目标：** 允许人工修改待审核内容。

**执行流程：** 中断返回 draft -> 人工提交字典 -> 节点写回新 draft。

**关键 API：** 结构化 interrupt payload、恢复字典。

**易错点：** 恢复数据需要校验，不能直接信任任意字段。

**练习：** 对长度、敏感词和格式做二次校验。

### c07_04_multiple_tool_interrupts.py

**目标：** 处理一个执行中的多个中断。

**执行流程：** `ask_name` 暂停 -> 恢复后进入 `ask_count` 再次暂停 -> 完成。

**关键 API：** 多次 `invoke(Command(resume=...))`。

**易错点：** 并行中断时不能再依赖顺序，需要按 Interrupt ID 返回映射。

**练习：** 改成并行中断，并用 `{id: value}` 一次性恢复。

### c07_05_retries_timeouts_errors.py

**目标：** 区分重试、超时和错误处理器。

**执行流程：** Retry 节点失败后重试；error handler 在重试耗尽后接管。

**关键 API：** `RetryPolicy`、`NodeError`、`error_handler`。

**易错点：** Timeout 只支持 async 节点；错误处理本身也必须避免无限失败。

**练习：** 给 handler 增加审计记录和人工升级路径。

### c07_06_graceful_drain.py

**目标：** 服务关闭时停在安全边界并保存 Checkpoint。

**执行流程：** 节点调用 `request_drain` -> 当前 superstep 完成 ->
`GraphDrained` -> 相同 config 恢复。

**关键 API：** `RunControl`、`GraphDrained`、`Runtime`。

**易错点：** Drain 不强制取消正在运行的代码，只会在 superstep 边界生效。

**练习：** 在 SIGTERM 信号处理器中调用 `request_drain`。

### c07_07_error_reproduction.py

**目标：** 通过错误了解 LangGraph 的约束。

**执行流程：** 根据 `--case` 构造错误场景并打印错误类型和修复方向。

**关键 API：** GraphRecursionError、Missing Checkpointer、Invalid Update。

**易错点：** `bad-history` 和部署型多子图问题依赖具体模型或部署环境，
示例会明确输出 `SKIP`，不伪造错误。

**练习：** 为每个会话增加断言，防止同类错误回归。

## 九、c08 Streaming、事件与可观测性

### c08_01_stream_values_updates.py

**目标：** 区分完整状态流和增量流。

**执行流程：** 两个节点递增 -> 同时监听 values/updates。

**关键 API：** `stream_mode=["values", "updates"]`、`version="v2"`。

**易错点：** values 每次是完整状态，数据量可能大；updates 只包含本步写入。

**练习：** 记录每个节点耗时并与事件流关联。

### c08_02_stream_messages.py

**目标：** 展示 LLM 消息流和元数据。

**执行流程：** 节点调用模型 -> messages 流返回 Message + metadata。

**关键 API：** `stream_mode="messages"`、`langgraph_node`。

**易错点：** 消息流顺序不保证等于图执行顺序；前端应结合 namespace。

**练习：** 增加 Token 拼接和工具调用状态显示。

### c08_03_custom_tasks_checkpoints.py

**目标：** 了解不同调试和业务流模式。

**执行流程：** 节点写 custom event -> 同时观察 tasks/debug/checkpoints。

**关键 API：** `get_stream_writer`、custom/tasks/debug/checkpoints。

**易错点：** debug/checkpoints 适合开发，不应把底层对象直接暴露给终端用户。

**练习：** 只向前端输出 custom 格式的进度百分比。

### c08_04_multiple_modes_subgraphs.py

**目标：** 同时观察父图和子图。

**执行流程：** 子图写 custom -> 父图继续处理 -> 多模式输出。

**关键 API：** 多 stream mode、子图、namespace。

**易错点：** 需要区分同名节点和不同 namespace 的事件来源。

**练习：** 前端按 namespace 构建父子执行树。

### c08_05_event_streaming.py

**目标：** 观察 v3 协议事件。

**执行流程：** `stream_events(version="v3")` 产生 values/lifecycle 等事件。

**关键 API：** `stream_events`、`method`、`params`、`seq`。

**易错点：** v3 协议仍标记 experimental，升级时需要检查兼容性。

**练习：** 把 values 事件投影成前端状态快照。

### c08_06_langsmith_tracing.py

**目标：** 检查追踪配置并理解可观测性边界。

**执行流程：** 读取 tracing 环境变量 -> 执行本地图 -> 打印配置状态。

**关键 API：** `LANGSMITH_TRACING`、`LANGSMITH_PROJECT`。

**易错点：** 开启追踪后输入和输出可能上传，必须先脱敏。

**练习：** 给 trace 增加 user/session metadata，并移除敏感字段。

## 十、c09 子图与多 Agent

### c09_01_subgraph_as_node.py

**目标：** 把编译子图直接组合进父图。

**执行流程：** 父子图共享 State Schema -> child_graph 作为父节点 -> 输出。

**关键 API：** `add_node("child", compiled_graph)`。

**易错点：** 这种组合方式要求父子状态边界兼容。

**练习：** 在父图中并行放两个独立子图。

### c09_02_subgraph_in_node.py

**目标：** 在节点内调用子图并转换状态。

**执行流程：** ParentState -> 构造 Child input -> `child.invoke` ->
只返回需要公开的字段。

**关键 API：** 子图 `invoke`、输入输出映射。

**易错点：** 显式调用时要处理异常、配置和子图 Checkpoint 语义。

**练习：** 把子图私有字段记录下来，但不写入父状态。

### c09_03_subgraph_persistence_stream.py

**目标：** 理解子图持久化和流式 namespace。

**执行流程：** 子图继承 Checkpointer -> 父图流式执行 -> 子图事件带 ns。

**关键 API：** InMemorySaver、`subgraphs=True`、custom stream。

**易错点：** 有 Checkpointer 时必须传 thread_id；不同持久化策略可能隔离
或共享子图状态。

**练习：** 对比父图有/无 Checkpointer 时的子图恢复行为。

### c09_04_multi_agent_patterns.py

**目标：** 对比四种常见多 Agent 结构。

**执行流程：** Supervisor 路由、Handoff 顺序移交、Agent-as-Tool 工具化、
Parallel 并行输出。

**关键 API：** 结构化路由、ToolNode、条件边、Reducer。

**易错点：** 多 Agent 会放大会话成本、延迟和状态复杂度；没有明确边界时
单 Agent 更容易维护。

**练习：** 为 Supervisor 增加最大移交次数和失败回退。

## 十一、c10 测试、数据 Agent 与综合项目

### c10_01_testing_partial_execution.py

**目标：** 先测节点契约，再从中间节点调试。

**执行流程：** 直接调用节点函数 -> 断言结果 -> `interrupt_after` 暂停 ->
`invoke(None)` 继续。

**关键 API：** 节点单元测试、`interrupt_after`、Checkpointer。

**易错点：** 不要只断言 LLM 原文；应断言 State、路由和关键结构。

**练习：** 为失败路径增加测试。

### c10_02_agentic_rag.py

**目标：** 构建带评分和重写的检索闭环。

**执行流程：** retrieve -> grade -> relevant 则 answer，否则 rewrite ->
retrieve。

**关键 API：** 本地检索、条件路由、循环、模型生成。

**易错点：** 相似度分数不等于事实正确；仍需限制循环次数并保留引用。

**练习：** 增加文档引用和命中分数输出。

### c10_03_sql_agent_hitl.py

**目标：** 让模型/规则生成 SQL，但由人工批准后执行。

**执行流程：** plan -> interrupt 审核 -> SELECT 校验 -> SQLite 执行。

**关键 API：** `interrupt`、`Command`、只读 SQL 校验、sqlite3。

**易错点：** 字符串级 SQL 校验不能替代数据库账号权限；生产环境应使用
只读账号和查询超时。

**练习：** 增加结果行数上限和查询超时。

### c10_04_application_local_server.py

**目标：** 理解应用目录和本地 Server/Studio 接入。

**执行流程：** 检查 FastAPI/Uvicorn/langgraph-cli -> 打印推荐布局。

**关键 API：** `langgraph.json`、`langgraph dev`、应用入口。

**易错点：** 本地开发服务不应把密钥写入仓库；部署需要独立认证和配额。

**练习：** 创建一个最小 `langgraph.json` 并让 Studio 发现图。

### c10_05_capstone_research_assistant.py

**目标：** 综合应用前面章节。

**执行流程：** 结构化路由 -> 本地研究工具 -> 人工审批 -> Checkpoint ->
模型生成最终回答。

**关键 API：** Pydantic、Tool、interrupt、InMemorySaver、Command。

**易错点：** 综合项目中真正的风险不在“图是否能跑”，而在权限、幂等、
恢复、脱敏、成本和失败可观测性。

**练习：** 把最终回答改为流式输出，并增加调用次数和 Token 统计。

## 十二、继续深入的方向

1. 把 InMemorySaver 替换为 SQLite/Postgres Checkpointer。
2. 把本地 Embedding 替换为正式 Embedding 和向量数据库。
3. 为 Agent 增加权限、审计、配额和超时。
4. 用 LangSmith 评估路由、工具选择和最终答案质量。
5. 把安全工具操作集中到受控服务，不让模型直接访问生产系统。
