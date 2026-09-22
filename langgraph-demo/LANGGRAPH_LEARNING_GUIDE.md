# LangGraph 1.x 系统学习手册

> 这是一份面向本仓库 `langgraph-demo/` 的完整学习文档。它不是 API 速查表，
> 而是按“概念 -> 官方依据 -> 示例代码 -> 执行结果 -> 常见错误 -> 练习”
> 组织的章节式教程。
>
> 文档基线：2026-09-16 的 LangChain / LangGraph Python 官方文档，以及本仓库
> 当前安装的 `langgraph 1.2.11`、`langgraph-checkpoint 4.2.0`、
> `langchain 1.4.0`、`langchain-openai 1.6.1`、`pydantic 2.13.5`。
>
> v2 修订：同日对照官方文档逐页扩充（Graph API、Functional API、Persistence、
> Checkpointers、Stores、Interrupts、Fault tolerance、Streaming、Event streaming、
> Subgraphs、Multi-agent、Time travel），各章新增"深入补充"小节（编号 x.y.1）。

## 章节导航

| 章节 | 主题 | 示例数 |
| --- | --- | --- |
| [第 0 章](#第-0-章-建立-langgraph-心智模型) | LangGraph 心智模型与执行模型 | 0 |
| [第 1 章](#第-1-章-入门从最小状态图到两种-api) | 入门与 Graph / Functional API | 3 |
| [第 2 章](#第-2-章-state-与-graph-基础) | State、Reducer、Node、Edge | 5 |
| [第 3 章](#第-3-章-高级图控制流) | 并行、循环、Send、Command、策略 | 6 |
| [第 4 章](#第-4-章-functional-api) | Entrypoint、Task、Future、互调 | 4 |
| [第 5 章](#第-5-章-工作流与-agent) | Workflow 模式与 Tool Agent | 7 |
| [第 6 章](#第-6-章-持久化记忆与时间旅行) | Checkpointer、Store、Replay、Fork | 6 |
| [第 7 章](#第-7-章-human-in-the-loop容错与错误诊断) | HITL、重试、超时、Drain、错误 | 7 |
| [第 8 章](#第-8-章-streaming事件协议与可观测性) | Streaming、Event Streaming、LangSmith | 6 |
| [第 9 章](#第-9-章-子图与多-agent) | Subgraph 与多 Agent 模式 | 4 |
| [第 10 章](#第-10-章-测试数据-agent应用结构与综合项目) | 测试、RAG、SQL、Server、Capstone | 5 |
| [附录 A](#附录-a53-个示例总索引) | 53 个示例总索引 | 53 |
| [附录 B](#附录-b核心-api-速查) | 核心 API 速查 | - |
| [附录 C](#附录-c常见错误排查顺序) | 常见错误排查 | - |
| [附录 D](#附录-d官方文档索引) | 官方文档索引 | - |
| [附录 E](#附录-e建议的-14-天学习计划) | 14 天学习计划 | - |

## 这份文档怎么用

`README.md` 负责说明“有哪些示例、如何安装和运行”；
`KNOWLEDGE_GUIDE.md` 负责按文件快速回忆；
本文负责建立完整知识体系，建议作为主教材使用。

推荐使用三遍学习法：

1. **先建模型**：读每章开头的“本章要解决的问题”和“核心概念”，不要先背 API。
2. **再跑示例**：每个示例先运行 `--offline`，观察输入、状态变化和最终输出。
3. **最后改代码**：完成每节练习，并回答章末自测题；只读代码不算真正掌握。

运行所有示例前先确认虚拟环境和依赖：

```bash
venv/bin/python langgraph-demo/c01_01_env_check.py --offline
venv/bin/python langgraph-demo/verify_examples.py --list
```

离线模式与在线模式的区别：

| 模式 | 命令 | 模型 | 网络与成本 | 适合用途 |
| --- | --- | --- | --- | --- |
| 离线 | 示例末尾加 `--offline` | `DeterministicFakeChatModel` | 无网络、无 API 成本 | 学习图结构、验证状态和路由 |
| 在线 | 不传 `--offline` | DeepSeek OpenAI 兼容接口 | 需要 Key，会产生调用成本 | 观察真实模型决策和生成质量 |

本模块的离线模型只替代模型层。`StateGraph`、Reducer、Checkpointer、
`interrupt()`、Streaming、Subgraph 等运行时行为仍然真实执行。

---

## 第 0 章 建立 LangGraph 心智模型

### 0.1 LangGraph 是什么

官方定义中，LangGraph 是用于构建、管理和部署长时间运行、有状态 Agent 的
**低层编排框架与运行时**。它的重点不是 Prompt，也不是替你决定 Agent 架构，
而是提供：

- 持久化执行：失败后可恢复，长任务可继续；
- Human-in-the-loop：暂停、检查、修改并恢复执行；
- 短期记忆与长期记忆；
- 流式输出；
- 确定性代码与模型驱动步骤的混合编排；
- 可观测、可测试、可部署的执行结构。

这意味着：

- 能用普通 Python 确定完成的事情，优先写成普通节点；
- 只有确实需要模型判断的步骤，才交给 LLM；
- Graph 是控制流和数据流的显式表达，不是“把一切交给 Agent”的魔法；
- 图能运行不等于生产可用，还要处理幂等、权限、幂等副作用、恢复、审计和成本。

官方依据：

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [Thinking in LangGraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph)
- [LangGraph runtime](https://docs.langchain.com/oss/python/langgraph/pregel)

### 0.2 一次图执行到底发生了什么

以 `START -> node_a -> node_b -> END` 为例：

1. `invoke()` 或 `stream()` 接收初始状态。
2. Runtime 把当前可运行的节点组成一个 Super-step。
3. 节点读取状态，并返回“状态增量”，不是必须返回完整状态。
4. Runtime 按字段的 Reducer 合并增量。
5. 当配置了 Checkpointer 时，每个 Super-step 边界保存 Checkpoint。
6. 边决定下一批节点；没有下一节点时执行结束。

Super-step 很关键：

- 同一 Super-step 中可并行执行多个节点；
- Checkpoint 主要在 Super-step 边界形成；
- `tasks` / `checkpoints` / `debug` 流能观察这些内部执行事件；
- Graceful Drain 也发生在安全边界，不会强行终止正在执行的节点。

### 0.3 State、Node、Edge、Reducer 是四个基础构件

| 构件 | 作用 | 学习时先问的问题 |
| --- | --- | --- |
| State | 图中跨节点共享的数据契约 | 哪些字段属于输入、输出或私有状态？ |
| Node | 一个 Python 函数、Runnable 或编译子图 | 它读取什么、返回什么增量？ |
| Edge | 固定下一跳或条件选择下一跳 | 为什么下一步是它？谁决定分支？ |
| Reducer | 合并同一字段的多次更新 | 默认覆盖是否足够？并行更新如何合并？ |

### 0.4 Graph API 与 Functional API

两者共享同一个底层 Runtime，可以在同一应用内组合。

| 对比项 | Graph API | Functional API |
| --- | --- | --- |
| 表达方式 | 声明式：State、Node、Edge | 命令式：Python 函数、`if`、`for` |
| 共享状态 | 显式 State Schema | 状态主要局限在函数参数和返回值 |
| 可视化 | 天然适合 | 运行期动态生成，不强调静态可视化 |
| 并行 | 显式 fan-out / fan-in、`Send` | 同时创建多个 Future，再聚合结果 |
| 适合 | 复杂分支、共享状态、并行和团队协作 | 改造既有过程式代码、线性流程、快速原型 |

选择原则不是“哪个更高级”，而是“哪种表达更贴合业务控制流”。

官方依据：

- [Choosing between Graph and Functional APIs](https://docs.langchain.com/oss/python/langgraph/choosing-apis)
- [Graph API overview](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [Functional API overview](https://docs.langchain.com/oss/python/langgraph/functional-api)

### 0.5 本手册的知识地图

```text
执行基础
  State / Node / Edge / Reducer / Compile
        |
        v
控制流
  branch / loop / Send / Command / runtime policy
        |
        +--------------------+
        |                    |
        v                    v
  Graph API           Functional API
        |                    |
        +----------+---------+
                   |
                   v
        Workflow patterns / Agent
                   |
                   v
 Persistence / Memory / Time travel / HITL
                   |
                   v
 Streaming / Subgraph / Multi-agent / Test / Deployment
```

### 0.6 必须先记住的边界

- `invoke()` 的普通输入若是 `Command(update=...)`，不是受支持的“继续会话”
  方式；恢复中断应使用 `Command(resume=...)`，多轮会话应传普通输入字典。
- `interrupt()` 依赖 Checkpointer 和稳定的 `thread_id`。
- Interrupt 节点恢复时会从该节点开头重跑；`interrupt()` 之前的副作用必须幂等。
- Checkpointer 管单个 thread 的图状态；Store 管跨 thread 的应用数据。
- `RetryPolicy` 处理“是否再次尝试”；Timeout 限制单次尝试；Error Handler 在重试
  耗尽后接管。
- Streaming 的 `v2` 是统一 `StreamPart` 格式；Event Streaming 的 `v3` 是
  typed projections，当前仍要关注实验性变化。

### 0.7 深入补充：生态定位、耐久模式与官方要点

#### 0.7.1 四层产品矩阵：每层解决什么问题

官方把产品矩阵分为四层，LangGraph 只负责"编排运行时"这一层。理解分层不是为了
背概念，而是为了回答一个实际问题：**我遇到的问题应该在哪一层解决？**

- **LangChain（Agent 框架层）**：提供模型抽象（`init_chat_model`）、工具定义
  （`@tool`）、Agent 循环等高层抽象与各家模型/向量库的集成适配。它解决
  "怎么调模型、怎么定义工具"的问题。LangGraph 常用其组件（比如
  `with_structured_output`、`bind_tools`），但**不强制**——你完全可以只用
  LangGraph + 原生 OpenAI SDK。
- **LangGraph（编排运行时层）**：提供 durable execution（可持久化、可恢复的
  执行）、streaming（流式输出协议）、human-in-the-loop（人工介入）、
  persistence（状态持久化与记忆）。它解决"Agent 跑起来之后，状态放哪、
  怎么恢复、怎么观察、怎么被人接管"的问题。这层是框架，其他层是可选项。
- **Deep Agents（Agent harness 层）**：构建在 LangGraph 之上的开箱即用 Agent
  骨架，内置规划（planning）、子 Agent、文件系统工具、上下文管理等模式。
  适合"我想要一个能干长活的通用 Agent，不想自己搭"的场景；如果你需要精细
  控制每一步流程，直接用 LangGraph。
- **LangSmith（可观测与平台层）**：tracing、评估、prompt 管理、部署。
  与 LangGraph 配合使用（开启环境变量即可上报 trace）。它的 Engine 还能
  分析 trace 中的失败模式并提出修复 PR。学习阶段可以不接，生产阶段强烈建议。

判断口诀：**调模型找 LangChain，管流程找 LangGraph，要现成 Agent 找
Deep Agents，看与评找 LangSmith。**

#### 0.7.2 Pregel 渊源：为什么图执行是"Super-step"模型

LangGraph 的设计灵感来自 Google Pregel 论文与 Apache Beam（执行模型），
公共接口借鉴 NetworkX（所以叫 `StateGraph`、`add_node`、`add_edge`）。
这不是历史趣闻，它直接解释了 LangGraph 的几个核心行为：

1. **Super-step（超步）**：图执行被切成一个个超步。每个超步内，所有被激活的
   节点**并行**收到上一超步的状态副本，各自计算并产出写入；超步结束时，
   所有写入按 Reducer 合并，形成新状态，进入下一个超步。第 3 章的并行分支
   就是同一超步内的多个节点。
2. **消息传递与隔离**：节点读到的状态是本超步开始时的快照，看不到同超步
   兄弟节点的写入——这就是"为什么两个并行节点不能随意写同一个普通字段"的
   根源（没有 Reducer 就无法合并冲突写入，直接报
   `InvalidUpdateError`）。
3. **Checkpoint 的天然切分点**：Super-step 边界就是状态一致点，所以
   Checkpoint 按 Super-step 保存是自然设计——每个超步结束时状态是完整、
   可恢复的。

带着这个模型去看第 2 章的 Reducer、第 3 章的并行、第 6 章的持久化，会发现
它们是同一个执行模型在不同侧面的体现。

#### 0.7.3 耐久模式（durability）：Checkpoint 什么时候落盘

`invoke` / `stream` 支持 `durability` 参数，控制 Checkpoint 的写入时机。
三种模式的语义差异用"进程在第 N 个 Super-step 中途崩溃"来理解最直观：

| 模式 | 写入时机 | 崩溃后能恢复到哪 | 性能 | 官方建议 |
| --- | --- | --- | --- | --- |
| `exit` | 仅在整次 run 结束时 | 只能从头重跑，**无法从中间恢复** | 最好（写的次数最少） | 关键任务避免 |
| `async` | 每个 Super-step 结束后异步写 | 上一个已写完的超步（可能丢最后一步） | 好 | **默认推荐** |
| `sync` | 每个 Super-step 同步写盘后才继续 | 上一个超步，零丢失 | 略损（写盘阻塞主流程） | 高一致性要求（金融、审批链） |

```python
# 典型用法：同一个图，不同场景选择不同耐久级别
graph.invoke(inputs, config, durability="sync")    # 涉及资金/审批的图
graph.invoke(inputs, config, durability="async")   # 普通对话/研究类图
graph.invoke(inputs, config, durability="exit")    # 一次性批处理，失败可整批重来
```

两个细节：

- `durability` 是**每次调用可选**的，不是编译期固定的；你可以只在特定
  `invoke` 上收紧耐久级别；
- HITL 依赖 Checkpoint 恢复，因此含 `interrupt()` 的图不要用 `exit`——
  中断本身会强制落盘一次，但崩溃与恢复语义在 `exit` 下最弱。

详见第 6 章 6.6.1 与第 7 章 7.9.1 的深入补充。

#### 0.7.4 Agent Server 提示：部署后哪些代码要换

若使用 LangChain 的 Agent Server 部署（`langgraph.json` + `langgraph dev/up`），
Checkpointer、Store 等持久化基础设施由服务端自动处理，**无需在代码里手动
配置**——这也是为什么本仓库示例都要手动传 `InMemorySaver`，而部署版不用。
对应地，进程内的 `stream()` / `stream_events()` 不再是正确的消费入口，
应改用 LangSmith Streaming API（HTTP/SSE 协议消费服务端推来的事件流）。
写代码时的判断标准：**你的进程负责执行图，就用进程内 API；图跑在服务端，
就用服务端 API。**

---

## 第 1 章 入门：从最小状态图到两种 API

对应目录：`c01_*`，共 3 个示例。

### 1.1 本章目标

学完本章，你应该能够：

- 解释 LangGraph 不是模型 SDK，而是状态化编排 Runtime；
- 写出最小的 `StateGraph`；
- 说清 `START`、节点、边、`END` 和 `compile()` 的关系；
- 判断一个问题更适合 Graph API 还是 Functional API；
- 使用离线 FakeModel 避免学习阶段产生 API 成本。

官方阅读顺序：

1. [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)
2. [Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart)
3. [Thinking in LangGraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph)
4. [Choosing between Graph and Functional APIs](https://docs.langchain.com/oss/python/langgraph/choosing-apis)

### 1.2 `c01_01_env_check.py`：确认学习环境

运行：

```bash
venv/bin/python langgraph-demo/c01_01_env_check.py --offline
```

**学习目标：** 分清 LangGraph、Checkpoint、LangChain 和模型适配包的职责。

**执行链路：** 读取 Python 与包版本 -> 检查 DeepSeek 配置是否存在 ->
输出环境结果。脚本只显示 Key 是 `configured` 还是 `missing`，不会打印 Key。

**关键 API：** `importlib.metadata.version()`、环境变量、统一 CLI 参数。

**需要理解：**

- `langgraph` 提供图编排 Runtime；
- `langgraph-checkpoint` 提供状态保存接口与内存实现；
- `langchain-openai` 是模型适配层，`base_url` 指向 DeepSeek；
- `--offline` 既不要求 Key，也不访问网络。

**常见误区：** 不要通过 `langgraph.__version__` 判断版本。发行包元数据才是稳定
来源。

**练习：** 增加 `LANGSMITH_TRACING` 和 `LANGSMITH_PROJECT` 的状态显示，但不得
输出任何密钥。

### 1.3 `c01_02_hello_state_graph.py`：最小 StateGraph

运行：

```bash
venv/bin/python langgraph-demo/c01_02_hello_state_graph.py --offline
```

**目标：** 建立 `State -> Node -> Edge -> Compile -> Invoke` 的最小闭环。

**执行链路：**

```text
{"name": "LangGraph"}
        |
        v
START -> greet -> END
        |
        v
{"name": "...", "message": "hello LangGraph"}
```

**关键代码：**

- `class HelloState(TypedDict)` 定义状态字段；
- `greet_node(state)` 读取 `name`，只返回 `{"message": ...}`；
- `StateGraph(HelloState)` 创建 Builder；
- `add_node()` 注册节点；
- `add_edge(START, "greet")` 和 `add_edge("greet", END)` 定义路径；
- `compile()` 校验并生成可执行图；
- `invoke()` 传入初始状态并等待最终状态。

**必须分清：** Builder 是“图定义”，Compiled Graph 是“可执行对象”。节点返回值
是状态增量，不需要复制并返回整个 State。

**练习：** 增加 `uppercase` 节点，把 `message` 转成大写，再接回 `END`。

### 1.4 `c01_03_graph_vs_functional.py`：两套 API 做同一件事

运行：

```bash
venv/bin/python langgraph-demo/c01_03_graph_vs_functional.py --offline
```

**Graph 版本：** 两个节点通过显式边串联，输出带有 `[graph]` 后缀。

**Functional 版本：** `@task` 定义可重放执行单元，`@entrypoint` 定义工作流边界，
用普通函数调用和 `.result()` 组织流程，输出带有 `[functional]` 后缀。

**关键 API：**

- `StateGraph`、`add_edge`、`compile`、`invoke`；
- `@entrypoint()`、`@task`、`.result()`、`.invoke()`。

**选择时看这些信号：**

- 需要共享状态、复杂分支和并行 join：优先 Graph；
- 需要把已有 Python 流程最小改造为可持久化工作流：优先 Functional；
- 两者可以互调，不需要二选一。

**常见误区：** Functional API 不等于“没有状态”。`@entrypoint` 的输入输出仍会
被持久化，任务结果也可能被 Checkpoint 复用。

**练习：** 给两套实现都增加失败重试，并比较代码结构差异。

### 1.5 本章总结

LangGraph 的入门难点不是 API 数量，而是把业务翻译成：

```text
哪些数据要共享 -> 每个处理步骤是什么 -> 下一步由谁决定 -> 如何结束
```

### 1.6 章末自测

1. 为什么节点只返回增量就够了？
2. `compile()` 之前和之后的对象有什么区别？
3. Functional API 为什么也不能忽略幂等性？
4. 一个需要三条并行分支再汇总的流程，优先考虑哪套 API，为什么？

---

## 第 2 章 State 与 Graph 基础

对应目录：`c02_*`，共 5 个示例。

### 2.1 本章目标

- 区分 TypedDict、Pydantic State 和 `MessagesState`；
- 理解“字段更新经过 Channel，再由 Reducer 合并”；
- 控制图输入、图输出和节点私有状态；
- 使用普通边与条件边表达基础控制流；
- 编译并可视化图结构。

官方依据：

- [Graph API overview](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [Use the graph API](https://docs.langchain.com/oss/python/langgraph/use-graph-api)

### 2.2 State 的本质

State 是图中的共享数据契约。每个字段都可以理解为一条 Channel：

```text
节点返回更新 -> 写入对应 Channel -> Reducer 合并 -> 新 State
```

没有 Reducer 的字段通常采用 last-value 语义，也就是后写覆盖。并行写入多个节点
时，同一无 Reducer 字段发生冲突会触发并发更新错误。

选择 Schema：

| 类型 | 优点 | 代价 | 适合场景 |
| --- | --- | --- | --- |
| `TypedDict` | 轻量、直观、启动成本低 | 运行期校验有限 | 大多数教学和内部工作流 |
| Pydantic `BaseModel` | 字段校验、默认值、类型约束 | 序列化和结构约束更多 | 外部输入、严格数据契约 |
| `MessagesState` | 内置消息 Reducer，适合聊天 | 只解决消息相关状态 | Chatbot、Tool Agent |

### 2.3 `c02_01_state_schemas.py`：三种 State

运行：

```bash
venv/bin/python langgraph-demo/c02_01_state_schemas.py --offline
```

**执行内容：** 分别构建 TypedDict、Pydantic 和 `MessagesState` 三个最小图。

**关键 API：** `TypedDict`、`BaseModel`、`MessagesState`、`Annotated`。

**观察重点：**

- TypedDict 示例用 `operator.add` 累积 `history`；
- Pydantic 示例可读取 `state.number`，同时接受字典输入；
- `MessagesState` 自动使用消息合并规则，输出包含用户和 AI 消息。

**常见误区：** 聊天历史不能简单用普通列表替代。消息具有 ID、角色、Tool Call、
删除等语义，应该使用 `add_messages` 或 `MessagesState`。

**练习：** 给 Pydantic State 增加数值范围校验，并观察非法输入在哪一层失败。

### 2.4 `c02_02_reducers_messages.py`：Reducer 决定如何合并

运行：

```bash
venv/bin/python langgraph-demo/c02_02_reducers_messages.py --offline
```

**目标：** 对比默认覆盖、数值累加和消息合并。

**典型定义：**

```python
class CounterState(TypedDict):
    count: Annotated[int, operator.add]
```

节点每次返回 `{"count": 1}`，Reducer 会把新值加到旧值，而不是覆盖旧值。

**关键认识：**

- Reducer 在字段级别生效，不在整个 State 级别生效；
- 并行节点写同一字段时，Reducer 决定能否安全合并；
- 返回空列表并不意味着清空已有值，具体行为由 Reducer 决定；
- 需要强制覆盖时，官方提供 `Overwrite`，但应明确理解其影响。

**常见误区：** `operator.add` 适合列表拼接，但消息列表仍应优先使用
`add_messages`，因为它能处理消息 ID 和删除操作。

**练习：** 增加 `RemoveMessage`，删除一条历史消息并验证 Checkpoint 中的结果。

### 2.5 `c02_03_input_output_private_state.py`：公开状态与私有状态

运行：

```bash
venv/bin/python langgraph-demo/c02_03_input_output_private_state.py --offline
```

**执行链路：**

```text
InputState(question)
      -> prepare
PrivateState(normalized)
      -> answer
OutputState(answer)
```

**关键 API：** `StateGraph(..., input_schema=..., output_schema=...)`。

**为什么要这样做：**

- 输入 Schema 限制调用者能提供什么；
- 输出 Schema 防止草稿、Token、内部索引等字段泄漏；
- 节点的函数参数类型决定它能读取哪些字段；
- 私有状态可以只在相邻节点间传递，不必进入公开 Overall State。

**常见误区：** “图总 Schema 里没有这个字段”不表示节点绝对不能传递私有数据；
但节点必须通过参数类型明确声明它需要什么。

**练习：** 增加 `audit_id` 作为内部字段，让它参与流程但不出现在最终输出。

### 2.6 `c02_04_nodes_edges.py`：节点和边

运行：

```bash
venv/bin/python langgraph-demo/c02_04_nodes_edges.py --offline
```

**执行链路：**

```text
START -> classify -> short -> END
                  \-> long  -> END
```

**关键 API：**

- `add_node()`：注册工作单元；
- `add_edge()`：固定顺序；
- `add_conditional_edges()`：由路由函数动态选择；
- `START` / `END`：入口和出口常量。

路由函数只应返回路径值，例如 `"short"` 或 `"long"`。状态修改应在节点中完成，
这样更容易测试和推理。

**常见误区：**

- 路由值没有出现在映射中；
- 节点返回类型与 State 字段不匹配；
- 忘记给所有分支连接到 `END`；
- 把大量业务逻辑塞进路由函数，导致路由函数有副作用。

**练习：** 增加 `medium` 分支，并编写三个输入分别覆盖三个分支。

### 2.7 `c02_05_compile_visualize.py`：编译与可视化

运行：

```bash
venv/bin/python langgraph-demo/c02_05_compile_visualize.py --offline
venv/bin/python langgraph-demo/c02_05_compile_visualize.py --offline --png
```

**目标：** 先用 Mermaid 观察图结构，再执行图。

**关键 API：**

- `graph.get_graph().draw_mermaid()`；
- 可选 `draw_mermaid_png()`。

**为什么先可视化：** 边写错时，代码看似正常，但数据会走到错误节点。Mermaid 能
快速检查入口、分支、回边和出口。

**常见误区：** PNG 渲染可能依赖外部服务或系统工具，失败不代表图本身不可运行。

**练习：** 为第 3 章的条件路由图输出 Mermaid，并标出所有可能终点。

### 2.7.1 深入补充：Reducer 边界、私有通道与节点的边界行为

#### 2.7.1.1 用 `Overwrite` 清空 Reducer 字段

合并型 Reducer 有一个反直觉行为：字段一旦定义了 Reducer，**所有**写入都会走
合并逻辑，包括"空值"。节点返回 `{"errors": []}` 时，`operator.add` 会把空
列表与旧值相加，结果等于旧值原样保留——你的"清空"什么都没做。

```python
from langgraph.types import Overwrite

def clear_errors(state: State):
    return {"errors": Overwrite([])}  # 真正清空：绕过 Reducer，直接赋值
```

`Overwrite` 的语义是"跳过该字段的 Reducer，直接把我给的值作为新值"。它是
一种显式的、可读性强的逃生通道，图的其他读取者能在代码里一眼看出"这里
故意不走合并"。

判断标准：**只要字段带 Reducer，"写入空值"和"清空字段"就是两件事。**
前者参与合并（往往无效），后者用 `Overwrite`。

#### 2.7.1.2 私有通道会在 values 流中泄漏

`input_schema` / `output_schema` 只约束 `invoke()` 的输入输出**契约**，
不约束流式输出：`stream_mode="values"` 默认发出**所有**状态通道，包括你
不想暴露给前端的私有字段（中间草稿、内部计数器、调试数据）。这是安全与
协议设计的常见盲区。两种处理方式：

```python
# 方式一：限制输出通道——只在 values 流中发出指定字段
stream = graph.stream_events(inputs, version="v3", output_keys=["graph_output"])

# 方式二：改用 updates 模式——只发节点真正返回的字段，天然不含未触碰的通道
for chunk in graph.stream(inputs, stream_mode="updates"):
    ...
```

选择建议：对外 API（给前端/客户端消费）用 `output_keys` 白名单，明确可控；
内部调试用 `updates`，信息量够且不需要维护白名单。

#### 2.7.1.3 Pydantic State 的限制

Pydantic `BaseModel` 可以做 State，好处是每个 Super-step 结束都会做字段
校验（类型错误早暴露）。但有三个限制要知道：

1. **性能开销**：每步都要实例化模型对象并校验，序列化也比 TypedDict/dataclass
   慢，在高频循环图中可感知；
2. **高层工厂不支持**：LangChain 的 `create_agent` 等**预构建工厂不支持**
   Pydantic State；
3. **Reducer 字段写法受限**：`Annotated` 合并与 Pydantic 校验器的交互有
   边界情况，复杂 Reducer 场景建议用 TypedDict。

结论：仅在外部输入需要强校验的场景使用 Pydantic State，内部流程状态用
TypedDict 或 dataclass。

#### 2.7.1.4 节点重执行与幂等

配置 Checkpointer 后，有三种场景会**重跑整个节点函数**（从函数开头执行，
不是从暂停行继续）：中断恢复（resume）、重试（RetryPolicy 触发）、
Replay/Time-travel（从历史 Checkpoint 重放）。

这对节点内部的副作用代码是硬约束：

```python
def charge_card(state: OrderState):
    # 危险：恢复/重试时会再次扣款
    payment_api.charge(state["order_id"], state["amount"])
    return {"charged": True}
```

对策二选一：副作用操作本身幂等（用业务 ID 去重，服务端拒绝重复扣款）；或
把副作用拆进 `@task`——Task 的结果会单独持久化，重跑节点时已完成的 Task
直接从 Checkpoint 取结果，不会重复执行。

#### 2.7.1.5 节点内使用 @task

节点可以把重操作拆成 Task，获得两个能力：**独立检查点**（重跑节点时已完成的
Task 不重复计算）与**并行**（多个 Task 先建 Future 再统一收割）：

```python
@task
def _fetch(url: str) -> str:
    return requests.get(url).text[:100]

def call_api(state: State):
    futures = [_fetch(url) for url in state["urls"]]  # 先建全部 Future
    return {"results": [f.result() for f in futures]}  # 实际并发执行
```

注意 Future 必须先全部创建再收割——如果"创建一个、立刻 result() 一个"，
就退化成串行了。这是 Python 同步 Future 的惯用法，在 LangGraph 里同样适用。

#### 2.7.1.6 节点缓存（CachePolicy）

输入相同且计算昂贵的**纯节点**可以缓存结果：

```python
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy

builder.add_node("expensive_node", expensive_node, cache_policy=CachePolicy(ttl=3))
graph = builder.compile(cache=InMemoryCache())
```

缓存键由节点输入状态计算，`ttl` 是存活秒数。适用场景：确定性翻译、嵌入、
格式转换等"同输入必然同输出"的重计算。前提仍然是**无外部副作用**——有副作用
的节点被缓存命中等于跳过副作用，这是隐蔽的 bug 来源。

**练习：** 写一个带 `errors: Annotated[list, operator.add]` 的图，分别验证
返回 `[]`、`Overwrite([])`、不返回三种情况在 Checkpoint 中的结果差异。

### 2.8 本章总结

一个成熟的 Graph 定义应同时回答：

1. 状态字段是什么类型？
2. 每个字段允许多个节点同时更新吗？
3. 路由判断依据什么字段？
4. 哪些字段只是内部实现细节？
5. 图的实际路径与设计图是否一致？

### 2.9 章末自测

1. 为什么两个并行节点不能随意写同一个普通字符串字段？
2. `MessagesState` 相比 `list` 多解决了哪些问题？
3. `input_schema`、`output_schema` 和节点参数类型各自控制什么？
4. 条件路由函数为什么应尽量保持无副作用？

---

## 第 3 章 高级图控制流

对应目录：`c03_*`，共 6 个示例。

### 3.1 本章目标

- 设计 fan-out / fan-in 并行结构；
- 使用条件边表达动态路由；
- 给循环设置业务终止条件；
- 用 `Send` 动态创建未知数量的 Worker；
- 用 `Command` 同时更新状态和选择下一跳；
- 配置 Runtime Context、Retry、Timeout 和 Cache。

官方依据：

- [Use the graph API](https://docs.langchain.com/oss/python/langgraph/use-graph-api)
- [Graph API overview](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [Fault tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance)

### 3.2 静态并行：`c03_01_branch_parallel.py`

运行：

```bash
venv/bin/python langgraph-demo/c03_01_branch_parallel.py --offline
```

**执行结构：**

```text
        -> left  ->
START             END
        -> right ->
```

**核心条件：** 两个分支都写 `branches`，所以字段必须是
`Annotated[list[str], operator.add]`。

**需要理解：**

- `START` 同时连接两个节点时形成 fan-out；
- 两个节点可进入同一 Super-step；
- 两个分支都到 `END` 后图才完成；
- 并发结果顺序不应作为业务契约，除非显式排序。

**练习：** 增加第三个分支和 `merge` 节点，在汇总前显式排序。

### 3.3 条件路由：`c03_02_conditional_routing.py`

运行：

```bash
venv/bin/python langgraph-demo/c03_02_conditional_routing.py --offline --value 3
venv/bin/python langgraph-demo/c03_02_conditional_routing.py --offline --value 8
```

**执行链路：** `classify` 写入分类 -> 条件边读取分类 -> 选择 `small` 或 `large`。

**关键 API：** `add_conditional_edges()`、路径映射、`Literal`。

**设计原则：**

- 路由函数应唯一、确定、易测试；
- 映射表让合法路径显式可见；
- 未知路由值应尽早失败，不能悄悄走默认分支；
- 生产 Agent 应记录路由决策，便于审计和评估。

**练习：** 增加 `other` 兜底路由，并为每个路由写一个单元测试。

### 3.4 有界循环：`c03_03_loops_recursion.py`

运行：

```bash
venv/bin/python langgraph-demo/c03_03_loops_recursion.py --offline --limit 3
```

**执行链路：** `increment` 更新 `count` -> 条件边判断是否继续回到自身 -> 到 `END`。

**关键 API：** 条件回边、`recursion_limit`。

**必须区分：**

- 业务终止条件：由 State 决定，例如 `count >= limit`；
- Runtime 保护：递归上限，用于防止程序失控。

递归上限不是业务逻辑。不能依赖它作为正常退出条件，否则达到上限会抛错。

**练习：** 增加 `remaining_steps` 检查，在预算不足时优雅结束而不是抛异常。

### 3.5 动态 Map-Reduce：`c03_04_send_map_reduce.py`

运行：

```bash
venv/bin/python langgraph-demo/c03_04_send_map_reduce.py --offline
```

**执行结构：**

```text
fan_out -> Send(worker, item_1) -> results += ...
       -> Send(worker, item_2) -> results += ...
       -> Send(worker, item_N) -> results += ...
```

**关键 API：** `Send(node, payload)`、条件边返回 `list[Send]`、列表 Reducer。

**什么时候使用 `Send`：**

- 编译图时还不知道 Worker 数量；
- 每个 Worker 需要独立输入；
- 输出要在父 Graph 中聚合。

**什么时候不要使用：**

- 分支数量固定，普通并行边更清晰；
- 每个任务依赖前一个任务的输出；
- Worker 数量可能无限制增长，必须增加上限。

**练习：** 给每个任务加 ID，限制最多 5 个 Worker，并按 ID 排序最终结果。

### 3.6 更新与跳转合并：`c03_05_command.py`

运行：

```bash
venv/bin/python langgraph-demo/c03_05_command.py --offline
```

**节点返回：** `Command(update={...}, goto="next_node")`。

**适用场景：**

- 工具调用后需要写状态并选择下一节点；
- Handoff 逻辑需要在同一步完成更新和转移；
- 图结构无法表达全部动态跳转时。

**重要区别：**

- 节点返回 `Command(update=..., goto=...)`：用于控制图执行；
- 调用图时传 `Command(resume=...)`：用于恢复 `interrupt()`；
- 多轮对话应继续传普通输入 State，而不是 `Command(update=...)`。

**练习：** 让 `first` 根据 State 选择 `second` 或 `fallback`。

### 3.7 Runtime 与可靠性策略：`c03_06_runtime_retry_timeout_cache.py`

运行：

```bash
venv/bin/python langgraph-demo/c03_06_runtime_retry_timeout_cache.py --offline
```

**四种能力：**

| 能力 | 作用 | 示例配置 |
| --- | --- | --- |
| Runtime Context | 注入请求级配置，不污染图状态 | `Runtime[DemoContext]` |
| Retry | 匹配错误后重新执行节点 | `RetryPolicy(max_attempts=2)` |
| Timeout | 限制单次节点尝试时长 | `timeout=1.0` |
| Cache | 复用相同输入的节点结果 | `CachePolicy(ttl=60)` |

**执行顺序：** 节点尝试 -> 失败 -> Retry 判断 -> 重试 -> 成功。重试耗尽后才会
进入 Error Handler。

**重点限制：** 同步节点不支持可取消 Timeout，带 Timeout 的同步节点会在编译时
被拒绝。阻塞 I/O 应放进 async 节点，必要时使用 `asyncio.to_thread()`。

**缓存风险：** 有外部副作用的节点不能在未理解业务幂等性时随意缓存。输入相同
不代表“下单、扣款、发邮件”可以只执行一次。

**练习：** 使用 `TimeoutPolicy` 分别配置 `run_timeout` 和 `idle_timeout`，
并让节点在长任务中发送 heartbeat。

### 3.7.1 深入补充：Runtime 注入、递归保护与图迁移

#### 3.7.1.1 Runtime 对象：运行时上下文的统一入口

节点函数声明 `runtime: Runtime[Context]` 参数即可获得注入。它的意义在于
**把"每次请求才会知道的信息"与"编译期固定的图定义"分开**——图是编译期
定义的，但 user_id、租户、审批人这些信息只有调用时才有。Runtime 的完整属性：

| 属性 | 用途 | 典型场景 |
| --- | --- | --- |
| `runtime.context` | `context_schema` 定义的请求级配置 | user_id、租户 ID、语言偏好、审批人 |
| `runtime.store` | 长期记忆 Store（跨 thread） | 在节点里读写用户画像、历史事实 |
| `runtime.execution_info` | 当前尝试次数 `node_attempt`、首次尝试时间、thread/run/checkpoint ID | 重试降级（第几次失败走备用 API）、追踪日志 |
| `runtime.heartbeat()` | 手动刷新 Idle Timeout 时钟 | 长批处理任务每批调一次，防止被空闲超时杀掉 |
| `runtime.drain_requested` | 是否已请求优雅排空 | 节点提前感知停机信号，跳过新工作返回收尾结果 |

Context 的完整传递链路：编译时声明 schema，调用时传入实例，节点里读取——
**不进入图状态**，因此不出现在 Checkpoint 里、不参与 Reducer 合并、也不会
被流式输出泄漏：

```python
from dataclasses import dataclass

@dataclass
class Context:
    user_id: str
    language: str = "zh"

graph = builder.compile(context_schema=Context)          # 1. 编译时声明
result = graph.invoke(inputs, context=Context(user_id="1"))  # 2. 调用时传入

def my_node(state: State, runtime: Runtime[Context]):    # 3. 节点里读取
    uid = runtime.context.user_id
```

与放进 State 的对比：State 适合"影响图执行结果"的数据（会被持久化、可
Time-travel）；Context 适合"执行环境"数据（每次调用可不同，不需要历史）。

#### 3.7.1.2 递归上限的三个细节

1. **位置**：`recursion_limit` 是 `config` 的**顶级键**，不是
   `configurable` 内的键。写错位置不会报错但也不生效，这是高频踩坑点：

   ```python
   config = {"recursion_limit": 5, "configurable": {"thread_id": "1"}}  # ✅
   # 错误写法：{"configurable": {"thread_id": "1", "recursion_limit": 5}}
   ```

2. **默认值**：v1.0.6 起默认 1000（更早版本是 25）。循环图如果依赖默认值
   兜底，超限时抛 `GRAPH_RECURSION_LIMIT` 错误，已完成的步骤仍保留在
   Checkpoint 里可回溯。
3. **主动防护用托管值 `RemainingSteps`**：被动超限抛错是最后防线，优雅的
   做法是在节点里读剩余预算、提前收尾：

   ```python
   from langgraph.managed import RemainingSteps

   class State(TypedDict):
       messages: Annotated[list, add_messages]
       remaining_steps: RemainingSteps  # 框架托管，业务代码只读

   def reasoning_node(state: State):
       if state["remaining_steps"] <= 2:
           return {"messages": ["预算即将耗尽，收尾中..."]}  # 主动走收尾分支
   ```

`RemainingSteps` 是托管字段——你只声明类型，框架每步自动写入剩余步数，
不需要（也不能）手动更新。当前步数也可通过
`config["metadata"]["langgraph_step"]` 读取。**主动（预算检查）与被动
（超限抛错）两种方式都应实现**：前者保证优雅收尾，后者是最后防线。

#### 3.7.1.3 子图导航父图：`Command(goto=..., graph=Command.PARENT)`

子图节点可以跳转**父图**的节点，典型用途：子图内部发生不可恢复错误时，
直接把控制权交回父图的兜底节点，而不用让子图把错误再走完。前提条件：
父图 State 对所有被更新的共享键**定义了 Reducer**——因为父图在收到跳转的
同时，可能还有其他并行节点在写同一批键，没有 Reducer 就无法合并，触发并发
更新错误。子图与父图的 State schema 不需要一致，但被 `goto` 携带的更新键
必须存在于父图 State。

#### 3.7.1.4 图迁移（Graph Migrations）：改图时旧线程怎么办

Checkpoint 中保存的是**通道名与拓扑信息**，不是 Python 代码引用。因此改图
代码时，旧线程能否继续恢复取决于改动是否触碰了 Checkpoint 里记录的东西：

- 线程**未中断**：可以随意修改拓扑（增删节点、改边）——新调用从新图定义
  开始执行，旧 Checkpoint 只是历史；
- 线程**中断中**：不能重命名或删除节点——Checkpoint 记录了"待执行的节点名"，
  新图里找不到这个名字就无法恢复；
- **新增 State 键**是安全的（旧线程里新键为默认值）；
- **重命名键**会导致旧线程中该字段的值丢失（Checkpoint 里存的旧名字对不上
  新 schema）；
- **类型不兼容的修改**（如 `list` 改 `dict`）可能导致旧线程反序列化报错。

生产建议：改动拓扑/键名前，先把活跃线程跑完或显式作废；schema 演进尽量
只做"加键"，不做"改名/改类型"。

#### 3.7.1.5 统一节点默认策略：set_node_defaults

```python
builder.set_node_defaults(
    retry_policy=RetryPolicy(max_attempts=2),
    error_handler=my_handler,
    timeout=TimeoutPolicy(idle_timeout=30),
    cache_policy=CachePolicy(ttl=60),
)
```

一次性给全图节点配置默认值，避免每个 `add_node` 重复传参。两条优先级规则：
`add_node` 里的**显式参数优先级更高**（可对个别节点覆盖默认值）；默认值
**不**被子图继承——子图要同样的默认策略需自行设置。

**练习：** 给一个循环图加 `remaining_steps`，把 `recursion_limit` 降到 5，
分别验证"优雅收尾"与"抛出 GRAPH_RECURSION_LIMIT"两种结局。

### 3.8 本章总结

控制流设计的核心顺序是：

1. 固定顺序能解决，就用普通边；
2. 有分支条件，用条件边；
3. 分支固定且并行，用 fan-out / fan-in；
4. 运行期才知道工作项数量，用 `Send`；
5. 更新状态与跳转必须原子表达，用 `Command`；
6. 可靠性和资源边界交给 Runtime 策略。

### 3.9 章末自测

1. 递归限制为什么不能替代业务终止条件？
2. `Send` 和普通并行边的关键区别是什么？
3. 节点返回的 `Command` 与输入图的 `Command(resume=...)` 有什么区别？
4. Retry、Timeout 和 Error Handler 的执行顺序是什么？

---

## 第 4 章 Functional API

对应目录：`c04_*`，共 4 个示例。

### 4.1 本章目标

- 使用 `@entrypoint` 定义工作流边界；
- 使用 `@task` 定义可持久化、可重试的执行单元；
- 同时创建 Future 进行并行调用；
- 理解 Functional API 的重放、幂等和恢复语义；
- 与 Graph API 双向互调。

官方依据：

- [Functional API overview](https://docs.langchain.com/oss/python/langgraph/functional-api)
- [Use the functional API](https://docs.langchain.com/oss/python/langgraph/use-functional-api)
- [Choosing between Graph and Functional APIs](https://docs.langchain.com/oss/python/langgraph/choosing-apis)

### 4.2 Entrypoint 与 Task 的职责边界

`@entrypoint`：

- 是工作流入口；
- 只能接收一个位置参数，多个值应放进字典；
- 管理执行流、长任务和中断；
- 输入输出需要可 JSON 序列化。

`@task`：

- 是一个离散、可重放的执行单元；
- 调用后返回 Future；
- 适合 API 调用、数据加工和需要独立重试的步骤；
- 结果可能被 Checkpoint 复用。

不是每个普通函数都必须变成 Task。只有需要持久化、重试、并行调度或隔离副作用的
边界才值得引入。

### 4.3 `c04_01_entrypoint_task.py`：最小函数式工作流

运行：

```bash
venv/bin/python langgraph-demo/c04_01_entrypoint_task.py --offline --value 21
```

**执行链路：** `entrypoint` 调用 `double_task` -> 用 `.result()` 取值 -> 返回结果。

**关键 API：** `@entrypoint()`、`@task`、`.result()`、`.invoke()`。

**为什么仍要确定性：** 工作流主体可能重放。条件判断、调度和输入构造应尽量由
已有状态和确定逻辑推导，不能随意依赖当前时间或随机数。

**练习：** 增加第二个 Task，并让两个 Task 的输入输出契约写成明确类型。

### 4.4 `c04_02_futures_parallel.py`：Future 并行

运行：

```bash
venv/bin/python langgraph-demo/c04_02_futures_parallel.py --offline --value 4
```

**正确模式：**

```python
square_future = square(value)
double_future = double(value)
return square_future.result() + double_future.result()
```

先创建全部 Future，再读取结果，任务才有机会并行。

**错误模式：** 创建 Future 后立刻调用 `.result()`，会退化为顺序等待。

**注意：**

- 并行提高 I/O 密集任务吞吐，但会放大模型成本和限流风险；
- 某个 Future 失败时，要明确是整体失败、降级还是部分成功；
- 结果顺序应与业务顺序显式绑定，不能依赖完成先后。

**练习：** 增加第三个并行任务，并让其中一个失败后返回 fallback。

### 4.5 `c04_03_retry_resume.py`：Task 重试与恢复

运行：

```bash
venv/bin/python langgraph-demo/c04_03_retry_resume.py --offline --value 7
```

**目标：** 第一次业务失败后，通过 `RetryPolicy` 再执行一次。

**关键 API：**

- `@task(retry_policy=...)`；
- `RetryPolicy(max_attempts=...)`；
- `retry_on=ValueError`；
- `initial_interval` 与 `jitter`。

**重要规则：** 重放会再次执行函数体，因此外部副作用必须幂等。典型风险包括
重复下单、重复扣款、重复发送邮件和重复写文件。

**练习：** 把尝试次数存进外部数据库，验证重试不会重复扣款。

### 4.6 `c04_04_graph_functional_interop.py`：两种 API 互调

运行：

```bash
venv/bin/python langgraph-demo/c04_04_graph_functional_interop.py --offline
```

**两条路径：**

1. Functional Workflow 内部调用编译 Graph；
2. Graph 节点内部调用 Functional Entrypoint。

**必须检查：** 输入输出 Schema、序列化边界、Checkpointer 继承和异常传播。
不要假设对象永远停留在同一进程内存中。

**练习：** 在 Graph 的两个不同节点中调用同一个 Entrypoint，并比较配置传递。

### 4.6.1 深入补充：Functional API 的注入参数、重放语义与陷阱

#### 4.6.1.1 可注入参数：按名字与类型注解自动识别

`@entrypoint` 函数按**参数名和类型注解**自动注入运行时能力——你声明哪个
参数，框架就注入哪个，全部可选：

```python
@entrypoint(checkpointer=checkpointer, store=store)
def workflow(inputs: dict, *, previous: Any, store: BaseStore,
             writer: StreamWriter, config: RunnableConfig):
    ...
```

| 注入参数 | 用途 | 细节 |
| --- | --- | --- |
| `previous` | 同一 thread 上一次保存的值（短期记忆） | 首次调用为 `None`（或参数默认值）；值来自上次 `entrypoint.final` 的 `save`（或默认的函数返回值） |
| `store` | 跨 thread 的长期记忆 | 需要 `@entrypoint(store=...)` 声明；接口与第 6 章的 Store 一致 |
| `writer` | 流式写入 | 写入的数据由 `stream_mode="custom"` 消费；Python<3.11 必须用参数注入而非 `get_stream_writer()` |
| `config` | 运行时配置 | 读取 `thread_id`、tags、metadata 等 |

`previous` 直接实现了跨调用累加：调用 1 输入 1 返回 1，调用 2 输入 2 返回
3——前提是两次 `invoke` 传入**相同 `thread_id`**。换 thread 就换记忆，
这也是"短期记忆按 thread 隔离"的最小演示。

#### 4.6.1.2 `entrypoint.final`：分离"返回值"与"保存值"

默认情况下，entrypoint 的返回值既给调用方、又存进 Checkpoint 作为下次的
`previous`。但很多场景两者应该不同——比如"返回累计值、保存本次增量"：

```python
@entrypoint(checkpointer=saver)
def doubler(number: int, *, previous: Any = None) -> entrypoint.final[int, int]:
    previous_val = previous or 0
    return entrypoint.final(value=previous_val + number, save=number)
```

类型注解 `entrypoint.final[返回类型, 保存类型]` 标注两个类型的顺序：
`value` 返回给调用方（上例：累计和），`save` 写入 Checkpoint 供下次
`previous` 使用（上例：本次输入）。反过来"返回增量、保存全量"同样只是
调换两个字段。

#### 4.6.1.3 重放语义：最容易误解的一点

恢复（resume）/重试（retry）时，工作流**从 `entrypoint` 开头重放**，
不是从暂停行继续。这一点与直觉相反，机制是：

- 已完成的 `@task`：结果直接从 Checkpoint 加载，**不重新计算**——Task 是
  Functional API 的持久化与重放单元；
- 未完成的 Task 与 entrypoint 主体代码：**重新执行**。

由此推出两条硬规则：

1. **副作用只能放 `@task` 里**。直接写在 entrypoint 主体里的写文件、发请求，
   重放时会再次执行。主体代码要当成"纯编排逻辑"来写。
2. **控制流必须确定性**。`if time.time() > ...` 这类依赖外部状态的分支要
   包成 `@task`（结果被持久化，重放时取旧值）。否则恢复时分支走向可能改变，
   导致 interrupt 按**索引**匹配时错位。

关于索引匹配要展开说：Functional API 的 `interrupt()` 与 `Command(resume=...)`
**严格按调用顺序索引对应，不按 ID**。如果 entrypoint 里有三个 `interrupt()`
调用，第一个 `resume` 值给第一个调用、第二个给第二个——顺序由代码路径决定。
一旦重放时分支走向改变，某个 interrupt 被跳过或新增，后续所有 resume 值就
整体错位（给审批 2 的值被审批 3 收到）。这是 Functional API 中断最隐蔽的
生产事故来源，确定性的控制流是唯一防线。

#### 4.6.1.4 Task 只能在受限作用域调用

`@task` 只能在 `@entrypoint`、另一个 `@task` 或图节点内部调用——从主程序
直接调用会报错，因为脱离了运行时的持久化上下文。另外，配置 Checkpointer 时，
entrypoint 输入输出与 Task 输出必须 **JSON 可序列化**（它们要写进
Checkpoint），自定义对象需要自己负责序列化方案。

#### 4.6.1.5 错误后恢复

底层错误已修复后（如外部 API 已恢复），用 `None` 作为输入、**相同
`thread_id`** 重新 `invoke` 即可从中断/失败处恢复——框架从最近 Checkpoint
重放，已完成的 Task 直接取结果。注意传 `None` 是"继续执行"的信号，传新
输入则相当于开新调用。

**练习：** 写一个 entrypoint，主体里故意用 `random.random()` 做分支并包含
`interrupt()`，重放两次观察索引错位现象；再把随机数包进 `@task` 验证恢复正确。

**练习：** 写一个 entrypoint，主体里故意用 `random.random()` 做分支并包含
`interrupt()`，重放两次观察索引错位现象；再把随机数包进 `@task` 验证恢复正确。

### 4.7 本章总结

Functional API 的目标是“最小改造既有代码”，代价是控制流主要由 Python 代码
承担，静态可视化能力较弱。它最适合线性流程、简单分支和函数作用域状态。

### 4.8 章末自测

1. 什么情况下普通函数不需要变成 `@task`？
2. 为什么 Entrypoint 的输入输出必须考虑 JSON 序列化？
3. 如何确保 Future 真正并行？
4. 为什么重试前必须先审查副作用？

---

## 第 5 章 工作流与 Agent

对应目录：`c05_*`，共 7 个示例。

### 5.1 本章目标

- 掌握常见 Agentic Workflow 模式；
- 分清 Workflow 的预定路径与 Agent 的动态决策；
- 使用结构化输出、工具调用和 ReAct 循环；
- 为每种模式设置终止条件、预算和失败回退。

官方依据：

- [Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [LangChain tools](https://docs.langchain.com/oss/python/langchain/tools)
- [LangChain structured output](https://docs.langchain.com/oss/python/langchain/structured-output)

### 5.2 Workflow 与 Agent 的区别

| 类型 | 路径是否预先确定 | 典型结构 | 主要风险 |
| --- | --- | --- | --- |
| Workflow | 是 | Chain、Route、Parallel、Orchestrator | 拆分不合理、步骤过多 |
| Agent | 否，由模型动态决定 | 模型-工具循环、Handoff | 不终止、越权、成本失控 |

生产系统通常是混合体：宏观流程由 Workflow 固定，局部决策交给 Agent。

### 5.3 `c05_01_structured_tools_warmup.py`：结构化输出与工具调用

运行：

```bash
venv/bin/python langgraph-demo/c05_01_structured_tools_warmup.py --offline
```

**两个能力：**

- `with_structured_output(RouteSelection, method="function_calling")` 让模型返回
  Pydantic 对象；
- `bind_tools([add])` 让模型生成工具名和参数。

**关键边界：** 模型只生成工具调用请求，真正的 Python 函数由程序或 `ToolNode`
执行。

**DeepSeek 适配提醒：** DeepSeek 当前可能不支持 LangChain 默认的 JSON Schema
`response_format`，因此示例显式使用 `function_calling`。

**安全要求：**

- 校验工具名和参数；
- 对高风险工具做权限和人工审批；
- 对未知工具返回可理解错误；
- 限制工具调用次数和超时。

**练习：** 增加未知工具处理，并给 `add` 工具加参数范围校验。

### 5.4 `c05_02_prompt_chaining.py`：Prompt Chaining

运行：

```bash
venv/bin/python langgraph-demo/c05_02_prompt_chaining.py --offline
```

**执行链路：** `outline -> draft -> polish`。

**价值：** 把复杂任务拆成可验证的小步骤，每一步都有明确输入输出契约，便于定位
错误和插入规则校验。

**适用：** 翻译、摘要、内容生成、格式转换、逐步审核。

**代价：** 延迟和调用次数增加；错误可能逐级放大；每一步都需要维护 Prompt 契约。

**练习：** 在 `outline` 与 `draft` 之间增加结构校验，失败时回到 `outline`。

### 5.5 `c05_03_parallelization.py`：并行化

运行：

```bash
venv/bin/python langgraph-demo/c05_03_parallelization.py --offline
```

**执行结构：** 同一问题并行生成“简答”和“详解”，再一起写进 Reducer 列表。

**适用：** 多视角回答、独立子任务、不同评估标准。

**必须处理：**

- 并发成本叠加；
- Provider 限流；
- 分支失败是否允许部分结果；
- 聚合顺序和来源标记。

**练习：** 为每个分支添加独立 RetryPolicy，并让最终结果携带分支名称。

### 5.6 `c05_04_routing.py`：Routing

运行：

```bash
venv/bin/python langgraph-demo/c05_04_routing.py --offline
```

**执行链路：** 结构化分类 -> 条件边 -> 数学链或通用链。

**优势：** 不同任务使用不同 Prompt、工具和校验规则，比一个万能 Prompt 更可控。

**风险：** 路由分类错误会让后面的处理完全走偏。生产环境应记录分类理由、置信度
和兜底策略。

**练习：** 增加 `other` 分支，并让低置信度请求转人工。

### 5.7 `c05_05_orchestrator_worker.py`：Orchestrator-Worker

运行：

```bash
venv/bin/python langgraph-demo/c05_05_orchestrator_worker.py --offline
```

**执行链路：**

```text
plan -> Planner 生成 subtasks
     -> Send 为每个 subtask 创建 Worker
     -> Reducer 汇总 drafts
```

**适合：** 子任务数量在运行期才能确定，并且子任务可独立执行。

**必须限制：**

- 最大子任务数量；
- 每个 Worker 的超时和重试；
- 总模型调用预算；
- 子任务与最终结果的来源映射。

**练习：** 限制最多 4 个子任务，并记录每个 draft 对应的 subtask。

### 5.8 `c05_06_evaluator_optimizer.py`：Evaluator-Optimizer

运行：

```bash
venv/bin/python langgraph-demo/c05_06_evaluator_optimizer.py --offline
```

**执行链路：** `generate -> evaluate -> PASS/REVISE`。若未通过且未达到最大轮数，
回到 `generate`。

**关键设计：**

- 必须有 `max_rounds`；
- 评估结果应是结构化、明确的；
- 评估器和生成器最好有清晰职责；
- 反馈需要可执行，不能只说“更好一点”。

**风险：** 两个模型互相“认可”可能过早停止，互相“否定”可能耗尽预算。

**练习：** 把反馈改成结构化评分，并设置最低分和最大轮数双条件。

### 5.9 `c05_07_react_tool_agent.py`：最小 ReAct Tool Agent

运行：

```bash
venv/bin/python langgraph-demo/c05_07_react_tool_agent.py --offline
```

**循环：**

```text
agent -> 需要工具 -> tools -> ToolMessage -> agent -> 最终回答
      \-> 不需要工具 -> END
```

**关键 API：** `bind_tools()`、`ToolNode`、`tools_condition`、`MessagesState`。

**终止条件：** `tools_condition` 判断是否存在 Tool Call；没有 Tool Call 时结束。
复杂 Agent 还要限制最大工具轮数和单次工具调用数量。

**常见错误：**

- 工具异常没有回传或处理；
- 模型持续提出工具调用；
- 把危险工具直接暴露给模型；
- 工具执行结果没有结构化来源信息。

**练习：** 增加最大工具轮数和工具异常回传，验证 Agent 能给出降级回答。

### 5.9.1 深入补充：ToolNode、ToolRuntime 与官方 Agent 构建方式

#### 5.9.1.1 ToolNode：官方预构建工具执行节点

相比手写"解析 tool_calls → 逐个调用 → 拼装 ToolMessage"的循环，`ToolNode`
内置了三件事：

1. **并行执行**：模型一次请求多个工具调用时，`ToolNode` 并发执行它们，
   而不是串行等待；
2. **错误处理**：工具抛异常不会让图崩溃，而是把异常包装成
   `ToolMessage(content="Error: ...")` 返回给模型，让模型决定重试还是换路——
   这是 Agent 鲁棒性的关键设计（模型"看得见"错误才能自我纠正）；
3. **状态注入**：配合 `ToolRuntime`，工具内部能拿到图状态与运行时上下文。

```python
from langgraph.prebuilt import ToolNode

builder.add_node("tools", ToolNode([search, calculator]))
```

手写循环仅当你需要完全自定义执行策略（按工具类型限制并发、专用重试、
灰度开关）时才值得。

#### 5.9.1.2 工具内访问图状态与上下文：ToolRuntime

工具函数声明 runtime 参数即可读取模型参数之外的数据——模型**看不到**也
**填不了**这些参数，它们由框架注入。这解决了"工具需要的凭证/租户信息
不能让 LLM 自己编"的安全问题：

```python
@tool
def get_user_info(runtime) -> str:
    user_id = runtime.state["user_id"]          # 读取图状态
    org_id = runtime.context.organization_id    # 读取运行时上下文
    return f"{user_id}@{org_id}"

agent.invoke(inputs, context=Context(organization_id="acme"))
```

两个使用要点：

- `ToolNode` 直接作为图节点时，输入即当前图状态，`runtime.state` 自然可用；
  若在别的节点里手动 `tool_node.invoke(state)`，要传**完整状态**才能暴露
  自定义字段——只传一个子 dict 会导致 `runtime.state` 缺键；
- 涉及鉴权的场景（查"当前用户"的数据），一律走 `runtime.context` 而不是
  把 user_id 放进模型可见的 prompt/工具参数里，避免提示注入伪造身份。

#### 5.9.1.3 官方文档当前的 Agent 构建方式

官方 Workflows and agents 页面用 `should_continue` 条件边函数演示最小
ReAct 循环（模型有 tool_calls → 走 tools 节点；没有 → 走 END）；
`tools_condition` 是同一逻辑的预构建封装，两行代码换一个条件函数。
`create_agent` 等高层工厂返回**编译好的图**——把它 `add_node` 进父图后，
它就成为**一个子图**：父图要看到它内部的 LLM token 流，必须在
`stream(subgraphs=True)` 下消费（第 8、9 章会反复用到这个规则）。

#### 5.9.1.4 本章模式与官方示例的对照

本章 7 个示例对应官方 Workflow 模式族（chaining / parallelization /
routing / orchestrator-worker / evaluator-optimizer / agent），官方示例
代码结构与本仓库一致：结构化输出用 `with_structured_output(Pydantic模型)`，
工具用 `bind_tools`，动态子任务用 `Send`。

值得记住的官方原则：**能用确定代码完成的步骤写成普通节点，只有需要模型
判断的分支才用 LLM**。判断型路由（`with_structured_output` 输出枚举）
优于让模型自由生成路由指令——前者可测试、可枚举失败模式。这与第 0 章
"确定性优先"的边界原则一脉相承。

**练习：** 用 ToolRuntime 重写 `c05_07`，让工具从 `runtime.context` 读取
审批人，而不是硬编码在工具函数里。

### 5.10 本章总结

选择模式时按以下顺序：

1. 能不能用确定代码完成？能，就直接写 Node。
2. 步骤固定吗？固定，用 Chaining。
3. 任务能并行吗？能，用 Parallel。
4. 任务类型决定流程吗？是，用 Routing。
5. 子任务运行期才知道吗？是，用 Orchestrator-Worker。
6. 需要循环改进吗？用 Evaluator-Optimizer，并设置上限。
7. 模型需要自主选择工具吗？才使用 Agent。

### 5.11 章末自测

1. `with_structured_output` 和 `bind_tools` 的产物有什么不同？
2. 为什么 Evaluator-Optimizer 必须有最大轮数？
3. ReAct Agent 的终止路径由谁决定？
4. 多 Agent 数量变多后，最先失控的三项资源是什么？

---

## 第 6 章 持久化、记忆与时间旅行

对应目录：`c06_*`，共 6 个示例。

### 6.1 本章目标

- 理解 `thread_id`、Checkpoint 和 Super-step；
- 查询状态历史和 Replay；
- 从历史状态 Fork 出新分支；
- 区分 Checkpointer 和 Store；
- 管理消息长度、删除和摘要。

官方依据：

- [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)
- [Stores](https://docs.langchain.com/oss/python/langgraph/stores)
- [Use time travel](https://docs.langchain.com/oss/python/langgraph/use-time-travel)

### 6.2 Checkpointer 与 Store

| 对比项 | Checkpointer | Store |
| --- | --- | --- |
| 保存内容 | 图的 State 快照与任务写入 | 应用自定义 key-value 数据 |
| 作用域 | 单个 thread | 跨 thread |
| 典型用途 | 会话连续、HITL、恢复、时间旅行 | 用户偏好、事实、共享知识 |
| 访问方式 | config 中传 `thread_id` | 按 namespace 读写 |

两者经常一起使用：Checkpointer 记录“这次执行到哪里”，Store 记录“这个用户长期
需要记住什么”。

### 6.3 `c06_01_thread_checkpointer.py`：线程与会话

运行：

```bash
venv/bin/python langgraph-demo/c06_01_thread_checkpointer.py --offline
```

**核心配置：**

```python
config = {"configurable": {"thread_id": "demo-thread"}}
```

**观察重点：**

- 同一 `thread_id` 的后续调用能看到之前消息；
- 不同 `thread_id` 相互隔离；
- `InMemorySaver` 只适合当前进程；
- 未传 `thread_id` 时，需要持久化的能力无法按预期工作。

**练习：** 增加第二个用户线程，并根据消息数量验证隔离。

### 6.4 `c06_02_state_history_replay.py`：历史与 Replay

运行：

```bash
venv/bin/python langgraph-demo/c06_02_state_history_replay.py --offline
```

**关键 API：**

- `graph.get_state(config)` 获取最新快照；
- `graph.get_state_history(config)` 获取按时间倒序排列的历史；
- `graph.invoke(None, config=snapshot.config)` 从指定快照继续。

**状态快照重点字段：**

| 字段 | 含义 |
| --- | --- |
| `values` | 该 Checkpoint 时的状态 |
| `next` | 下一步将执行哪些节点；空元组表示完成 |
| `metadata` | `source`、`writes`、`step` 等执行信息 |
| `config` | 包含 thread、namespace、checkpoint ID |
| `parent_config` | 上一个 Checkpoint 的配置 |

**Replay 风险：** 后续节点会再次执行，因此外部副作用必须幂等。

**练习：** 比较两个快照的 `next`、`metadata.step` 和 `parent_config`。

### 6.5 `c06_03_fork_time_travel.py`：Fork 时间旅行

运行：

```bash
venv/bin/python langgraph-demo/c06_03_fork_time_travel.py --offline
```

**执行链路：**

1. 找到 `write` 执行前的 Checkpoint；
2. `update_state(..., as_node="prepare")` 修改历史状态；
3. 从返回的新 config 继续执行；
4. 新分支产生不同结果，原分支仍可查询。

**关键 API：** `update_state()`、`as_node`、snapshot config。

**使用场景：** 对比不同 Prompt、不同用户输入、不同工具返回对后续结果的影响。

**练习：** 从同一快照创建三个不同主题分支，并输出三个结果。

### 6.6 `c06_04_sqlite_checkpointer.py`：跨进程持久化

运行：

```bash
venv/bin/python langgraph-demo/c06_04_sqlite_checkpointer.py --offline
```

如果依赖未安装，脚本输出 `SKIP`：

```bash
venv/bin/pip install 'langgraph-checkpoint-sqlite>=3,<4'
```

**关键 API：** `SqliteSaver.from_conn_string()`。

**和 InMemorySaver 的差别：** SQLite 文件在进程退出后仍保留，适合本地开发；
生产通常选择 PostgreSQL 等后端。

**练习：** 进程 A 写入一个 thread，进程 B 用相同数据库和 thread 读取状态。

### 6.6.1 深入补充：Checkpointer 的存储结构、耐久模式与序列化

#### 6.6.1.1 Checkpoint 的存储布局：两张表

自定义 Checkpointer 或排查持久化问题时，需要知道底层是**两张表**，各有
明确分工：

| 表 | 粒度 | 每行内容 |
| --- | --- | --- |
| Checkpoints 表 | 每个 Super-step 一行 | 该超步的完整快照：`channel_values`（各通道当时的值）、`channel_versions`（通道版本号）、`versions_seen`（各任务见过的版本），并通过 `parent_checkpoint_id` 链接父 Checkpoint 形成链 |
| Writes 表 | 每个节点（任务）的每次写入一行 | `(task_id, channel, value)`——即"哪个节点往哪个通道写了什么" |

为什么要两张表？快照表回答"第 N 步结束时状态整体是什么"，Writes 表回答
"这一步里每个节点分别贡献了什么"。快照用于恢复与 Time-travel；Writes 用于
**Pending Writes** 机制（见下）与多节点写入的去重、故障恢复。

#### 6.6.1.2 Pending Writes：故障恢复的粒度保证

同一 Super-step 中**已成功节点**的写入会作为 Pending Writes 持久化。
它的含义：并行分支中一个节点失败时，已成功兄弟节点的输出**不会丢失也不会
重跑**——恢复后框架从写入记录得知它们已完成，只重跑失败节点。没有这个
机制，任何一步失败都会把整个超步的并行工作全部重做。这也反过来要求节点
写入是幂等的（见 2.7.1.4）。

#### 6.6.1.3 StateSnapshot 的补充字段

6.4 节已介绍 `values`、`next`、`config`、`metadata` 等字段，还有两个：

| 字段 | 含义 | 用途 |
| --- | --- | --- |
| `created_at` | ISO 8601 时间戳 | 按时间定位历史 Checkpoint、审计 |
| `tasks` | 本步要执行的任务元组，每项含 `id`、`name`、`error`、`interrupts` | 判断哪一步出错（`tasks[i].error`）、拿到子图状态（`tasks[0].state`，见 9.7.1） |

#### 6.6.1.4 耐久模式（durability）

```python
graph.invoke(inputs, config, durability="sync")
```

`exit` 性能最好但崩溃后无法从中间恢复；`async` 是官方推荐的默认选择；
`sync` 在每个 Super-step 结束时同步落盘、零丢失。完整语义、崩溃恢复行为
与选型建议见 0.7.3 节。

#### 6.6.1.5 序列化与加密

默认 `JsonPlusSerializer`：底层用 ormsgpack（MessagePack 格式），对
datetime、enum、Pydantic v2 模型等常见类型开箱即用。遇到不支持的类型
（如 Pandas DataFrame、自定义 C 扩展对象），可开启 pickle 回退：

```python
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
checkpointer = SqliteSaver(path, serde=JsonPlusSerializer(pickle_fallback=True))
```

注意 pickle 回退的安全边界：**Checkpoint 数据一旦可能被篡改，反序列化
pickle 就是代码执行漏洞**，只用于可信环境。

敏感状态（对话里的隐私、凭证）应整链路加密：

```python
from langgraph.checkpoint.serde.encryption import EncryptedSerializer

serde = EncryptedSerializer.from_pycryptodome_aes()  # 读取 LANGGRAPH_AES_KEY 环境变量
checkpointer = PostgresSaver.from_conn_string(uri, serde=serde)
```

#### 6.6.1.6 存储增长与修剪

默认**每个 Checkpoint 全量写入所有通道值**——不是增量 diff。50 轮对话的
thread 会有 50 份接近全量的快照，长对话线性膨胀。三种应对：

1. **定期删除旧 Checkpoint**（配合保留策略），但要遵守红线：修剪历史时
   **必须保留祖先 write 行**——状态重建依赖写入记录，删掉会导致该 thread
   的状态重建为空；
2. **`DeltaChannel`**（`langgraph>=1.2`，beta）：对指定通道只存增量，
   大列表/大文档字段的存储量显著下降；beta 期不建议核心业务直接上；
3. **从设计上控制**：把大体积数据（原始文档、图片）放外部对象存储，
   State 里只存引用 ID。

#### 6.6.1.7 PostgresSaver 实用细节

- 首次使用必须执行 `checkpointer.setup()` 建表（幂等，可重复调用）；
- `thread_id` 列有 **255 字符**长度限制，超长报数据库错误（不是友好提示），
  建议直接用 UUID；
- 官方生产推荐矩阵：`PostgresSaver` / `AsyncPostgresSaver`（LangSmith
  同款实现）、`MongoDBSaver`、Azure Cosmos DB Saver；社区还有 Redis 等
  实现，选型前先确认其对当前 langgraph-checkpoint 接口版本的兼容性。

#### 6.6.1.8 自定义 Checkpointer

继承 `BaseCheckpointSaver`，需要实现的方法族（每个都有同步/异步两版）：
`put/aput`（存快照）、`put_writes/aput_writes`（存节点写入）、
`get_tuple/aget_tuple`（取某 Checkpoint 及其 pending writes）、
`list/alist`（按条件遍历历史）、`delete_thread/adelete_thread`。

两条纪律：序列化必须走 `self.serde`（保证与加密/格式插件兼容），**不要**
直接 pickle metadata；实现完成后接入官方 **conformance suite** 做一致性
测试——自研 Checkpointer 最常见的坑不是存不进去，而是边界行为（并发写、
部分失败）与官方实现不一致。

**练习：** 用 SQLite 打开两个进程验证跨进程读写；把 `durability` 设为 `exit`
并在节点中途 kill 进程，对比 `sync` 模式下的恢复行为差异。

**练习：** 用 SQLite 打开两个进程验证跨进程读写；把 `durability` 设为 `exit`
并在节点中途 kill 进程，对比 `sync` 模式下的恢复行为差异。

### 6.7 `c06_05_store_long_term_memory.py`：长期记忆

运行：

```bash
venv/bin/python langgraph-demo/c06_05_store_long_term_memory.py --offline
```

**执行链路：** `store.put()` 写记忆 -> `store.get()` 精确读取 ->
`store.search()` 查询。

**关键概念：**

- namespace 是分层元组，例如 `("user", "alice", "memories")`；
- key 是命名空间内的稳定标识；
- value 是应用定义的字典；
- 配 Embedding 后可按语义搜索。

DeepSeek 当前未提供示例所需的 Embedding，因此 demo 使用本地确定性向量，只演示
Store 接口。生产环境应替换为正式 Embedding 服务。

**长期记忆策略必须包含：**

1. 什么时候写入？
2. 同一事实如何更新？
3. 冲突信息以谁为准？
4. 什么时候删除或过期？
5. 检索不到时如何处理？

**练习：** 增加删除和覆盖记忆流程，并验证 namespace 隔离。

### 6.7.1 深入补充：Store 的检索语义与生产化

#### 6.7.1.1 Item 对象与排序陷阱

`store.search` 返回 `Item` 列表，属性为 `value`、`key`、`namespace`、
`created_at`、`updated_at`。注意一个**不同后端行为不一致**的陷阱：

- `InMemoryStore` 按插入序返回（最新在**最后**）；
- `PostgresStore` 按 `updated_at` 倒序返回（最新在**最前**）。

同一份代码换后端后"最新/最旧"就反了。业务依赖顺序时应在客户端显式按
`updated_at` 排序，**不要依赖后端默认顺序**——这是跨环境迁移最常踩的坑。

#### 6.7.1.2 Prefix 匹配、分页与命名空间枚举

`search(("alice",))` 是 **prefix 匹配**：返回其所有子命名空间（如
`("alice", "memories")`）的条目，不只是恰好名为 `("alice",)` 的那一层。
这个语义适合"把某用户的所有记忆都拿出来"，但要注意两个细节：

1. **结果被静默截断**：超出 `limit` 的条目直接丢弃，**没有溢出信号**——
   你不知道还有多少没拿到。对完整性有要求的读取必须循环翻页；
2. **分页**：用 `offset` 参数继续向后取；想遍历"命名空间本身"（比如列出
   某用户下有哪几类记忆）用：

```python
store.list_namespaces(prefix=("alice",), max_depth=2)
```

#### 6.7.1.3 语义搜索配置

配置 Embedding 后，`search(query=...)` 会按向量相似度（而非精确 key 匹配）
返回条目：

```python
store = InMemoryStore(index={
    "embed": init_embeddings("openai:text-embedding-3-small"),
    "dims": 1536,                        # 必须与 embedding 模型输出维度一致
    "fields": ["food_preference", "$"],  # 要嵌入哪些字段；$ 表示整个 value
})
```

三个层级可以控制"什么被嵌入"：

- Store 级：`index.fields` 声明默认嵌入的字段；
- 条目级覆盖：`store.put(..., index=["food_preference"])` 只嵌入该字段；
- 完全跳过：`store.put(..., index=False)`——该条目仍可按 key 精确读取，
  但不可被语义检索命中。适合存大体积但不需要语义匹配的数据。

部署在 Agent Server 时，语义搜索需在 `langgraph.json` 中配置
`store.index`，进程内写法不生效。

#### 6.7.1.4 生产后端与自定义 Store

`PostgresStore`、`MongoDBStore`、`RedisStore`、`UpstashStore` 等均实现
`BaseStore`，接口与 `InMemoryStore` 完全一致——开发用内存实现，生产换
后端，业务代码不改。自定义 Store 需实现五个异步方法：`aput`、`aget`、
`adelete`、`asearch`（支持 `query` 语义检索；后端不支持向量检索时抛
`NotImplementedError`，框架会降级为普通搜索）、`alist_namespaces`。
与 Checkpointer 不同：Store 的值是**普通 dict，直接 JSON 序列化**，
没有复杂的 serde 层。

#### 6.7.1.5 在节点中通过 Runtime 使用

Store 的标准消费方式不是全局变量，而是从 `runtime.store` 拿——这样每个
请求可以用自己的 namespace 前缀（按用户隔离）：

```python
async def call_model(state: MessagesState, runtime: Runtime[Context]):
    namespace = (runtime.context.user_id, "memories")
    memories = await runtime.store.asearch(
        namespace, query=state["messages"][-1].content, limit=3)
    info = "\n".join(d.value["memory"] for d in memories)
```

这段代码同时展示了"检索-拼装-注入 prompt"的标准记忆流水线：按当前用户
隔离 namespace → 用最新消息做语义查询 → 限制条数 → 拼成上下文文本。

**练习：** 同一份数据分别写入 InMemoryStore 与 PostgresStore，比较 `search`
的默认排序差异，并写一个显式排序的读取函数。

### 6.8 `c06_06_memory_management.py`：上下文长度管理

运行：

```bash
venv/bin/python langgraph-demo/c06_06_memory_management.py --offline
```

**执行链路：** 写入多条消息 -> 使用 `RemoveMessage` 删除旧消息 ->
保留最近 N 条 -> 构造摘要。

**为什么重要：** 无限保留消息会增加 Token、延迟、费用和噪声，也可能把不相关
旧信息长期带入上下文。

**记忆策略：**

| 策略 | 优点 | 风险 |
| --- | --- | --- |
| 截断 | 简单、成本可控 | 可能丢掉关键事实 |
| 摘要 | 信息密度高 | 摘要可能失真 |
| 结构化记忆 | 可精确更新和检索 | 需要设计 Schema |
| 混合 | 兼顾近期原文和长期事实 | 实现复杂度更高 |

**常见误区：** 删除 Checkpoint 中的消息和删除 Store 长期记忆是两套不同操作。

**练习：** 把旧消息替换为 LLM 摘要，并验证摘要仍保留三个关键事实。

### 6.9 本章总结

持久化不是“存数据库”一个动作，而是四层设计：

1. 会话边界：`thread_id`；
2. 执行快照：Checkpoint；
3. 跨会话记忆：Store；
4. 记忆治理：写入、检索、摘要、更新、删除和保留策略。

### 6.10 章末自测

1. Checkpointer 和 Store 分别解决什么问题？
2. Replay 为什么可能重复执行副作用？
3. 历史列表为什么是倒序的？
4. Fork 后原分支是否被覆盖？
5. 什么情况下应删除或摘要旧消息？

---

## 第 7 章 Human-in-the-loop、容错与错误诊断

对应目录：`c07_*`，共 7 个示例。

### 7.1 本章目标

- 使用 `interrupt()` 在节点中动态暂停；
- 使用相同 `thread_id` 和 `Command(resume=...)` 恢复；
- 处理批准、拒绝和人工修改；
- 处理多个中断；
- 组合 Retry、Timeout 和 Error Handler；
- 在服务关闭时安全 Drain；
- 读懂常见 LangGraph 错误。

官方依据：

- [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [Fault tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance)
- [Error reference](https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT)

### 7.2 Interrupt 的工作机制

当节点执行 `interrupt(payload)`：

1. 图在准确位置暂停；
2. Checkpointer 保存当前状态；
3. payload 返回给调用方；
4. 图等待外部输入；
5. 调用方用相同 `thread_id` 和 `Command(resume=value)` 恢复；
6. `value` 成为原 `interrupt()` 的返回值；
7. 节点会从开头重新执行。

因此，`interrupt()` 之前的副作用必须可重复执行。不要把不可撤回的外部动作放在
中断之前。

### 7.3 `c07_01_interrupt_resume.py`：最小暂停与恢复

运行：

```bash
venv/bin/python langgraph-demo/c07_01_interrupt_resume.py --offline --answer Alice
```

**执行链路：**

```text
invoke -> ask 节点 interrupt -> 返回 __interrupt__
       -> Command(resume="Alice") -> 节点继续 -> END
```

**必要条件：** Checkpointer、稳定 `thread_id`、JSON 可序列化 payload。

**常见误区：**

- 没有 Checkpointer；
- 恢复时换了 `thread_id`；
- `interrupt()` 的调用顺序发生变化；
- 将 `interrupt()` 包在会吞异常的 `try/except` 中。

**练习：** 返回结构化问题，并校验人工答案格式。

### 7.4 `c07_02_approve_reject.py`：批准与拒绝

运行：

```bash
venv/bin/python langgraph-demo/c07_02_approve_reject.py --offline --decision approve
venv/bin/python langgraph-demo/c07_02_approve_reject.py --offline --decision reject
```

**执行链路：** interrupt 返回布尔决定 -> 条件边进入批准或拒绝节点。

**设计重点：** 拒绝是正常业务结果，不是系统异常。拒绝路径也应有明确输出、日志和
后续处理。

**练习：** 增加“修改参数后重新审批”的第三分支。

### 7.5 `c07_03_review_edit_state.py`：审核并修改内容

运行：

```bash
venv/bin/python langgraph-demo/c07_03_review_edit_state.py --offline
```

**执行链路：** interrupt 返回草稿 -> 人工提交新字典 -> 节点写回修改后的草稿。

**安全边界：** 恢复载荷来自外部，必须校验长度、格式、枚举值和权限，不能直接
信任并写入关键状态。

**练习：** 对修改内容增加敏感词和最大长度校验。

### 7.6 `c07_04_multiple_tool_interrupts.py`：多个中断

运行：

```bash
venv/bin/python langgraph-demo/c07_04_multiple_tool_interrupts.py --offline
```

**顺序中断：** `ask_name` 恢复后进入 `ask_count` 再暂停一次。

**并行中断：** 分支可能同时暂停，不能用隐式顺序配对回答。应使用各 Interrupt 的
ID 构成恢复映射：

```python
resume_map = {interrupt.id: value for interrupt in interrupts}
Command(resume=resume_map)
```

**练习：** 改成两个并行中断，并使用 ID 映射一次性恢复。

### 7.7 `c07_05_retries_timeouts_errors.py`：重试与错误处理

运行：

```bash
venv/bin/python langgraph-demo/c07_05_retries_timeouts_errors.py --offline
```

**执行链路：**

1. `flaky` 第一次抛 `ConnectionError`；
2. RetryPolicy 判断匹配并重试；
3. 成功后继续；
4. 另一个图中 `risky` 失败；
5. Error Handler 接收 `NodeError` 并返回恢复状态。

**三层职责：**

| 机制 | 回答的问题 |
| --- | --- |
| Retry | 这次错误是否值得再试？ |
| Timeout | 单次尝试最多允许运行多久？ |
| Error Handler | 重试耗尽后，业务如何恢复或降级？ |

**注意：** Error Handler 本身也要有失败边界，不能无限递归或再次访问失控资源。

**练习：** 给 Error Handler 增加审计字段和升级到人工处理的路径。

### 7.8 `c07_06_graceful_drain.py`：优雅排空

运行：

```bash
venv/bin/python langgraph-demo/c07_06_graceful_drain.py --offline
```

**执行链路：** 节点调用 `runtime.control.request_drain("sigterm")` ->
当前 Super-step 完成 -> 抛出 `GraphDrained` -> 保存 Checkpoint ->
相同 config 恢复。

**关键语义：**

- Drain 不强行取消正在运行的节点；
- 它在安全边界停止；
- 必须配合 Checkpointer 才能恢复；
- 适合服务收到 SIGTERM 时保存进度并退出。

**练习：** 在信号处理器中调用 `request_drain`，并验证进程重启后继续执行。

### 7.9 `c07_07_error_reproduction.py`：常见错误复现

运行：

```bash
for case in recursion missing-checkpointer invalid-return concurrent-update bad-history multiple-subgraphs; do
  venv/bin/python langgraph-demo/c07_07_error_reproduction.py --case "$case"
done
```

**主要错误：**

| 错误 | 根因 | 修复方向 |
| --- | --- | --- |
| `GRAPH_RECURSION_LIMIT` | 循环缺少业务终止条件 | 增加状态条件或剩余步数 |
| `MISSING_CHECKPOINTER` | 使用 Interrupt 等能力却没保存状态 | 编译时配置 Checkpointer |
| `INVALID_GRAPH_NODE_RETURN_VALUE` | 节点返回了不受支持的类型 | 返回状态增量字典或 `Command` |
| `INVALID_CONCURRENT_GRAPH_UPDATE` | 并行节点写同一无 Reducer 字段 | 增加 Reducer 或合并策略 |
| `INVALID_CHAT_HISTORY` | 消息历史结构不合法 | 使用消息对象和 `add_messages` |
| `MULTIPLE_SUBGRAPHS` | 部署时子图组合限制 | 检查应用结构和子图命名 |

**方法：** 先分类错误属于 Graph 结构、State、运行配置还是部署约束，再修改对应层，
不要只在最后一行异常信息上打补丁。

**练习：** 为每个错误写回归测试，确保修复后不会重新出现。

### 7.9.1 深入补充：Retry、Timeout、Error Handler 的完整参数与行为

#### 7.9.1.1 RetryPolicy 全参数解析

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `retry_on` | `default_retry_on` | 可重试异常类型 / 列表 / callable；callable 收到异常实例返回 bool |
| `max_attempts` | 3 | 最大尝试次数（**含首次**，即最多重试 2 次） |
| `initial_interval` | 0.5 | 首次重试前等待秒数 |
| `backoff_factor` | 2.0 | 指数退避因子：每次间隔 = 上次 × factor |
| `max_interval` | 128.0 | 重试间隔上限，退避再久也不超过它 |
| `jitter` | True | 加入随机抖动，避免多个实例同一时刻集体重试（惊群） |

**默认可重试范围的细节**比参数表更重要：

- `default_retry_on` 默认重试一切异常，但**排除编程错误类**——`ValueError`、
  `TypeError`、`ArithmeticError` 等。逻辑：编程错误重试也不会好，快速失败
  才能尽早暴露 bug；
- 对 `requests`/`httpx` 的 HTTP 错误**只重试 5xx**（服务端错误），4xx
  （请求本身有问题）不重试；
- `NodeTimeoutError` 默认可重试——超时往往意味着下游暂时不可用，重试
  是合理动作；
- 自定义时可以 `from langgraph.retry import default_retry_on` 拿到默认
  函数，在默认行为之上扩展（例如追加排除某个自定义异常），而不是从零写
  判断逻辑。

#### 7.9.1.2 用重试状态做主备切换

RetryPolicy 只是"再试一次"，但节点内部可以通过 `runtime.execution_info`
感知"这是第几次尝试"，实现比盲目重试更聪明的策略——降级：

```python
def call_api(state, runtime):
    if runtime.execution_info.node_attempt > 1:
        return call_fallback_api()   # 已经失败过一次，改走备用
    return call_primary_api()
```

第 1 次尝试走主 API；重试触发时 `node_attempt` 变为 2，直接改走备用 API——
与其再赌一次主 API，不如切换目标。这是"重试 + 熔断降级"在一个节点里的
最小实现。

#### 7.9.1.3 Timeout 细节（仅限 async 节点）

同步节点传 `timeout` 在**编译期被拒绝**（直接报错，不是运行时忽略）；阻塞
I/O 用 `asyncio.to_thread` 包装成 async。三种超时形态，语义完全不同：

| 写法 | 语义 | 适用场景 |
| --- | --- | --- |
| `timeout=60` 或 `TimeoutPolicy(run_timeout=120)` | **墙钟硬上限**：从节点开始计时，到点即杀，不刷新 | 预算封顶：无论如何不允许超过的总体时间 |
| `TimeoutPolicy(idle_timeout=30)` | **空闲上限**：状态写入、流输出、子任务调度等"进度信号"都会重置时钟 | 只要还在产出就不打断，只杀真正卡死的节点 |
| `TimeoutPolicy(idle_timeout=30, refresh_on="heartbeat")` | **心跳上限**：只有 `runtime.heartbeat()` 能重置时钟 | 长批处理：节点内每完成一批调一次 heartbeat，静默超时即杀 |

`Send` 派发时可**按任务覆盖**超时——动态子任务给了更细的控粒度：

```python
Send("process_item", item, timeout=TimeoutPolicy(idle_timeout=15))
```

#### 7.9.1.4 Error Handler 的边界

```python
def payment_handler(state, error: NodeError):
    return Command(goto="cleanup", update={"refund_needed": True})

builder.add_node("pay", pay_node, error_handler=payment_handler)
```

行为边界逐条说明：

- 每个节点**至多一个** `error_handler`；签名支持 `(state)` 或
  `(state, error: NodeError)`，可选第三参数 `RunnableConfig`；
- `NodeError` 含 `node`（出错的节点名）与 `error`（原始异常）两个字段；
- handler 返回 `Command` 可实现 **Saga 补偿**：出错后 goto 到补偿节点，
  执行退款、释放库存等回滚动作，再进入收尾——分布式事务思想在单图内的
  最小落地；
- **handler 自身抛错会向上冒泡**：错误处理器不是异常防火墙，它自己坏了
  照样炸；
- `interrupt()` 内部走 `GraphBubbleUp` 机制，**绕过**重试与错误处理器——
  中断不是错误，图照常暂停等待人工输入，不会被重试策略误处理；
- 失败上下文会被 Checkpoint：图在 handler 完成前崩溃，恢复后 handler 仍能
  看到**相同的错误对象**（resume-safe failure），补偿逻辑不会因崩溃丢失；
- 子图未处理的异常会冒泡到父节点 handler 的 `error.error`——父层的
  补偿逻辑可以覆盖子图内部的所有失败。

#### 7.9.1.5 interrupt 与静态断点的分工

`compile(interrupt_before=[...])` 属于静态断点，恢复方式是
`invoke(None, config)`。官方对它的定位是**调试**：逐步执行、在 Studio 里
检查每步状态。生产 HITL **一律用 `interrupt()`**——静态断点写死在编译期，
无法携带审批上下文、无法在运行时按业务条件触发。

#### 7.9.1.6 工具内中断与输入校验

`@tool` 函数内可直接调用 `interrupt()`，实现工具级审批（如邮件发送前
确认收件人与正文）。需要**校验人工输入**时，用"单次 interrupt + 条件边
循环回节点"的模式，而不是节点内 `while True`：

```python
# ✅ 正确：interrupt 拿一次输入，条件边决定"通过"还是"回到修改节点"
# ❌ 错误：while True: value = interrupt(...) 直到满意为止
```

原因：Functional API 的 interrupt 按调用顺序索引（见 4.6.1.3），循环内的
多次 interrupt 会让索引与 resume 值的对应关系变得不可预测。

#### 7.9.1.7 Drain（优雅排空）补充

节点内可用 `runtime.drain_requested` 提前感知停机信号：收到 SIGTERM 后，
排空中的图不再派发新工作，长节点可检查该标志返回"跳过"结果，让进程尽快
退出。SIGTERM 钩子写法：

```python
signal.signal(signal.SIGTERM, lambda *_: control.request_drain("sigterm"))
```

与硬杀（kill -9）的差别：排空让已开始的工作有机会到达下一个 Checkpoint，
重启后从一致状态恢复；硬杀依赖 `sync`/`async` 耐久模式兜底（见 0.7.3）。

**练习：** 组合 `TimeoutPolicy(idle_timeout=..., refresh_on="heartbeat")` +
`RetryPolicy` + error handler，构造"慢 API → 超时 → 重试 → 降级"全链路，
并记录每一层的触发时机。

### 7.10 Interrupt 的硬性规则

1. 不要在 `try/except` 中捕获并吞掉 `interrupt()`。
2. 不要在恢复前后改变同一节点中的 `interrupt()` 调用顺序。
3. payload 应是可序列化、尺寸合理的值，不要塞入巨大对象或密钥。
4. `interrupt()` 前的副作用必须幂等。
5. 多个并行中断恢复时使用 Interrupt ID 映射。
6. 生产环境必须使用持久化 Checkpointer，而不是只在内存中保存。

### 7.11 章末自测

1. 为什么恢复时节点会从头执行？
2. Retry 和 Error Handler 的边界在哪里？
3. 为什么拒绝分支不能只当成异常？
4. Graceful Drain 为什么只能保证在安全边界停止？
5. 并行中断为什么不能简单按列表顺序传 resume 值？

---

## 第 8 章 Streaming、事件协议与可观测性

对应目录：`c08_*`，共 6 个示例。

### 8.1 本章目标

- 区分 `values`、`updates`、`messages` 和 `custom`；
- 使用 `version="v2"` 获得统一 StreamPart；
- 理解 `tasks`、`checkpoints`、`debug` 的调试用途；
- 区分 Stream Mode API 与 Event Streaming；
- 理解 namespace 如何标识子图事件；
- 建立 LangSmith Trace 与脱敏意识。

官方依据：

- [Streaming](https://docs.langchain.com/oss/python/langgraph/streaming)
- [Event streaming](https://docs.langchain.com/oss/python/langgraph/event-streaming)
- [LangSmith Observability](https://docs.langchain.com/oss/python/langgraph/observability)

### 8.2 两种流式 API

官方当前把两者分成两个层次：

| 层次 | API | 主要用途 |
| --- | --- | --- |
| 原始 Graph Streaming | `graph.stream(..., stream_mode=...)` | 直接读取 Pregel 产生的 mode chunks |
| Event Streaming | `graph.stream_events(..., version="v3")` | 用 messages、values、subgraphs、output 等 typed projections 构建应用 |

对新的应用代码，官方推荐优先评估 Event Streaming；但本仓库当前大量示例仍使用
`stream(..., version="v2")`，因为它的 StreamPart 结构稳定、直观，适合先理解底层。

Event Streaming 的 `v3` 在本地环境中仍可能触发 Beta Warning，升级时要检查兼容性。

### 8.3 StreamPart v2 的统一格式

```python
{
    "type": "values | updates | messages | custom | checkpoints | tasks | debug",
    "ns": (),
    "data": ...,
}
```

| 字段 | 含义 |
| --- | --- |
| `type` | 当前 chunk 来自哪种流模式 |
| `ns` | 子图 namespace；根图为空元组 |
| `data` | 各模式对应的实际载荷 |

使用 `version="v2"` 后，无论只有一个模式还是多个模式，输出结构都一致，前端和
测试代码可以按 `type` 分派。

### 8.4 常见 Stream Mode

| mode | 输出 | 适用 |
| --- | --- | --- |
| `values` | 每一步后的完整 State | 状态面板、最终快照 |
| `updates` | 每个节点返回的增量 | 进度显示、节点审计 |
| `messages` | `(message_chunk, metadata)` | 聊天 Token 流 |
| `custom` | `get_stream_writer()` 写入的数据 | 业务进度事件 |
| `checkpoints` | Checkpoint 事件，需要 Checkpointer | 状态历史调试 |
| `tasks` | 任务开始/结束与结果/错误 | 节点级监控 |
| `debug` | 更完整的底层事件 | 开发调试 |

生产前端通常优先消费 `messages` 和结构化的 `custom`，不要把 `debug` 内部对象
直接暴露给终端用户。

### 8.5 `c08_01_stream_values_updates.py`：完整状态与增量

运行：

```bash
venv/bin/python langgraph-demo/c08_01_stream_values_updates.py --offline
```

**观察重点：**

- `values` 每一步都展示完整状态；
- `updates` 只展示当前节点真正返回的字段；
- 两个模式可以在一次 `stream()` 中同时产生；
- `values` 数据量更大，适合状态面板；`updates` 更适合增量 UI。

**练习：** 为每个 update 记录节点耗时，并输出一张执行时间线。

### 8.6 `c08_02_stream_messages.py`：模型消息与 Token

运行：

```bash
venv/bin/python langgraph-demo/c08_02_stream_messages.py --offline
```

**执行链路：** 节点调用模型 -> `stream_mode="messages"` 接收消息 chunk ->
根据 metadata 识别来源节点。

**重要认识：** 消息 chunk 的到达顺序不等同于整张图的节点拓扑顺序。前端应结合
metadata、namespace 和消息 ID 构建正确的对话流。

**练习：** 拼接 Token、显示当前模型节点名，并标记工具调用 chunk。

### 8.7 `c08_03_custom_tasks_checkpoints.py`：多种调试模式

运行：

```bash
venv/bin/python langgraph-demo/c08_03_custom_tasks_checkpoints.py --offline
```

**同时观察四种数据：**

- `custom`：节点主动发送的业务进度，如 `progress=50`；
- `tasks`：任务开始、结果、错误；
- `debug`：组合了更完整的内部事件；
- `checkpoints`：需要 Checkpointer 的状态保存事件。

**设计建议：**

- 前端只依赖你有意维护的 `custom` Schema；
- `tasks` / `checkpoints` / `debug` 适合开发、诊断和内部监控；
- 不要向前端泄露 Prompt、内部路径或敏感状态。

**练习：** 把 custom event 改成稳定的结构化协议，例如
`{"event": "progress", "percent": 50}`。

### 8.8 `c08_04_multiple_modes_subgraphs.py`：多模式与子图 namespace

运行：

```bash
venv/bin/python langgraph-demo/c08_04_multiple_modes_subgraphs.py --offline
```

**执行链路：** 子图发送 custom 事件 -> 父图 finalize -> 同时输出 updates/custom。

**namespace 的意义：**

- 空元组表示根图；
- 子图事件带独立 namespace；
- 同名节点可以通过 namespace 区分；
- 多级子图 namespace 会形成层次路径。

**练习：** 根据 namespace 构建父子执行树，显示每层状态。

### 8.9 `c08_05_event_streaming.py`：v3 事件协议

运行：

```bash
venv/bin/python langgraph-demo/c08_05_event_streaming.py --offline
```

**脚本观察：** 使用 `graph.stream_events(..., version="v3")` 迭代协议事件，并收集
`method` 名称。

**Event Streaming 的 typed projections：**

| projection | 用途 |
| --- | --- |
| `stream.messages` | 模型消息与 Token |
| `stream.values` | 状态快照和最终值 |
| `stream.output` | 等待最终输出 |
| `stream.subgraphs` | 观察子图执行 |
| `stream.interrupts` | 读取中断 payload |
| `stream.interrupted` | 判断是否因人工输入暂停 |
| `stream.extensions` | 自定义 transformer projection |

多个 projection 可以同时消费同一个底层事件流，不需要像手工处理 stream_mode
那样自己分派所有 chunk。

**练习：** 把 `stream.values` 投影成前端状态快照，并用 `stream.output` 获取最终
结果。

### 8.10 `c08_06_langsmith_tracing.py`：追踪与脱敏

运行：

```bash
venv/bin/python langgraph-demo/c08_06_langsmith_tracing.py --offline
```

**关键环境变量：**

- `LANGSMITH_TRACING=true`
- `LANGSMITH_API_KEY=...`
- `LANGSMITH_PROJECT=...`

**Trace 能看到什么：** 图、节点、模型、工具、耗时、输入输出和状态流转。

**Trace 的主要风险：** 输入输出可能包含个人信息、业务数据、Prompt 或敏感工具
参数。开启前必须确定脱敏策略和保留策略。

**练习：** 增加 user/session metadata，并用 LangSmith anonymizer 移除敏感字段。

### 8.10.1 深入补充：v1/v2/v3 演进、Token 过滤与事件生命周期

#### 8.10.1.1 v1 → v2 迁移对照

v2 的核心变化是**把所有流事件统一为 `StreamPart{type, ns, data}` 结构**，
不同模式/子图不再用元组形状区分：

| 场景 | v1 | v2 |
| --- | --- | --- |
| 单流模式 | 原始数据（裸 dict/list） | `StreamPart{type, ns, data}` |
| 多流模式 | `(mode, data)` 元组，靠位置猜 | 同上，按 `chunk["type"]` 分派 |
| 子图 | `(namespace, data)` 元组 | 同上，看 `chunk["ns"]` |
| `invoke()` 返回 | 普通 dict | `GraphOutput`（`.value` / `.interrupts`；dict 方式访问已弃用） |
| Pydantic/dataclass 状态 | 返回 plain dict | 强制转为模型/数据类实例（类型不丢失） |

迁移的实际收益：消费端从"按元组形状 + 位置"的猜测式解析，变成按 `type`
字段的显式分派；同一套处理代码可以覆盖所有模式与子图层级。

#### 8.10.1.2 messages 流的三种过滤

多模型/多节点的图里，不是所有 token 都该进流（内部分类器、隐藏评估模型的
token 会污染前端输出）。三种过滤手段，作用在不同层面：

```python
# 1. 按模型 tag 过滤（选择：只让某些模型的 token 进流）
#    产端：llm.bind_tools(...).with_config(tags=["joke"])
#    消费端：判断 chunk.metadata["tags"] == ["joke"]

# 2. 屏蔽内部模型（排除：某个模型的 token 完全不进流）
#    产端：with_config(tags=["nostream"])——模型照常执行，只是 token 不入流

# 3. 按节点过滤（位置：只要某个节点里产生的消息）
#    消费端：metadata["langgraph_node"] == "call_model"
```

不支持流式的模型用 `init_chat_model(..., streaming=False)` 或基类参数
`disable_streaming=True` 排除——否则它会以"一次性整块消息"的形式出现在
messages 流里，前端会看到突兀的整段输出。

#### 8.10.1.3 Python < 3.11 的两个坑

两个限制都源于旧版 Python 的 asyncio task 上下文传递缺陷：

1. 异步代码必须把 `RunnableConfig` **显式传给** `model.ainvoke()` 才有
   token 流——否则 config 丢失，模型回调链断掉；
2. 不能用 `get_stream_writer()`（它依赖 `contextvars` 自动传播），要在
   节点/工具签名里声明 `writer: StreamWriter` 参数由框架注入。

Python ≥ 3.11 两者都自动正常。写库代码时若要兼容旧版本，统一用参数
注入与显式传 config 是最稳的。

#### 8.10.1.4 子图 token 需要 subgraphs=True

`create_agent` 返回的编译图作为节点加入父图后即成子图。父图的
`stream_mode="messages"` 默认只收**本层**的 LLM token；不开
`subgraphs=True`，子 agent 内部的 token 对流完全不可见——表现为"子 Agent
明明在干活，前端却收不到字"。这是多 Agent 流式最高频的疑问。

#### 8.10.1.5 Event Streaming（v3）：投影模型

v3 的设计是**一条事件流、多个投影视图**：底层事件只发一次，不同投影
（projection）各自过滤/变换出专用视图。完整投影：

| 投影 | 用途 |
| --- | --- |
| `stream` | 原始 `ProtocolEvent`（`seq`、`method`、`params{namespace, timestamp, data}`），做自定义解析或录放 |
| `stream.messages` | 消息与 token（`.text`、`.tool_calls`、usage） |
| `stream.values` | 每步状态快照 |
| `stream.output` | 最终输出（可 await） |
| `stream.subgraphs` | 嵌套图执行（`.graph_name`、`.path`） |
| `stream.interrupts` / `stream.interrupted` | 中断负载与暂停标记 |
| `stream.extensions` | 自定义 StreamTransformer 的输出 |

关键性质：**多个投影可并发消费且互不消耗事件**——同时开 `consume_messages()`
和 `consume_subgraphs()` 两个协程，各自完整拿到数据（不是瓜分一条流）。
同步代码用 `stream.interleave("values", "messages")` 按到达顺序交错读取。

自定义投影：继承 `StreamTransformer`，实现 `init/process/finalize/fail`
四个钩子，声明 `required_stream_modes`（告诉框架需要订阅哪些底层模式），
注册位置二选一：`stream_events(transformers=[...])`（调用级）或
`compile(transformers=[...])`（编译级，所有调用生效）。

#### 8.10.1.6 原始事件通道的生命周期

底层通道包括 `values`、`updates`、`messages`、`tools`、`lifecycle`
（started / running / completed / failed / interrupted）、`checkpoints`、
`tasks`、`custom`、`custom:<name>`。两个结构化通道有完整生命周期事件，
适合做高保真前端：

- 消息通道：`message-start` → `content-block-delta`（逐 token）→
  `message-finish`——能区分"新消息开始"与"旧消息追加"，渲染多轮对话流；
- 工具通道：`tool-started` / `tool-output-delta` / `tool-finished` /
  `tool-error`——能实时展示"正在调用什么工具、收到什么输出、是否失败"。

#### 8.10.1.7 部署形态提示

图若部署在 Agent Server 之后，进程内 `stream_events` 不再是正确入口，
应改用 LangSmith Streaming API（HTTP/SSE 消费服务端推来的同一套协议事件）。
协议本身一致，前端消费逻辑基本可复用。

**练习：** 把 `c08_05` 改写为"messages 与 values 双投影并发消费"，再用
`StreamTransformer` 自定义一个只吐进度百分比的投影。

### 8.11 本章总结

流式设计要先确认消费者：

1. 聊天 UI：优先 `messages`；
2. 进度条：定义 `custom` 协议；
3. 状态面板：使用 `values`；
4. 节点审计：使用 `updates` 或 `tasks`；
5. 框架级应用：优先评估 Event Streaming 的 typed projections；
6. 调试：临时使用 `debug` / `checkpoints`，不要直接暴露。

### 8.12 章末自测

1. `values` 和 `updates` 的数据量和语义差异是什么？
2. 为什么不能依赖 Token chunk 的到达顺序重建图拓扑？
3. Event Streaming 的 projection 与 stream mode 是什么关系？
4. 为什么开启 LangSmith 前要先做数据分类？

---

## 第 9 章 子图与多 Agent

对应目录：`c09_*`，共 4 个示例。

### 9.1 本章目标

- 理解子图是“被父图当作节点使用的图”；
- 根据 State Schema 是否兼容选择组合方式；
- 理解子图持久化、namespace 和流式输出；
- 对比 Supervisor、Handoff、Agent-as-Tool 和并行 Agent；
- 判断何时不应引入多 Agent。

官方依据：

- [Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)
- [Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [Multi-agent](https://docs.langchain.com/oss/python/langchain/multi-agent)

### 9.2 子图的两种组合方式

| 方式 | 条件 | 写法 | 优点 |
| --- | --- | --- | --- |
| 编译子图直接作为节点 | 父子共享兼容状态字段 | `add_node("child", child_graph)` | 组合简洁、共享 Reducer |
| 在节点中显式调用子图 | 输入输出 Schema 不同 | `child.invoke(...)` 后转换输出 | 隔离状态、边界清晰 |

选择依据不是个人偏好，而是 State Schema 是否需要显式映射。

### 9.3 `c09_01_subgraph_as_node.py`：编译子图作为节点

运行：

```bash
venv/bin/python langgraph-demo/c09_01_subgraph_as_node.py --offline
```

**执行链路：** 父图和子图共享 `SharedState` -> 编译子图直接注册为父节点 ->
父图执行时进入子图。

**需要注意：**

- 父子 State 字段和 Reducer 必须兼容；
- 子图内部节点仍按自己的边执行；
- 持久化策略和 namespace 会影响流式观察和状态查询。

**练习：** 在父图中并行放两个共享 State 的独立子图，并用 Reducer 汇总。

### 9.4 `c09_02_subgraph_in_node.py`：节点内显式调用

运行：

```bash
venv/bin/python langgraph-demo/c09_02_subgraph_in_node.py --offline
```

**执行链路：** 从 `ParentState` 构造 `ChildState` -> `child.invoke()` ->
只把 `result` 转成父图的 `child_output`。

**优势：**

- 父图和子图可以有不共享字段；
- 子图私有字段不会自动泄漏到父状态；
- 输入输出映射非常明确。

**必须自己处理：** 异常、配置传递、Checkpointer 语义和序列化边界。

**练习：** 给子图增加两个私有字段，只把摘要返回父图。

### 9.5 `c09_03_subgraph_persistence_stream.py`：持久化与流式

运行：

```bash
venv/bin/python langgraph-demo/c09_03_subgraph_persistence_stream.py --offline
```

**执行链路：** 父子图共享 Checkpointer -> 子图节点发送 custom event ->
父图调用 `stream(..., subgraphs=True, version="v2")`。

**持久化规则：**

- 有状态子图可以继承父图 Checkpointer；
- 子图有自己的 Checkpoint namespace；
- 恢复和流式事件都应结合 namespace 解释；
- 父图不一定会立即看到子图每个内部变化。

**练习：** 对比父图有/无 Checkpointer 时，子图中断与恢复行为。

### 9.6 `c09_04_multi_agent_patterns.py`：四种多 Agent 结构

运行：

```bash
for pattern in supervisor handoff agent-tool parallel; do
  venv/bin/python langgraph-demo/c09_04_multi_agent_patterns.py \
    --offline --pattern "$pattern"
done
```

#### Supervisor

一个协调节点选择 Worker，Supervisor 决定任务给谁。需要防止无限转交，并设置最大
轮数和失败回退。

#### Handoff

一个 Agent 完成后把控制权明确交给下一个 Agent。适合固定协作流水线，例如
Researcher -> Writer -> Reviewer。

#### Agent-as-Tool

把另一个 Agent 包装成工具供主 Agent 调用。适合主 Agent 需要按需咨询专家 Agent，
但不应让嵌套调用无限扩张。

#### Parallel Agents

多个 Agent 独立处理同一任务的不同视角，结果通过 Reducer 汇总。适合研究和评审，
代价是模型成本并发叠加。

### 9.7 多 Agent 的引入条件

只有出现以下真实需求时才考虑拆分：

- 状态必须隔离，不能共享全部上下文；
- 不同角色需要不同工具和权限；
- 子任务可以并行，且分工收益大于协调成本；
- 独立团队维护不同子图边界；
- 需要独立测试、部署或评估某个专业 Agent。

如果只是 Prompt 稍不同、工具相同、流程固定，通常单个图或单 Agent 更易维护。

**练习：** 给 Supervisor 增加最大移交次数，并为每次移交记录来源、目标和理由。

### 9.7.1 深入补充：子图持久化三模式与时间旅行粒度

#### 9.7.1.1 子图 Checkpointer 的三种配置

`subgraph.compile(checkpointer=...)` 的取值决定子图的持久化行为，三种模式
的完整对比：

| 配置 | 中断 | 跨调用记忆 | 并行调用 | 状态检查 |
| --- | --- | --- | --- | --- |
| `None`（默认，继承父图） | ✅（单次调用内） | ❌ | ✅ | ⚠️ 有限 |
| `True`（per-thread） | ✅ | ✅ 线程内累积 | ❌ 同命名空间写冲突 | ✅ |
| `False`（stateless） | ❌ | ❌ | ✅ | ❌ |

逐行解释这张表：

- **继承父图（None）**：子图没有自己的 Checkpointer，中断时依赖父图的
  Checkpoint 挂起整棵调用树。中断能工作，但记忆只限单次调用——下一次
  `invoke` 子图从零开始；
- **per-thread（True）**：子图在父 Checkpoint 下获得**自己的命名空间**，
  同一 thread 多次调用可累积记忆、可独立恢复。代价是不支持并行工具调用
  （同一 checkpoint namespace 写冲突），官方建议用
  `ToolCallLimitMiddleware` 限制调用数规避；
- **stateless（False）**：完全不持久化，中断不支持，但并行安全——适合
  "每次调用都是独立无状态转换"的子流程。

**命名空间分配规则**（per-thread 模式必需）：直接 `add_node` 时自动按
节点名分配（稳定、可预测）；节点内调用则按**调用顺序**分配（第一次调用
是 0 号、第二次 1 号……极易混淆）。需要多个 per-thread 子图时，应把节点内
调用包装成拥有**唯一节点名**的 `StateGraph`，靠节点名获得稳定命名空间。

#### 9.7.1.2 查看子图状态

```python
graph.get_state(config, subgraphs=True).tasks[0].state
```

可拿到子图自己的快照与 config。前提是子图是"**静态可发现**"的——作为
节点添加，或在节点内直接调用；经工具函数间接调用的子图不可见（框架无法
建立父子链接）。但注意：**不可见 ≠ 不会中断**——工具内子图的中断仍会向上
传播，只是你在父层看不到它的内部状态，恢复时要靠工具层的 resume 逻辑。

#### 9.7.1.3 时间旅行粒度：父图把子图当成一个 Super-step

默认（继承父 Checkpointer）时，父图把整个子图当作**一个 Super-step**：
只能在父层从子图之前 fork，整个子图从头重跑，**无法**落到子图内部两个
节点之间。给子图 `compile(checkpointer=True)` 后，子图内部每步产生
Checkpoint，可以 fork 到子图内部任意点：

```python
parent_state = graph.get_state(config, subgraphs=True)
sub_config = parent_state.tasks[0].state.config      # 子图自己的 checkpoint
fork_config = graph.update_state(sub_config, {"value": ["forked"]})
result = graph.invoke(None, fork_config)             # 只重跑子图后半段
```

粒度选择的判断：子图是"黑盒能力"（如一次检索、一次总结）→ 继承父图，
粒度到子图边界就够；子图是"长流程"（含中断、多次模型调用）→ per-thread，
获得细粒度恢复与重放。

另外一个容易忽略的行为：**时间旅行会重新触发 interrupt**——Replay/Fork
到含 `interrupt()` 的节点时，会暂停等待新的 `Command(resume=...)`，可以
给出与原分支不同的答案。这让"改一个审批决定、只重放后半段"成为可能。

#### 9.7.1.4 多 Agent 模式的官方分类与成本视角

官方把多 Agent 模式归纳为五类，每类的控制权与上下文流向不同：

1. **Subagents**：主 Agent 把子 Agent 当**工具**调用——所有路由都经主
   Agent，子 Agent 的上下文对主 Agent 不可见（隔离）；
2. **Handoffs**：通过工具调用**切换控制权与配置**——控制权移交后由接手
   Agent 直接驱动；
3. **Skills**：按需加载专业 prompt/知识——同一 Agent 切换"技能包"；
4. **Router**：先分类后分发——轻量的入口分流模式；
5. **Custom workflow**：LangGraph 自定义流程，可嵌入其他模式作为组件。

选型不只是"结构好不好看"，有明确的**成本含义**：

- Subagents 设计上**无状态**：单请求 4 次模型调用，重复请求每次都从头来；
- Handoffs / Skills **有状态**：重复请求可省 40-50% 调用（上轮的 handoff
  目标或已加载的 skill 直接复用）；
- 多域并行时 Subagents 因上下文隔离，token 总量可显著**低于** Handoffs
  的串行会话累积（隔离省掉了无关上下文的重复传输）。

核心判断标准是 **context engineering**：每个 Agent 该看到什么信息——
而不是"要不要用多个 Agent"。官方同时提醒：不是所有复杂任务都需要多
Agent——一个配好动态工具和 prompt 的单 Agent 往往就够；多 Agent 的
复杂度（状态同步、错误传播、调试难度）要先付出，收益要靠任务结构证明。

**练习：** 把 `c09_03` 的子图改为 `checkpointer=True`，验证能否 fork 到子图
内部两个 interrupt 之间；再创建两个 per-thread 子图验证 namespace 隔离。

### 9.8 本章总结

子图解决的是边界问题，多 Agent 解决的是协作问题。先定义 State 和权限边界，再
决定用哪个模式；不要先创建多个 Agent 再倒推职责。

### 9.9 章末自测

1. 直接注册子图与节点内调用子图分别适合什么 Schema？
2. 子图 namespace 为什么重要？
3. Supervisor 相比固定 Handoff 多解决了什么问题？
4. Agent-as-Tool 的主要嵌套风险是什么？

---

## 第 10 章 测试、数据 Agent、应用结构与综合项目

对应目录：`c10_*`，共 5 个示例。

### 10.1 本章目标

- 测试节点契约和图级行为；
- 使用局部执行降低调试成本；
- 构建 Agentic RAG；
- 构建带人工审核的只读 SQL Agent；
- 理解应用结构、本地 Server 和 Studio；
- 把前九章能力组合成可恢复的端到端项目。

官方依据：

- [Test](https://docs.langchain.com/oss/python/langgraph/test)
- [Build a custom RAG agent with LangGraph](https://docs.langchain.com/oss/python/langgraph/agentic-rag)
- [Build a custom SQL agent](https://docs.langchain.com/oss/python/langgraph/sql-agent)
- [Application structure](https://docs.langchain.com/oss/python/langgraph/application-structure)
- [Run a local server](https://docs.langchain.com/oss/python/langgraph/local-server)

### 10.2 测试策略

按成本从低到高分层：

1. **纯函数测试**：节点输入输出契约；
2. **单节点测试**：直接调用图节点，绕过不需要的整个流程；
3. **局部图测试**：从指定状态开始，到指定节点后暂停；
4. **整图离线测试**：使用 FakeModel 验证路由、循环和终止；
5. **在线 Smoke Test**：验证 Provider、工具和真实模型兼容性；
6. **评估**：用数据集评估路由、工具选择、答案质量和安全。

不要只断言模型原文。优先断言 State、路由、消息角色、工具调用结构、边界条件和
最终业务对象。

### 10.3 `c10_01_testing_partial_execution.py`：节点测试与局部执行

运行：

```bash
venv/bin/python langgraph-demo/c10_01_testing_partial_execution.py --offline
```

**执行链路：**

1. 直接调用 `step_one` 和 `step_two` 验证纯函数契约；
2. 使用 `interrupt_after=["step_one"]` 执行前半段；
3. 使用 `invoke(None, config=...)` 从 Checkpoint 继续。

**关键 API：** Checkpointer、`interrupt_after`、节点级测试。

**练习：** 增加失败节点，测试错误路径和恢复路径。

### 10.4 `c10_02_agentic_rag.py`：Agentic RAG

运行：

```bash
venv/bin/python langgraph-demo/c10_02_agentic_rag.py --offline
```

**执行链路：**

```text
retrieve -> grade -> relevant?
                    | yes -> answer
                    | no  -> rewrite -> retrieve ...
```

**比基础 RAG 多出的能力：**

- 检索结果相关性和评分；
- 首次检索失败后重写问题；
- 最多重试次数；
- 资料不足时明确说明；
- 可扩展引用和命中分数。

**生产风险：**

- 相似度不等于事实正确；
- 重写循环可能消耗过多调用；
- 文档权限和租户隔离；
- Prompt Injection；
- 引用不可追溯；
- 向量索引更新和删除策略。

**练习：** 输出每条引用的文档 ID、分数和最终答案使用到的证据。

### 10.5 `c10_03_sql_agent_hitl.py`：只读 SQL 与人工审批

运行：

```bash
venv/bin/python langgraph-demo/c10_03_sql_agent_hitl.py --offline
```

**执行链路：**

```text
plan SQL -> interrupt 人工审批 -> 只读校验 -> SQLite 执行 -> rows
                            \-> 拒绝 -> denied
```

**两层防护：**

- 应用层：只允许 `SELECT`，检查危险写入关键字；
- 人工层：高风险查询执行前审批。

**生产要求：**

- 数据库使用真正的只读账号；
- 设置查询超时和最大返回行数；
- SQL 解析优先于字符串判断；
- 记录 SQL、审批人、时间和结果摘要；
- 不把生产凭据直接暴露给模型或图节点。

**练习：** 增加行数上限、超时和 SQL 审计记录。

### 10.6 `c10_04_application_local_server.py`：应用结构与本地服务

运行：

```bash
venv/bin/python langgraph-demo/c10_04_application_local_server.py --offline
```

脚本只检查依赖和推荐目录，不会在验证时启动服务。

**典型应用结构：**

```text
my-app/
├── my_agent/
│   ├── agent.py       # 构建图
│   ├── state.py       # State Schema
│   ├── tools.py       # 工具
│   └── utils/
├── .env
├── requirements.txt
└── langgraph.json
```

**`langgraph.json` 的职责：**

- 指定依赖；
- 声明要加载的 graph 名称和路径；
- 指定环境变量文件。

示例：

```json
{
  "dependencies": ["langchain_openai", "./my_agent"],
  "graphs": {
    "agent": "./my_agent/agent.py:graph"
  },
  "env": "./.env"
}
```

**本地开发流程：**

1. 安装 `langgraph-cli`；
2. 创建应用目录；
3. 安装项目依赖；
4. 创建 `.env`；
5. 执行 `langgraph dev`；
6. 在 Studio 或 API 中测试。

**安全要求：** 密钥不能提交进仓库；部署环境需要独立认证、授权、配额和日志。

**练习：** 创建一个最小 `langgraph.json`，让 Studio 能发现一个编译图。

### 10.7 `c10_05_capstone_research_assistant.py`：综合研究助手

运行：

```bash
venv/bin/python langgraph-demo/c10_05_capstone_research_assistant.py --offline
```

**综合链路：**

```text
route -> research tool -> human approval -> answer
      \-> general ----------------------------> answer
```

**覆盖能力：**

- Pydantic 结构化路由；
- 工具调用；
- `interrupt()` 人工审批；
- `InMemorySaver` 持久化；
- `Command(resume=...)` 恢复；
- 最终生成和拒绝路径。

**生产化检查清单：**

1. 工具权限是否最小化？
2. 恢复后的副作用是否幂等？
3. 模型、工具和总流程是否有预算？
4. 是否记录路由、工具、审批和最终答案？
5. 能否从失败 Checkpoint 恢复？
6. 敏感数据是否脱敏？
7. 是否能用离线数据评估质量？
8. 是否有超时、重试和降级？

**练习：** 把最终回答改成流式输出，并增加模型调用次数和 Token 统计。

### 10.8 本章总结

生产级 LangGraph 应用不是单个“大图”，而是以下能力的组合：

```text
清晰 State
  + 明确控制流
  + 持久化与恢复
  + HITL
  + 流式反馈
  + 测试与评估
  + 权限、审计、成本和安全边界
```

### 10.9 章末自测

1. 为什么优先测试节点契约而不是模型原文？
2. Agentic RAG 的循环必须有哪两个上限？
3. SQL Agent 为什么不能只依赖字符串校验？
4. `langgraph.json` 在部署中承担什么职责？
5. Capstone 从演示项目升级到生产还需要哪些能力？

---

## 附录 A：53 个示例总索引

| 章节 | 示例 | 一句话知识点 |
| --- | --- | --- |
| c01 | `c01_01_env_check.py` | 环境、依赖和配置自检 |
| c01 | `c01_02_hello_state_graph.py` | 最小 StateGraph |
| c01 | `c01_03_graph_vs_functional.py` | Graph 与 Functional API 对比 |
| c02 | `c02_01_state_schemas.py` | TypedDict、Pydantic、MessagesState |
| c02 | `c02_02_reducers_messages.py` | Reducer 与消息合并 |
| c02 | `c02_03_input_output_private_state.py` | 输入、输出和私有状态 |
| c02 | `c02_04_nodes_edges.py` | 普通边、条件边、START、END |
| c02 | `c02_05_compile_visualize.py` | 编译与 Mermaid 可视化 |
| c03 | `c03_01_branch_parallel.py` | 并行分支与 Reducer 聚合 |
| c03 | `c03_02_conditional_routing.py` | 条件路由 |
| c03 | `c03_03_loops_recursion.py` | 循环和递归限制 |
| c03 | `c03_04_send_map_reduce.py` | Send 动态 Map-Reduce |
| c03 | `c03_05_command.py` | Command 更新与跳转 |
| c03 | `c03_06_runtime_retry_timeout_cache.py` | Runtime、Retry、Timeout、Cache |
| c04 | `c04_01_entrypoint_task.py` | Entrypoint 与 Task |
| c04 | `c04_02_futures_parallel.py` | Future 并行 |
| c04 | `c04_03_retry_resume.py` | Task 重试与恢复 |
| c04 | `c04_04_graph_functional_interop.py` | 两种 API 互调 |
| c05 | `c05_01_structured_tools_warmup.py` | 结构化输出与工具调用 |
| c05 | `c05_02_prompt_chaining.py` | Prompt Chaining |
| c05 | `c05_03_parallelization.py` | LLM 并行化 |
| c05 | `c05_04_routing.py` | 任务路由 |
| c05 | `c05_05_orchestrator_worker.py` | Orchestrator-Worker |
| c05 | `c05_06_evaluator_optimizer.py` | Evaluator-Optimizer |
| c05 | `c05_07_react_tool_agent.py` | ReAct Tool Agent |
| c06 | `c06_01_thread_checkpointer.py` | thread_id 与短期记忆 |
| c06 | `c06_02_state_history_replay.py` | 历史状态与 Replay |
| c06 | `c06_03_fork_time_travel.py` | Fork 时间旅行 |
| c06 | `c06_04_sqlite_checkpointer.py` | SQLite 跨进程持久化 |
| c06 | `c06_05_store_long_term_memory.py` | Store 长期记忆 |
| c06 | `c06_06_memory_management.py` | 消息裁剪、删除与摘要 |
| c07 | `c07_01_interrupt_resume.py` | Interrupt 与 Resume |
| c07 | `c07_02_approve_reject.py` | 批准与拒绝 |
| c07 | `c07_03_review_edit_state.py` | 人工审核和修改状态 |
| c07 | `c07_04_multiple_tool_interrupts.py` | 多个中断 |
| c07 | `c07_05_retries_timeouts_errors.py` | 重试、超时和错误处理 |
| c07 | `c07_06_graceful_drain.py` | Graceful Drain |
| c07 | `c07_07_error_reproduction.py` | 常见错误复现 |
| c08 | `c08_01_stream_values_updates.py` | values 与 updates |
| c08 | `c08_02_stream_messages.py` | 消息与 Token 流 |
| c08 | `c08_03_custom_tasks_checkpoints.py` | custom/tasks/debug/checkpoints |
| c08 | `c08_04_multiple_modes_subgraphs.py` | 多模式流与子图 namespace |
| c08 | `c08_05_event_streaming.py` | v3 Event Streaming |
| c08 | `c08_06_langsmith_tracing.py` | LangSmith Tracing |
| c09 | `c09_01_subgraph_as_node.py` | 编译子图作为节点 |
| c09 | `c09_02_subgraph_in_node.py` | 节点内调用子图 |
| c09 | `c09_03_subgraph_persistence_stream.py` | 子图持久化与流 |
| c09 | `c09_04_multi_agent_patterns.py` | 四种多 Agent 模式 |
| c10 | `c10_01_testing_partial_execution.py` | 测试与局部执行 |
| c10 | `c10_02_agentic_rag.py` | Agentic RAG |
| c10 | `c10_03_sql_agent_hitl.py` | SQL Agent 与 HITL |
| c10 | `c10_04_application_local_server.py` | 应用结构和本地服务 |
| c10 | `c10_05_capstone_research_assistant.py` | 综合研究助手 |

## 附录 B：核心 API 速查

### 图构建

| API | 作用 |
| --- | --- |
| `StateGraph(StateSchema)` | 创建 Graph Builder |
| `add_node(name, function, ...)` | 注册节点与策略 |
| `add_edge(a, b)` | 固定边 |
| `add_conditional_edges(a, router, mapping)` | 条件路由 |
| `set_entry_point()` | 设置入口 |
| `compile(checkpointer=..., store=..., cache=...)` | 编译为可执行图 |
| `get_graph().draw_mermaid()` | 输出 Mermaid |

### 运行

| API | 作用 |
| --- | --- |
| `invoke(input, config=...)` | 同步执行到结束 |
| `ainvoke(...)` | 异步执行 |
| `stream(..., stream_mode=..., version="v2")` | 按模式流式执行 |
| `astream(...)` | 异步流式执行 |
| `stream_events(..., version="v3")` | Event Streaming |
| `graph.nodes[name].invoke(...)` | 单独测试节点 |

### 状态与控制

| API | 作用 |
| --- | --- |
| `Annotated[T, reducer]` | 为字段设置 Reducer |
| `MessagesState` | 带消息 Reducer 的状态 |
| `Send(node, payload)` | 动态 fan-out |
| `Command(update=..., goto=...)` | 节点内更新并跳转 |
| `Command(resume=...)` | 恢复中断 |
| `interrupt(payload)` | 暂停并等待外部输入 |
| `Overwrite(value)` | 绕过 Reducer 强制覆盖 |

### 持久化与可靠性

| API | 作用 |
| --- | --- |
| `InMemorySaver` | 进程内 Checkpointer |
| `SqliteSaver` | SQLite Checkpointer |
| `InMemoryStore` | 进程内长期记忆 Store |
| `get_state(config)` | 最新状态快照 |
| `get_state_history(config)` | 历史状态快照 |
| `update_state(config, values, as_node=...)` | 修改状态或 Fork |
| `RetryPolicy` | 节点/Task 重试 |
| `TimeoutPolicy` | Run / Idle 超时 |
| `CachePolicy` | 节点结果缓存 |
| `RunControl.request_drain()` | 安全排空 |

### Functional API

| API | 作用 |
| --- | --- |
| `@entrypoint()` | 定义工作流入口 |
| `@task` | 定义可重放执行单元 |
| `future.result()` | 同步等待结果 |
| `await future` | 异步等待结果 |

### 运行时注入与上下文（v2 扩充）

| API | 作用 | 详细说明 |
| --- | --- | --- |
| `Runtime[Context]` | 节点内获取运行时上下文 | 声明 `runtime: Runtime[Context]` 参数即可注入；属性包括 `context`（请求级配置，见 3.7.1.1）、`store`（长期记忆）、`execution_info`（重试次数与各类 ID，做重试降级）、`heartbeat()`（刷新空闲超时）、`drain_requested`（优雅排空标志） |
| `RemainingSteps` | State 托管字段 | 在 State 里声明该类型，框架每步自动写入剩余步数预算；节点里读它做"优雅收尾"，避免被动触发 `GRAPH_RECURSION_LIMIT`。只读，不能手动更新（见 3.7.1.2） |
| `ToolRuntime` | 工具内读取图状态与运行时上下文 | 工具函数声明 runtime 参数即可拿到 `runtime.state`（图状态）与 `runtime.context`（请求级配置）；模型看不到这些参数，适合承载鉴权信息（见 5.9.1.2） |
| `get_stream_writer()` | 节点/工具内发 custom 流数据 | 函数内调用拿 writer，写入的数据由 `stream_mode="custom"` 消费；**Python<3.11 不可用**，需改签名声明 `writer: StreamWriter` 参数由框架注入（见 8.10.1.3） |
| `StreamWriter` | async 环境注入的流写入参数 | `get_stream_writer()` 的参数注入替代形式，兼容旧版 Python；同步/异步代码均可用 |

### Functional API 补充（v2 扩充）

| API | 作用 | 详细说明 |
| --- | --- | --- |
| `entrypoint.final[value_type, save_type]` | 分离返回值与 Checkpoint 保存值 | 类型注解两个类型参数：`value=...` 返回给调用方，`save=...` 写入 Checkpoint 作为下次 `previous`；解决"返回增量、保存全量"（或反之）的场景（见 4.6.1.2） |
| `previous` 注入参数 | 同 thread 上次保存值（短期记忆） | entrypoint 签名声明 `previous: Any` 即注入同一 `thread_id` 上次 `save` 的值，首次为 `None`；换 thread 即换记忆（见 4.6.1.1） |
| `@entrypoint(checkpointer=..., store=...)` | 声明持久化与长期记忆 | 不传 checkpointer 则无持久化能力（`previous` 恒 None、interrupt 不可用）；`store` 用于跨 thread 的长期记忆（见 4.6.1.1） |
| `stream.interleave(...)` | 多投影按到达顺序交错消费 | 同步代码同时读多个 v3 投影的入口，事件按到达顺序交错产出；异步代码用多协程并发消费各自投影（见 8.10.1.5） |

### 持久化与可靠性补充（v2 扩充）

| API | 作用 | 详细说明 |
| --- | --- | --- |
| `durability="exit" / "async" / "sync"` | Checkpoint 写入耐久模式 | 控制每个 Super-step 结束后何时落盘：`exit` 只在 run 结束写（崩溃无法从中间恢复）、`async` 异步写（官方默认推荐）、`sync` 同步写完才继续（零丢失）。每次调用可选，不是编译期固定（见 0.7.3） |
| `PostgresSaver / AsyncPostgresSaver` | 生产级 Checkpointer | LangSmith 同款实现；首次使用需 `setup()` 建表，`thread_id` 上限 255 字符，建议 UUID（见 6.6.1.7） |
| `MongoDBSaver`、Cosmos DB Saver | 生产级 Checkpointer | 官方生产推荐矩阵的另两个成员，接口与 PostgresSaver 一致 |
| `JsonPlusSerializer(pickle_fallback=True)` | 序列化回退 | 默认序列化器不支持 pandas DataFrame 等类型时开启 pickle 回退；注意 pickle 反序列化的安全边界，只用于可信环境（见 6.6.1.5） |
| `EncryptedSerializer.from_pycryptodome_aes()` | 状态加密 | 用 `LANGGRAPH_AES_KEY` 环境变量的密钥对 Checkpoint 加解密，传入 checkpointer 的 `serde` 参数生效（见 6.6.1.5） |
| `DeltaChannel` | 增量存储 | `langgraph>=1.2`，beta。对指定通道只存增量 diff，大列表/大文档字段的存储显著下降；beta 期不建议核心业务直接使用（见 6.6.1.6） |
| `BaseCheckpointSaver` | 自定义 Checkpointer 基类 | 实现 put/put_writes/get_tuple/list/delete_thread 五组（各含同步/异步）；序列化必须走 `self.serde`，完成后跑官方 conformance suite 验证一致性（见 6.6.1.8） |
| `PostgresStore / MongoDBStore / RedisStore` | 生产级长期记忆 Store | 实现 `BaseStore`，接口与 `InMemoryStore` 完全一致，开发→生产换后端不改业务代码（见 6.7.1.4） |
| `BaseStore` | 自定义 Store 基类 | 需实现 `aput/aget/adelete/asearch/alist_namespaces` 五个异步方法；值是普通 dict 直接 JSON 序列化，无复杂 serde 层 |
| `store.list_namespaces(prefix, max_depth)` | 枚举命名空间 | 列出某前缀下有哪几类命名空间（如某用户下有哪几类记忆），`max_depth` 限制返回层级（见 6.7.1.2） |
| `TimeoutPolicy(run_timeout=, idle_timeout=, refresh_on=)` | 运行/空闲超时与心跳刷新 | `run_timeout` 是不刷新的墙钟硬上限；`idle_timeout` 遇进度信号即重置；`refresh_on="heartbeat"` 只认 `runtime.heartbeat()`。仅限 async 节点（见 7.9.1.3） |
| `set_node_defaults(...)` | 全图节点默认策略 | 一次配置全图默认 retry/timeout/error_handler/cache；`add_node` 显式参数优先级更高；默认值不被子图继承（见 3.7.1.5） |
| `Command(goto=..., graph=Command.PARENT)` | 子图跳转父图节点 | 子图内直接把控制权交给父图某节点（如错误兜底）；要求父图 State 对被更新的共享键定义 Reducer（见 3.7.1.3） |
| `graph.get_state(config, subgraphs=True)` | 查看子图状态 | 返回的 `tasks[0].state` 是子图自己的快照与 config；要求子图静态可发现（作为节点添加或节点内直接调用）（见 9.7.1.2） |

## 附录 C：常见错误排查顺序

遇到问题不要先改 Prompt，按以下顺序定位：

1. **输入 State 是否满足 Schema？**
2. **节点返回值是否是合法状态增量？**
3. **并行字段是否有 Reducer？**
4. **路由是否覆盖所有可能值？**
5. **循环是否有业务终止条件？**
6. **使用 Interrupt 时是否有 Checkpointer 和相同 thread_id？**
7. **恢复时是否重新执行了不幂等的副作用？**
8. **节点失败时，Retry、Timeout、Error Handler 顺序是否正确？**
9. **流式消费是否按 StreamPart type 或 projection 分派？**
10. **子图 State、Checkpointer 和 namespace 是否符合预期？**
11. **`Command(update=...)` 是否被误用作 invoke 输入？** 恢复多轮对话应传普通
    输入字典；`Command(resume=...)` 只用于恢复 interrupt。误用 `update` 作为
    输入会让图从最新 Checkpoint 恢复后"卡住"。
12. **`thread_id` 是否超过 255 字符？** PostgresSaver 的 thread_id 列有长度
    限制，超长报数据库错误，用 UUID 生成。
13. **Checkpoint 是否无限增长？** 长对话需定期修剪或使用 `DeltaChannel`；
    修剪时必须保留祖先 write 行。
14. **values 流是否泄漏私有通道？** 用 `output_keys` 限制输出通道，或改用
    `updates` 模式。
15. **同步节点是否传了 timeout？** 会在编译期被拒绝；改 async 节点或用
    `asyncio.to_thread` 包装阻塞 I/O。
16. **子图 per-thread 模式是否发生并行写冲突？** 同一 checkpoint namespace
    不支持并行工具调用，用 `ToolCallLimitMiddleware` 限制或禁用并行。
17. **多 Agent 的并行中断是否按 Interrupt ID 映射恢复？** 不能按列表顺序
    隐式配对，必须用 `{interrupt.id: value}` 映射。

## 附录 D：官方文档索引

基础：

- [Overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart)
- [Thinking in LangGraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph)
- [Choosing APIs](https://docs.langchain.com/oss/python/langgraph/choosing-apis)

核心 API：

- [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [Use the graph API](https://docs.langchain.com/oss/python/langgraph/use-graph-api)
- [Functional API](https://docs.langchain.com/oss/python/langgraph/functional-api)
- [Use the functional API](https://docs.langchain.com/oss/python/langgraph/use-functional-api)
- [LangGraph runtime](https://docs.langchain.com/oss/python/langgraph/pregel)

持久化、中断和可靠性：

- [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)
- [Stores](https://docs.langchain.com/oss/python/langgraph/stores)
- [Use time travel](https://docs.langchain.com/oss/python/langgraph/use-time-travel)
- [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [Fault tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance)

Agent、流式和组合：

- [Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [Streaming](https://docs.langchain.com/oss/python/langgraph/streaming)
- [Event streaming](https://docs.langchain.com/oss/python/langgraph/event-streaming)
- [Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)
- [Multi-agent](https://docs.langchain.com/oss/python/langchain/multi-agent)

测试和部署：

- [Test](https://docs.langchain.com/oss/python/langgraph/test)
- [LangSmith Observability](https://docs.langchain.com/oss/python/langgraph/observability)
- [Application structure](https://docs.langchain.com/oss/python/langgraph/application-structure)
- [Run a local server](https://docs.langchain.com/oss/python/langgraph/local-server)
- [Agentic RAG](https://docs.langchain.com/oss/python/langgraph/agentic-rag)
- [SQL agent](https://docs.langchain.com/oss/python/langgraph/sql-agent)

完整索引：

- [LangGraph Python Docs Index](https://docs.langchain.com/oss/python/langgraph/llms.txt)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)

## 附录 E：建议的 14 天学习计划

| 天数 | 内容 | 完成标准 |
| --- | --- | --- |
| Day 1 | 第 0-1 章 | 能画出最小图的执行路径 |
| Day 2 | 第 2 章 State/Reducer | 能解释并行字段为什么需要 Reducer |
| Day 3 | 第 3 章前半 | 能实现分支、循环和 Send |
| Day 4 | 第 3 章后半 | 能配置 Runtime、Retry、Timeout、Cache |
| Day 5 | 第 4 章 Functional API | 能改造一段线性 Python 流程 |
| Day 6 | 第 5 章前半 | 能区分 Chaining、Parallel、Routing |
| Day 7 | 第 5 章后半 | 能实现有界的 ReAct Agent |
| Day 8 | 第 6 章持久化 | 能解释 thread、Checkpoint、Store |
| Day 9 | 第 6 章时间旅行 | 能完成 Replay 和 Fork |
| Day 10 | 第 7 章 HITL/容错 | 能设计批准、恢复和错误处理 |
| Day 11 | 第 8 章 Streaming | 能为主流 UI 选择流模式 |
| Day 12 | 第 9 章子图/多 Agent | 能根据 State 边界选择组合方式 |
| Day 13 | 第 10 章测试/RAG/SQL | 能运行并扩展三个综合示例 |
| Day 14 | Capstone 复盘 | 能列出生产化缺口和改进计划 |
