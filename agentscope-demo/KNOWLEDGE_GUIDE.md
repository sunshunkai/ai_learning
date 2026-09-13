# AgentScope 2.0.9dev 知识整合指南

> 本文把官方中文文档、GitHub `b82253b` 源码、官方 examples 和本目录
> 示例整合成一条学习主线。代码默认接入 DeepSeek，差异能力单独标注。

## 1. 总体架构

AgentScope 2.x 分为两层：

```text
SDK 层：Agent / Model / Tool / Middleware / Context / Workspace
服务层：FastAPI / Session / Team / RAG / Channel / Hub / Scheduler
```

SDK 层解决“如何构建和运行一个智能体”，服务层解决“如何把智能体变成
多租户、多会话、可持久化的 HTTP 应用”。本目录以 SDK 层为主，
第 8 章给出最小服务。

官方定位：

- 核心抽象是**无状态的推理-行动循环引擎** `Agent`
- 不把能力锁死在提示词里，而是通过模型、工具、权限、事件和中间件组合
- 消息是完整轮次，事件是流式增量；事件可以无损重建消息

对应示例：`c03_01_basic_agent.py`、`c08_01_agent_service.py`

## 2. 消息与事件

### Message

`Msg` 字段：`id`、`name`、`role`、`content`、`metadata`、
`created_at`、`finished_at`、`usage`、`finished_reason`、
`structured_output`、`error`。

快捷工厂：

```python
from agentscope.message import UserMsg, AssistantMsg, SystemMsg

UserMsg(name="user", content="你好")
AssistantMsg(name="Friday", content="你好")
SystemMsg(name="system", content="系统提示")
```

`content` 是类型化内容块：

| Block | 用途 |
| --- | --- |
| `TextBlock` | 纯文本 |
| `ThinkingBlock` | 模型推理 |
| `DataBlock` | 图片、音频、视频等二进制数据 |
| `ToolCallBlock` | 工具调用请求 |
| `ToolResultBlock` | 工具结果 |
| `HintBlock` | 系统提醒、RAG 命中、团队消息等 |

角色约束：

- user 只允许 `TextBlock` / `DataBlock`
- system 只允许 `TextBlock`
- assistant 可包含所有类型

常用读取方法：

```python
msg.get_text_content()
msg.get_content_blocks("tool_call")
msg.has_content_blocks("tool_result")
```

对应示例：`c01_03_message_event.py`

### Event

`reply_stream` 产出的事件有完整生命周期：

```text
ReplyStart
  ModelCallStart
    ThinkingBlockStart -> Delta -> End
    TextBlockStart -> Delta -> End
  ModelCallEnd
  ToolCallStart -> ToolCallDelta -> ToolCallEnd
  ToolResultStart -> ToolResultDelta -> ToolResultEnd
ReplyEnd
```

特殊事件：

| 事件 | 作用 |
| --- | --- |
| `RequireUserConfirmEvent` | 权限系统请求人工确认 |
| `RequireExternalExecutionEvent` | 外部执行工具暂停 |
| `UserConfirmResultEvent` | 回传确认结果并恢复 |
| `ExternalExecutionResultEvent` | 回传外部执行结果并恢复 |
| `UserInterruptEvent` | 中止一个处于暂停状态的回复 |
| `HintBlockEvent` | 上下文注入或后台消息到达 |

事件可以逐步 `append_event()` 到一个 `AssistantMsg`，最终得到完整消息。
这个特性支撑 SSE 前端、断线重放和审计。

对应示例：`c03_04_streaming_events.py`、`c07_06_console_chat.py`

## 3. Agent 核心

### 构造

```python
agent = Agent(
    name="Friday",
    system_prompt="...",
    model=model,
    toolkit=toolkit,
    middlewares=[...],
    state=state,
    offloader=workspace,
    model_config=...,
    context_config=...,
    react_config=...,
    injection_config=...,
)
```

核心接口：

- `reply(inputs, structured_schema)`：运行循环，返回最终消息
- `reply_stream(inputs, structured_schema, yield_final_msg)`：流式事件
- `observe(msgs)`：只写上下文，不触发推理
- `compress_context(...)`：按阈值压缩上下文

主循环每轮选择：推理、行动、暂停等待外部交互、结束。

对应示例：`c03_01_basic_agent.py`

### ReActConfig

重要字段：

- `max_iters`：一轮回复最多推理-行动次数
- `structured_output_grace_iters`：结构化输出额外重试
- `stop_on_reject`：工具被拒绝后是否停止
- `interruption_message`：被中断时的回退文本
- `interruption_raise_cancelled_error`

### AgentState

包含：

- `session_id`
- `reply_id` / `reply_context`
- `context`
- `summary`
- `permission_context`
- `tool_context`
- `tasks_context`
- `middle_context`

它继承 Pydantic `BaseModel`，可用 `model_dump_json()` 和
`model_validate_json()` 持久化。

对应示例：`c03_03_multi_turn.py`、`c03_05_state_persistence.py`

### 结构化输出

```python
reply = await agent.reply(
    user_msg,
    structured_schema=MyModel,
)
print(reply.structured_output)
```

AgentScope 会把 schema 转成内部强制工具调用，再校验和修复 JSON。

对应示例：`c03_02_agent_structured_output.py`

### 中断

- 运行中：`task.cancel()`
- 已暂停：`UserInterruptEvent(reply_id=agent.state.reply_id)`

中断后 Agent 保留部分内容，标记 `finished_reason=INTERRUPTED`，
会话仍可继续。

对应示例：`c03_06_interrupt_agent.py`

## 4. 模型层

### 凭证与模型卡片

模型由“凭证 + 模型名 + Parameters”组成。DeepSeek：

```python
from agentscope.credential import DeepSeekCredential
from agentscope.model import DeepSeekChatModel

model = DeepSeekChatModel(
    credential=DeepSeekCredential(api_key="..."),
    model="deepseek-v4-flash",
    stream=True,
    context_size=1_000_000,
    parameters=DeepSeekChatModel.Parameters(
        thinking_enable=True,
        reasoning_effort="high",
        temperature=0.3,
        max_tokens=2000,
    ),
)
```

模型卡片通过 YAML 描述：

- 支持输入/输出类型
- 上下文长度、输出长度
- 参数 override

`list_models()` 可发现可用模型。

对应示例：`c01_01_env_check.py`、`c01_02_basic_model.py`

### 流式契约

`stream=True` 时 `await model(messages)` 返回 async generator：

- 中间 chunk：`is_last=False`，只带增量
- 最后 chunk：`is_last=True`，带完整累计内容

非流式返回单个 `ChatResponse`。

### DeepSeek Thinking

- `thinking_enable=True` 开启 `reasoning_content`
- `reasoning_effort`：`high` / `max`
- AgentScope 转换为 `ThinkingBlock`

对应示例：`c02_01_thinking_chat.py`

### Formatter

默认 `DeepSeekChatFormatter` 适合单智能体。
`DeepSeekMultiAgentFormatter` 会把多智能体历史包进 `<history>`。

对应示例：`c02_04_multi_agent_formatter.py`

### 自定义 Provider

实现 `CredentialBase` 和 `ChatModelBase` 即可接入任意 API：

1. 凭证定义唯一 `type`，返回对应模型类
2. 模型实现 `_call_api`
3. 可选添加 `_models/*.yaml`

## 5. 工具系统

### ToolBase

核心字段：

- `name`、`description`、`input_schema`
- `is_concurrency_safe`
- `is_read_only`
- `is_external_tool`
- `is_state_injected`
- `is_mcp`

核心方法：

- `check_permissions(tool_input, context)`
- `check_read_only(tool_input)`
- `match_rule(rule_content, tool_input)`
- `generate_suggestions(tool_input)`
- `call(**kwargs)`

### FunctionTool

自动从函数名、docstring、类型注解推导 schema。默认权限为 ASK。

```python
FunctionTool(
    get_weather,
    permission=PermissionDecision(
        behavior=PermissionBehavior.ALLOW,
        message="read-only",
    ),
)
```

对应示例：`c04_01_function_tool.py`

### 内置工具

| 工具 | 说明 |
| --- | --- |
| `Bash` | 命令执行，带静态安全分析 |
| `Read/Write/Edit` | 文件工具，强制先读后写 |
| `Glob/Grep` | 文件与内容搜索 |
| `TaskCreate/TaskGet/TaskList/TaskUpdate` | 计划任务 |
| `AskUser` | 外部交互问题 |
| `reset_tools` | 元工具，动态激活工具组 |

对应示例：`c04_03_builtin_tools.py`、`c05_01_plan_mode.py`、`c04_05_ask_user.py`、
`c04_06_tool_groups.py`

### Tool Middleware

工具级洋葱钩子，只包裹 `tool.call()`，即使不经过 Agent 也生效。

对应示例：`c04_02_tool_middleware.py`

### 外部执行工具

设置 `is_external_tool=True`，Agent 调用时发出
`RequireExternalExecutionEvent`，由宿主系统执行并回传结果。

## 6. 权限系统

### PermissionMode

| 模式 | 行为 |
| --- | --- |
| `DEFAULT` | 默认最安全，通常需要确认 |
| `ACCEPT_EDITS` | 自动放行工作目录内读写与受控文件命令 |
| `EXPLORE` | 只读 |
| `BYPASS` | 跳过安全 ASK，仅 deny/ask 规则仍生效 |
| `DONT_ASK` | 无人值守，把 ASK 转成 DENY |

### PermissionRule

规则包含：

- `tool_name`
- `rule_content`
- `behavior`
- `source`

Bash 使用前缀通配，文件工具使用 glob。Deny/ask 规则在所有模式下
都优先。

### Safety ASK

工具可以返回 `bypass_immune=True` 的 ASK，表示 allow rule 也不能静默
覆盖。DEFAULT/ACCEPT_EDITS 会强制询问，DONT_ASK 会拒绝，BYPASS 明确
跳过。

对应示例：`c04_04_permission_hitl.py`

## 7. 上下文管理

### 三层结构

```text
System Prompt
  + Skill 指令
  + on_system_prompt 中间件
Summary（压缩历史）
Context（最近未压缩消息）
```

### Runtime State Injection

默认注入：

- 当前时间与时区
- 计划任务
- 上下文使用率
- 同一工具连续失败提示

使用 `InjectionConfig` 控制，`extra_fields` 注入业务字段。

### Context Compression

`ContextConfig`：

- `trigger_ratio`：开始压缩阈值
- `reserve_ratio`：保留的最近上下文比例
- `tool_result_limit`：单条工具结果 token 上限
- `compression_tool_enabled`：允许 Agent 自主触发
- `max_image_num`：图片数量限制

压缩失败可回退为截断。

### Offloader

`LocalWorkspace` 等实现：

```python
offload_context(session_id, msgs)
offload_tool_result(session_id, tool_result)
```

被压缩/截断内容写回工作区，模型仍可通过文件工具回查。

对应示例：`c06_01_context_management.py`

## 8. Agent Middleware

七个扩展点：

| Hook | 类型 |
| --- | --- |
| `on_reply` | 洋葱，包住完整回复 |
| `on_reasoning` | 洋葱，包住单轮推理 |
| `on_acting` | 洋葱，包住工具执行 |
| `on_model_call` | 洋葱，最接近模型 API |
| `on_check_permission` | 权限判断拦截 |
| `on_compress_context` | 上下文压缩拦截 |
| `on_system_prompt` | Transformer，顺序改写提示词 |

`list_tools()` 可让中间件提供工具，但不会自动注册。

`on_reply` 可以吞掉 `ReplyEndEvent` 强制再来一轮，但必须防止死循环。

内置中间件：

- `TracingMiddleware`
- `ReplyBudgetControlMiddleware`
- `TTSMiddleware`
- `AgenticMemoryMiddleware`
- `Mem0Middleware`
- `ReMeMiddleware`
- `RAGMiddleware`

对应示例：`c05_02_custom_middleware.py`

## 9. Plan 与 Pipeline

### Plan

任务工具共享 `agent.state.tasks_context`：

```text
pending -> in_progress -> completed
```

任务还支持 `blocks` / `blocked_by` 依赖。

对应示例：`c05_01_plan_mode.py`

### GoalPipeline

执行者返回结构化 `report`，验证者返回 `pass/fail/impossible`。
fail 的 `message` 回给执行者继续修正。

参数：

- `verifier_reset_context`
- `max_iters`
- `max_retries`

对应示例：`c05_03_goal_pipeline.py`

## 10. RAG

流水线：

```text
bytes -> Parser -> Section[] -> Chunker -> Chunk[]
      -> Embedding -> Vector Store -> KnowledgeBase
```

组件：

- Parser：Text/PDF/PPT/Word/Excel/Image
- Chunker：`ApproxTokenChunker`
- Embedding：DashScope/OpenAI/Gemini/Ollama
- Vector Store：Qdrant/Milvus/MongoDB/Elasticsearch

`KnowledgeBase` 统一提供：

```python
insert_document()
search()
list_documents()
delete_document()
```

`RAGMiddleware` 两种模式：

- `static`：首轮自动检索并注入 `HintBlock`
- `agentic`：模型自主调用 `search_knowledge`

2.0.8 起支持传入 `rerank_model`，用 LLM 重排向量检索候选。

DeepSeek 没有 Embedding API，本目录 `c06_02_rag.py` 使用本地哈希嵌入教学；
生产可替换为 DashScope/OpenAI Embedding，并把 DeepSeek 保留为重排器。

## 11. 长期记忆

### AgenticMemory

Markdown 文件记忆：

```text
Memory/
├── MEMORY.md
├── user_profile.md
└── feedback_answer_style.md
```

模型自主用 `Read/Write/Edit` 维护记忆。跨会话共享 workdir 即可共享记忆。

对应示例：`c06_03_long_term_memory.py`

### ReMe

- 文件原生、进程内
- 自动从对话写回，无需 add 工具
- 检索模式：static_control / agent_control / both
- 需要 `agentscope[memory-reme]`

### Mem0

- 支持 `mem0.AsyncMemory` 或 `mem0.AsyncMemoryClient`
- 有 `search_memory` / `add_memory`
- 需要 `agentscope[memory-mem0]`

## 12. Workspace

Workspace 同时承担：

- 内置文件/Shell 工具后端
- MCP 与 Skill 隔离
- Offloader
- 沙箱运行环境

后端：

- Local
- Docker
- E2B
- Apple Container
- Bubblewrap
- Daytona
- K8s
- OpenSandbox

沙箱内 MCP 通过内部 MCP Gateway 暴露，主机侧工具接口不变。

对应示例：`c07_01_workspace_skill.py`

## 13. MCP 与 Skill

### MCP

两种连接：

- Stateful：STDIO / HTTP，显式 `connect()` / `close()`
- Stateless：HTTP，每次调用临时会话

工具名形如：

```text
mcp__{server_name}__{tool_name}
```

对应示例：`c07_02_mcp_server.py`、`c07_03_mcp_tool.py`

### Skill

Skill 是 `SKILL.md` 指令集，不是直接调用对象。Agent 先调用 SkillViewer
读取完整 Markdown，再用已有工具执行。

## 14. A2A 协议

`A2AAgent` 是远端 A2A 智能体的本地代理：

- `reply/reply_stream` 与本地 Agent 接口一致
- `observe` 缓冲消息
- `compress_context` 是 no-op
- 模型、工具、权限由远端服务决定

关键映射：

- A2A `context_id` 相当于远端 session
- A2A `task_id` 不直接等于一次 `reply`
- Text Part -> `TextBlock`
- Bytes/URL Part -> `DataBlock`

对应示例：`c07_04_a2a_server.py`、`c07_05_a2a_client.py`

## 15. Console

- `launch_console`：交互输入、流式渲染、工具确认、Ctrl+C
- `ConsoleRenderer`：嵌入自有事件循环，只负责渲染
- verbosity：quiet / default / debug

对应示例：`c07_06_console_chat.py`

## 16. Realtime Voice

2.0.8 新增：

- `RealtimeAgent`
- `agentscope.realtime`
- 打断、工具调用、轮次延迟统计

内置模型支持 DashScope Qwen 实时语音模型，文档同时说明 OpenAI/Gemini/
xAI 支持。DeepSeek 当前不提供实时语音 API，因此本目录不伪造一个
DeepSeek 语音示例；集成方式见官方
`https://docs.agentscope.io/versions/2.0.9dev/zh/building-blocks/realtime/`。

## 17. Agent Service

`create_app` 组装 FastAPI 服务：

```python
create_app(
    storage=...,
    message_bus=...,
    workspace_manager=...,
    knowledge_base_manager=...,
    channels=...,
    mcp_hubs=...,
    skill_hubs=...,
)
```

能力：

- 多租户、多会话
- Agent / Session / Chat API
- Team：Leader 与 worker
- SSE 事件流
- RAG 服务、索引 worker
- 工作区管理
- 飞书、钉钉、Discord 渠道
- MCP/Skill Hub
- 定时任务
- 资源共享

最小本地实现用 SQLite，避免 Redis/Docker。

对应示例：`c08_01_agent_service.py`、`c08_02_service_smoke.py`

## 18. 2.0.8 发布日志关键变化

### Realtime

- `RealtimeAgent` 与 `agentscope.realtime`
- 打断、工具调用、延迟统计
- DashScope Qwen-Omni/Qwen-Audio 实时模型

### A2A

- 新增 `A2AAgent`

### Pipeline

- 新增 `GoalPipeline`

### Agent

- `CompressContext` 自主压缩工具
- 迭代耗尽时收尾总结
- 重复工具失败提示

### Model

- 火山方舟 `VolcengineChatModel`

### RAG

- LLM rerank

### Workspace / Service

- Agent 级 MCP 会话与 Skill 隔离
- 工作区预热
- `TeamMemberLoopMiddleware`
- 钉钉渠道
- SQL 渠道持久化
- 自动会话命名

### 修复重点

- 上下文压缩 token 计入用量
- 文件工具读缓存并发修复
- 任务 blocked_by 清理
- 距离型检索分数归一化

## 19. 与 Claude SDK 的对照

| 概念 | Claude SDK | AgentScope |
| --- | --- | --- |
| 模型调用 | `client.messages.create` | `await model(messages)` |
| 流式 | `client.messages.stream` | `model(stream=True)` async generator |
| 工具 | 手工 tool_use/tool_result | Agent 自动 ReAct 循环 |
| 权限 | 自己实现审批 | PermissionEngine + HITL Event |
| 结构化输出 | `tool_choice` | `generate_structured_output` |
| 记忆 | 自己维护 messages | AgentState + 长期记忆中间件 |
| 上下文压缩 | 自己实现 | `ContextConfig` + Offloader |
| 多智能体 | 无原生抽象 | MultiAgentFormatter、Team、Pipeline |
| 服务化 | 无内置 | `create_app` + FastAPI |

Claude SDK 更接近底层 API，AgentScope 更接近 Agent 运行框架；
AgentScope 的事件系统和权限系统是主要新增复杂度。

## 20. DeepSeek 差异清单

```text
支持：
- deepseek-v4-flash / deepseek-v4-pro
- Chat Completions 兼容接口
- Thinking / reasoning_effort
- Function Calling
- Structured Output

不支持：
- Embedding API
- Realtime Audio API
- TTS API
- Anthropic 风格的 prompt cache
```

接入时：

1. 用 `DeepSeekCredential` + `DeepSeekChatModel`
2. 结构化输出与 Agent 内部需要时显式关闭 Thinking
3. Embedding 换成第三方或本地模型
4. Realtime/TTS 换成 DashScope、OpenAI、Gemini、xAI 等
