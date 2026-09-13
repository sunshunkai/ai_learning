# AgentScope 学习模块

本目录学习 AgentScope 2.x 的核心 Python SDK，并统一接入 DeepSeek。
内容以官方 `2.0.9dev` 中文文档和 GitHub 提交 `b82253b` 的
`docs/`、`examples/`、`scripts/model_examples/` 为基准。

目录故意命名为 `agentscope-demo/`，避免与 pip 安装的
`agentscope` 包同名并遮蔽真实库。

逐文件知识整合见 [`KNOWLEDGE_GUIDE.md`](./KNOWLEDGE_GUIDE.md)。

可运行示例统一使用 `c章_序号_名称.py` 命名：

- `c01`：环境与基础
- `c02`：模型能力
- `c03`：Agent 核心
- `c04`：工具与权限
- `c05`：编排与中间件
- `c06`：上下文、RAG 与记忆
- `c07`：工作区与协议集成
- `c08`：Agent Service

`common.py` 是各示例共享的 DeepSeek 模型工厂。

## 安装

AgentScope 需要 Python 3.11+。本项目已有 `venv/`，运行：

```bash
venv/bin/pip install -r agentscope-demo/requirements.txt
```

`requirements.txt` 固定安装 GitHub 源码提交
`b82253b`（对应本文档基线）。若网络较慢，
可以先只安装基础包，再按需安装可选依赖：

```bash
venv/bin/pip install -e /tmp/agentscope-checkout
venv/bin/pip install qdrant-client a2a-sdk
```

验证安装：

```bash
venv/bin/python agentscope-demo/c01_01_env_check.py
```

全部示例可以一键验证：

```bash
venv/bin/python agentscope-demo/verify_examples.py
```

校验脚本会逐个运行普通示例，并自动完成 MCP、A2A 双端和
Agent Service HTTP 流程。

## 配置

项目根目录 `.env` 需要：

```text
DEEPSEEK_API_KEY=你的 DeepSeek Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

所有示例通过 `common.py` 统一创建：

```python
from agentscope.credential import DeepSeekCredential
from agentscope.model import DeepSeekChatModel

model = DeepSeekChatModel(
    credential=DeepSeekCredential(api_key=...),
    model="deepseek-v4-flash",
    stream=True,
    parameters=DeepSeekChatModel.Parameters(
        thinking_enable=True,
        reasoning_effort="high",
    ),
)
```

## 学习路线

### 1. 环境与基础

| 文件 | 学习点 |
| --- | --- |
| `c01_01_env_check.py` | 依赖、模型卡片、Key、Base URL |
| `c01_02_basic_model.py` | 非流式、流式、Thinking |
| `c01_03_message_event.py` | Message、ContentBlock、Event 重建 |

```bash
venv/bin/python agentscope-demo/c01_02_basic_model.py
venv/bin/python agentscope-demo/c01_03_message_event.py
```

### 2. 模型能力

| 文件 | 学习点 |
| --- | --- |
| `c02_01_thinking_chat.py` | `reasoning_effort`、temperature、top_p |
| `c02_02_tool_calling.py` | 模型层单次工具往返 |
| `c02_03_structured_output.py` | `generate_structured_output` |
| `c02_04_multi_agent_formatter.py` | `DeepSeekMultiAgentFormatter` |

```bash
venv/bin/python agentscope-demo/c02_02_tool_calling.py
venv/bin/python agentscope-demo/c02_03_structured_output.py
```

### 3. Agent 核心

| 文件 | 学习点 |
| --- | --- |
| `c03_01_basic_agent.py` | ReAct 循环、AgentState |
| `c03_02_agent_structured_output.py` | Agent 层结构化输出 |
| `c03_03_multi_turn.py` | 多轮上下文、observe |
| `c03_04_streaming_events.py` | 完整事件流与消息重建 |
| `c03_05_state_persistence.py` | AgentState 序列化/恢复 |
| `c03_06_interrupt_agent.py` | asyncio 取消与中断恢复 |

```bash
venv/bin/python agentscope-demo/c03_01_basic_agent.py
venv/bin/python agentscope-demo/c03_04_streaming_events.py
```

### 4. 工具与权限

| 文件 | 学习点 |
| --- | --- |
| `c04_01_function_tool.py` | `FunctionTool` 自动 ReAct |
| `c04_02_tool_middleware.py` | Tool Middleware 洋葱模型 |
| `c04_03_builtin_tools.py` | Bash/Read/Write/Edit |
| `c04_04_permission_hitl.py` | ASK、确认、恢复 |
| `c04_05_ask_user.py` | 外部执行工具 |
| `c04_06_tool_groups.py` | ToolGroup 与 reset_tools |

人工审批支持自动测试：

```bash
AGENTSCOPE_APPROVAL=yes venv/bin/python agentscope-demo/c04_04_permission_hitl.py
AGENTSCOPE_APPROVAL=no  venv/bin/python agentscope-demo/c04_04_permission_hitl.py
```

### 5. 编排与中间件

| 文件 | 学习点 |
| --- | --- |
| `c05_01_plan_mode.py` | TaskCreate/TaskList/TaskUpdate |
| `c05_02_custom_middleware.py` | Agent Middleware 生命周期钩子 |
| `c05_03_goal_pipeline.py` | 执行者/验证者闭环 |

```bash
venv/bin/python agentscope-demo/c05_03_goal_pipeline.py
```

### 6. 上下文、RAG 与记忆

| 文件 | 学习点 |
| --- | --- |
| `c06_01_context_management.py` | Runtime State、压缩、Offloader |
| `c06_02_rag.py` | Parser/Chunker/Embedding/VectorStore/RAGMiddleware |
| `c06_03_long_term_memory.py` | AgenticMemory 跨会话记忆 |

```bash
venv/bin/python agentscope-demo/c06_01_context_management.py
venv/bin/python agentscope-demo/c06_02_rag.py
```

### 7. 工作区与协议集成

| 文件 | 学习点 |
| --- | --- |
| `c07_01_workspace_skill.py` | LocalWorkspace、Skill |
| `c07_02_mcp_server.py` + `c07_03_mcp_tool.py` | 本地 STDIO MCP |
| `c07_04_a2a_server.py` + `c07_05_a2a_client.py` | A2A 1.0 |
| `c07_06_console_chat.py` | launch_console、ConsoleRenderer |

MCP：

```bash
venv/bin/python agentscope-demo/c07_03_mcp_tool.py
```

A2A 需要两个终端：

```bash
# 终端 1
venv/bin/python agentscope-demo/c07_04_a2a_server.py

# 终端 2
venv/bin/python agentscope-demo/c07_05_a2a_client.py
```

Console 非交互回归：

```bash
AGENTSCOPE_MESSAGE="你好，请介绍一下你自己" \
  venv/bin/python agentscope-demo/c07_06_console_chat.py
```

### 8. Agent Service

| 文件 | 学习点 |
| --- | --- |
| `c08_01_agent_service.py` | SQLite + InMemory Bus 的最小 FastAPI 服务 |
| `c08_02_service_smoke.py` | 凭据/Agent/Session/Chat 完整 HTTP 流程 |

```bash
# 终端 1
venv/bin/python agentscope-demo/c08_01_agent_service.py

# 终端 2
venv/bin/python agentscope-demo/c08_02_service_smoke.py
```

服务默认使用临时身份请求头 `X-User-ID`。AgentScope 官方说明该机制
会由 JWT 认证替换，生产环境必须接入自己的鉴权网关。

## DeepSeek 兼容性

| 能力 | DeepSeek 支持情况 | 本目录处理方式 |
| --- | --- | --- |
| Chat、流式、Thinking | 支持 | 直接使用 `DeepSeekChatModel` |
| Tool Calling、结构化输出 | 支持 | 直接使用，模型层会关闭 Thinking 避免服务端拒绝强制工具调用 |
| 多智能体 Formatter | 支持 | `DeepSeekMultiAgentFormatter` |
| Embedding | 不支持 | `c06_02_rag.py` 使用本地哈希嵌入演示完整流程 |
| LLM Rerank | 支持 | 使用 DeepSeek Chat Model 做 RAGMiddleware 重排 |
| Realtime Voice | 不支持 | 知识指南说明 DashScope/OpenAI/Gemini/xAI 路径 |
| TTS | 不支持 | 知识指南说明可替换供应商 |
| MCP、Skill、Workspace、A2A、Console | 与模型无关 | 示例均可运行 |
| Agent Service | 与模型有关但可通过 DeepSeek 凭据配置 | `c08_02_service_smoke.py` 演示 |

## 版本说明

- 官方文档：`https://docs.agentscope.io/versions/2.0.9dev/zh`
- 官方仓库：`https://github.com/agentscope-ai/agentscope`
- 发布日志：`https://docs.agentscope.io/versions/2.0.9dev/zh/release-notes`

文档版本标记为 `2.0.9dev`；代码示例按 GitHub 提交
`b82253b` 的源码验证。
安装后应运行 `c01_01_env_check.py` 检查实际版本，不要把 AgentScope 1.x
与 2.x API 混用。

## 安全提醒

`c04_03_builtin_tools.py` 等工作区示例使用 `PermissionMode.BYPASS` 是为了
让学习脚本无人值守跑通。生产环境应根据场景选择 `DEFAULT`、
`ACCEPT_EDITS`、`EXPLORE` 或 `DONT_ASK`，并为敏感路径配置 deny rule。
