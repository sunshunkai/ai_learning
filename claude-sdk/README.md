# Claude SDK 学习模块

本目录学习 Anthropic 官方 Claude SDK（Python 版），覆盖基础调用、流式、
工具调用、人工审批、结构化输出、异步、异常处理和 thinking 等能力。

## 前置：安装依赖

```bash
venv/bin/pip install -r claude-sdk/requirements.txt
```

## 前置：配置 API Key

脚本统一通过项目根目录 `.env` 或系统环境变量读取：

```text
ANTHROPIC_API_KEY=你的Key
ANTHROPIC_BASE_URL=接口地址
ANTHROPIC_MODEL=模型名
```

如果使用 DeepSeek 的 Anthropic 兼容端点：

```text
ANTHROPIC_API_KEY=你的 DeepSeek Key
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
```

`ANTHROPIC_MODEL` 未配置时默认使用 `claude-3-5-sonnet-latest`；
DeepSeek 会自动映射到自己的模型。

## 学习顺序

### 基础

1. `env_check.py`：检查依赖、Key 和 Base URL
2. `basic_chat.py`：最小的一问一答
3. `multi_turn.py`：手动维护完整消息历史
4. `stream_chat.py`：`messages.stream()` 流式文本
5. `tool_use.py`：单工具手动执行和 tool_result 回传

### 工具与 Human-in-the-loop

6. `human_approval.py`：高风险工具执行前暂停，人工批准或拒绝
7. `multi_tool_loop.py`：多工具调用、统一分发、工具异常和循环控制
8. `structured_output.py`：通过 tool_choice 强制输出结构化 JSON

### 并发、流与参数

9. `async_chat.py`：`AsyncAnthropic`、异步调用和异步流
10. `stream_events.py`：遍历 message/content_block 等底层流事件
11. `parameter_control.py`：system、stop_sequences、metadata 和兼容参数

### 生产实践

12. `error_handling.py`：超时、重试、限流和连接错误处理
13. `thinking_chat.py`：解析 extended thinking 与最终文本块
14. `prompt_caching.py`：cache_control 与缓存 token 统计
15. `interactive_chat.py`：带 `/exit`、`/clear`、`/history` 的交互式会话

`common.py` 是公共工具，负责加载 `.env`、创建同步/异步客户端和提取文本。

## 运行示例

```bash
venv/bin/python claude-sdk/env_check.py
venv/bin/python claude-sdk/basic_chat.py
venv/bin/python claude-sdk/human_approval.py
venv/bin/python claude-sdk/multi_tool_loop.py
venv/bin/python claude-sdk/async_chat.py
```

人工审批案例支持自动测试：

```bash
CLAUDE_DEMO_APPROVAL=yes venv/bin/python claude-sdk/human_approval.py
CLAUDE_DEMO_APPROVAL=no  venv/bin/python claude-sdk/human_approval.py
```

## DeepSeek 兼容性说明

- `structured_output.py` 使用 `tool_choice={"type":"any"}`，因为 DeepSeek
  thinking 模式不支持指定具体工具名。
- `thinking_chat.py` 的 `budget_tokens` 可能被兼容端点忽略。
- `prompt_caching.py` 在官方 Anthropic 上体现缓存命中；DeepSeek 目前会忽略
  `cache_control`，缓存 token 统计通常为 0。
- Files、Batch、Token Counting、MCP 等属于 Anthropic 官方差异能力，使用前需检查
  当前中转服务是否支持。

## 关于 API Key

所有示例都不会硬编码 Key，统一从环境变量读取，避免把密钥提交到 Git。
