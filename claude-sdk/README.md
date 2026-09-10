# Claude SDK 学习模块

本目录用来学习 Anthropic 官方 Claude SDK（Python 版）。

## 文件说明（建议按顺序学习）

1. `env_check.py`    —— 自检 API Key 是否配置好（先跑这个）
2. `basic_chat.py`   —— 一问一答（最基础的调用）
3. `multi_turn.py`   —— 多轮对话（手动维护上下文）
4. `stream_chat.py`  —— 流式输出（打字机效果）
5. `tool_use.py`     —— 工具调用 / Function Calling

## 前置：安装依赖

```bash
pip install -r claude-sdk/requirements.txt   # 即安装 anthropic
```

## 前置：配置 API Key

SDK 会自动读取环境变量 `ANTHROPIC_API_KEY`。

```bash
# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-xxxx"

# Windows (PowerShell)
# $env:ANTHROPIC_API_KEY="sk-ant-xxxx"
```

> 国内直连 Anthropic 通常有限制，可能需要代理或兼容 Anthropic 格式的中转服务。
> 如果你用的是第三方中转，可通过设置 `ANTHROPIC_BASE_URL` 指向中转地址。

## 用 DeepSeek Key 跑本模块

DeepSeek 官方提供 Anthropic 兼容端点，所以这些 Claude SDK 示例可以直接使用 DeepSeek：

```text
ANTHROPIC_API_KEY=你的 DeepSeek Key
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
```

脚本里的 `claude-3-5-sonnet-latest` 会被 DeepSeek 自动映射到 `deepseek-v4-flash`，
不需要改模型名。

## 先跑自检脚本

```bash
python claude-sdk/env_check.py
```

若提示缺少 Key，就去配置后重试；若显示网络/连接错误，则是网络或中转问题。

## 关于 API Key 的说明

脚本里都不会硬编码 Key，统一走环境变量，避免把密钥提交到 git。
