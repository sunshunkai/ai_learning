# LangChain 学习模块

本目录用来学习 LangChain（Python 版）—— 一个帮你编排 LLM 应用的框架。

## 重要提示：与 claude-sdk 的区别

- `claude-sdk/`：直接用某个厂商的 SDK 调一个模型，简单直接。
- `langchain/`：是一个"抽象层/框架"，可以对接很多厂商模型（OpenAI、Claude、
  本地模型等），并提供提示模板、记忆、工具编排、RAG、Agent 等能力。
  适合做复杂点的应用，但封装多、上手略重。

> 建议先学完 claude-sdk/ 有了"裸调模型"的体感，再对比 LangChain 的封装
> 会更容易理解它解决了什么问题。

## 安装与配置

```bash
pip install -r langchain/requirements.txt
```

LangChain 本身不绑定某个厂商，对接 Claude 需要装 langchain-anthropic；
若用 OpenAI 则装 langchain-openai。本 module 示例以可灵活切换的方式演示。
同样需要配置对应的 API Key（如 ANTHROPIC_API_KEY 或 OPENAI_API_KEY）。

## 版本提醒

网上很多 LangChain 教程是 0.x 时代的（用 Chain/LLMChain 等旧 API），
而当前是 1.x。请以 https://python.langchain.ac.cn 的文档版本为准，
本示例尽量贴近 1.x 新写法。

## 文件说明

- `quickstart.py`：接入一个聊天模型并做最简单问答。
- （后续可按需补充）提示模板 / 记忆 / 工具 / Agent / RAG 等示例。
