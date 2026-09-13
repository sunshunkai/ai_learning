# -*- coding: utf-8 -*-
"""公共配置：环境变量、DeepSeek 模型、示例文档。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.load_env import load as load_env  # noqa: E402

load_env()


def build_deepseek_model(model: str | None = None, temperature: float = 0):
    """创建 DeepSeek Chat 模型。"""
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "缺少 DEEPSEEK_API_KEY，请先复制项目根目录的 .env.example 为 .env 并填写。"
        )

    model_name = model or os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"
    base_url = os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )


def load_documents() -> list[Document]:
    """返回一组用于演示的知识库文档。"""
    return [
        Document(
            page_content=(
                "LangChain 是一个用于构建大语言模型应用的框架。"
                "它提供统一的方式接入聊天模型、Prompt、输出解析、记忆、工具和 Agent。"
                "LCEL 是 LangChain 的声明式组合语法，可以使用竖线把组件连接成链。"
            ),
            metadata={"source": "langchain.md", "topic": "framework"},
        ),
        Document(
            page_content=(
                "RAG 指检索增强生成。"
                "先从文档库检索相关内容，再把检索结果作为上下文交给模型回答。"
                "RAG 可以降低模型编造信息的概率。"
            ),
            metadata={"source": "rag.md", "topic": "rag"},
        ),
        Document(
            page_content=(
                "DeepSeek 提供 OpenAI 兼容的 Chat Completions 接口。"
                "DeepSeek 不提供 Embedding API，所以 RAG 项目需要单独选择 embedding 模型。"
            ),
            metadata={"source": "deepseek.md", "topic": "deepseek"},
        ),
    ]
