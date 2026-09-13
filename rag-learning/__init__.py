# -*- coding: utf-8 -*-
"""
RAG 学习模块。

这个模块把 RAG 拆成多个独立 demo，重点说明每个组件的作用：
01 文档加载 -> 02 文本分块 -> 03-05 Embedding ->
06-08 VectorStore -> 09-12 RAG Chain 与 Provider 切换。

切换 embedding 或 vectorstore 时，业务代码只依赖 retriever，
不需要修改 RAG 主流程。

详细说明见 KNOWLEDGE_GUIDE.md。
"""
