# -*- coding: utf-8 -*-
"""
RAG 最小示例：文本切分 -> 向量化 -> 检索 -> 结合上下文回答

DeepSeek 不提供 embedding API，所以这里用本地确定性 HashEmbeddings
演示完整链路；生产环境可换成 OpenAI/HuggingFace/BGE 等 embedding。

运行: python langchain-demo/09_rag.py
"""

import hashlib
import math

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common import build_model


class HashEmbeddings(Embeddings):
    """基于词哈希的本地向量，仅供演示 RAG 流程。"""

    def __init__(self, dimensions: int = 64):
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def main():
    content = (
        "LangChain 是一个用于构建大语言模型应用的框架。"
        "它提供统一的方式接入聊天模型、Prompt 模板、输出解析、记忆、工具、Agent。"
        "LCEL 是 LangChain 的声明式组合语法，可以使用竖线把组件连接成链。"
        "RAG 指检索增强生成：先从文档库检索相关内容，再让模型基于上下文回答。"
        "DeepSeek 提供 OpenAI 兼容接口，LangChain 可以通过 ChatOpenAI 接入。"
    )
    docs = [Document(page_content=content, metadata={"source": "demo"})]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=10,
    )
    chunks = splitter.split_documents(docs)
    print(f"切分后得到 {len(chunks)} 个 chunk")

    embedding = HashEmbeddings(dimensions=64)
    vectorstore = InMemoryVectorStore.from_documents(chunks, embedding)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "只根据下面的资料回答，资料中没有的信息不要编造：\n\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    def format_docs(retrieved_docs: list[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | build_model(temperature=0)
        | StrOutputParser()
    )

    question = "RAG 是什么？LangChain 如何接入 DeepSeek？"
    print(f"\n问题: {question}\n")
    print(rag_chain.invoke(question))


if __name__ == "__main__":
    main()
