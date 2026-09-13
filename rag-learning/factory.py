# -*- coding: utf-8 -*-
"""RAG 组件工厂：让 embedding 和 vectorstore 可以按 provider 切换。"""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common import load_documents


class HashEmbeddings(Embeddings):
    """本地确定性 hash embedding，仅用于在没有外部服务时跑通 RAG 链路。"""

    def __init__(self, dimensions: int = 64):
        self.dimensions = dimensions

    @staticmethod
    def _features(text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text.lower())
        words = re.findall(r"[a-z0-9\u4e00-\u9fff]+", text)
        features: list[str] = []
        for word in words:
            features.append(word)
            for index in range(len(word) - 1):
                features.append(word[index : index + 2])
        return features

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for feature in self._features(text):
            digest = hashlib.md5(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@dataclass
class RAGConfig:
    embedding_provider: str = "hash"
    vectorstore_provider: str = "memory"
    top_k: int = 3
    chunk_size: int = 120
    chunk_overlap: int = 20
    embedding_kwargs: dict[str, Any] = field(default_factory=dict)
    vectorstore_kwargs: dict[str, Any] = field(default_factory=dict)


def build_embedding(provider: str, **kwargs: Any) -> Embeddings:
    if provider == "hash":
        return HashEmbeddings(dimensions=int(kwargs.get("dimensions", 64)))

    if provider == "bge-local":
        try:
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "bge-local 需要安装 langchain-community 和 sentence-transformers"
            ) from exc
        model_name = kwargs.get("model_name") or os.environ.get(
            "BGE_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
        )
        return HuggingFaceBgeEmbeddings(model_name=model_name)

    if provider == "dashscope":
        try:
            from langchain_community.embeddings import DashScopeEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "dashscope 需要安装 langchain-community 和 dashscope"
            ) from exc
        api_key = kwargs.get("dashscope_api_key") or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY")
        model = kwargs.get("model") or os.environ.get(
            "DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4"
        )
        return DashScopeEmbeddings(model=model, dashscope_api_key=api_key)

    raise ValueError(f"不支持的 embedding provider: {provider}")


def build_vectorstore(
    provider: str,
    chunks: list[Document],
    embedding: Embeddings,
    **kwargs: Any,
):
    if provider == "memory":
        return InMemoryVectorStore.from_documents(chunks, embedding)

    if provider == "chroma":
        try:
            from langchain_community.vectorstores import Chroma
        except ImportError as exc:
            raise RuntimeError("chroma 需要安装 langchain-community 和 chromadb") from exc
        persist_directory = kwargs.pop(
            "persist_directory",
            os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db"),
        )
        return Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=persist_directory,
            **kwargs,
        )

    if provider == "faiss":
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError as exc:
            raise RuntimeError("faiss 需要安装 langchain-community 和 faiss-cpu") from exc
        return FAISS.from_documents(chunks, embedding)

    raise ValueError(f"不支持的 vectorstore provider: {provider}")


def build_retriever(vectorstore, top_k: int):
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def format_docs(docs: list[Document]) -> str:
    """把检索结果格式化成带来源的上下文。"""
    return "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


def build_rag_chain(model, retriever, prompt=None):
    """构建基础 RAG Chain。这里只依赖 retriever，不关心底层 vectorstore。"""
    prompt = prompt or ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "只根据下面的资料回答。资料中没有的信息不要编造：\n\n{context}",
            ),
            ("human", "{question}"),
        ]
    )
    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )


class RAGFactory:
    """把文档、配置、组件创建和 Chain 创建放在一起。"""

    def __init__(self, config: RAGConfig | None = None, documents=None):
        self.config = config or RAGConfig()
        self.documents = documents or load_documents()

    def split_documents(self) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "],
        )
        return splitter.split_documents(self.documents)

    def create_components(self):
        chunks = self.split_documents()
        embedding = build_embedding(
            self.config.embedding_provider,
            **self.config.embedding_kwargs,
        )
        vectorstore = build_vectorstore(
            self.config.vectorstore_provider,
            chunks,
            embedding,
            **self.config.vectorstore_kwargs,
        )
        retriever = build_retriever(vectorstore, self.config.top_k)
        return chunks, embedding, vectorstore, retriever

    def create_chain(self, model, retriever=None, prompt=None):
        if retriever is None:
            _, _, _, retriever = self.create_components()
        return build_rag_chain(model, retriever, prompt)
