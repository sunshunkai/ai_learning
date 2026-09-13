# -*- coding: utf-8 -*-
"""
RAG 最小示例：文本切分 -> 向量化 -> 检索 -> 结合上下文回答

DeepSeek 不提供 embedding API，所以这里用本地确定性 HashEmbeddings
演示完整链路；生产环境可换成 OpenAI/HuggingFace/BGE 等 embedding。

完整数据流：
1. 原始 Document 按 chunk_size 切成较小的文本块。
2. embedding 模型把每个 chunk 转成向量并写入向量库。
3. retriever 根据用户问题找出最相近的 k 个 chunk。
4. Prompt 将检索结果放进 {context}，把原问题放进 {question}。
5. 聊天模型根据资料回答，Parser 最后把 AIMessage 转成字符串。

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
    """
    基于词哈希的本地向量，仅供演示 RAG 流程。

    它只能表达“相同 token 更容易相似”，不理解同义词和语义；
    并且 split() 只按空白分词，中文最好改用 jieba 等分词器；
    真正的 RAG 应使用训练好的 embedding 模型。这个类的价值是让示例
    在没有外部 embedding 服务时，也能完整跑通向量化和检索链路。
    """

    def __init__(self, dimensions: int = 64):
        # 向量维度也是哈希桶数量；维度越大，哈希冲突通常越少。
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 文档入库时调用；返回列表的元素顺序必须和 texts 一一对应。
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        # 查询和文档必须使用同一套向量算法，否则相似度计算没有意义。
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        # 先创建固定长度零向量，再把每个词累加到哈希后的桶中。
        vector = [0.0] * self.dimensions
        # split() 是刻意保持极简的实现；生产环境应先做适合语言的 tokenization。
        for token in text.lower().split():
            # 同一 token 始终映射到同一维度，形成确定性的稀疏词袋向量。
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0
        # L2 归一化后比较余弦相似度时，不受文本长短造成的词频尺度影响。
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def main():
    # 这里把一小段资料当作“知识库”；Document 是 LangChain 的文档对象。
    # metadata 可保存来源、页码等信息，检索后可用于展示引用。
    content = (
        "LangChain 是一个用于构建大语言模型应用的框架。"
        "它提供统一的方式接入聊天模型、Prompt 模板、输出解析、记忆、工具、Agent。"
        "LCEL 是 LangChain 的声明式组合语法，可以使用竖线把组件连接成链。"
        "RAG 指检索增强生成：先从文档库检索相关内容，再让模型基于上下文回答。"
        "DeepSeek 提供 OpenAI 兼容接口，LangChain 可以通过 ChatOpenAI 接入。"
    )
    docs = [Document(page_content=content, metadata={"source": "demo"})]

    # RecursiveCharacterTextSplitter 优先按段落、换行、句子等自然边界切分。
    # overlap 让相邻 chunk 保留少量重复内容，降低答案正好被切断的风险。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=10,
    )
    chunks = splitter.split_documents(docs)
    print(f"切分后得到 {len(chunks)} 个 chunk")

    # from_documents 会依次调用 embed_documents，并用向量和原文构建向量库。
    embedding = HashEmbeddings(dimensions=64)
    vectorstore = InMemoryVectorStore.from_documents(chunks, embedding)
    # Retriever 是“向量库查询”的标准接口；k=2 表示返回最相似的 2 个 chunk。
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # RAG Prompt 把检索结果和用户问题分成两个独立槽位。
    # system 消息明确约束“只依据资料回答”，降低模型编造答案的概率。
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
        # Retriever 返回 Document 列表；这里把正文拼成一段可供模型阅读的文本。
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    # 方括号中的字典负责把同一个问题分流到两个并行输入：
    # - context 分支先检索，再把 Document 列表格式化成字符串；
    # - question 分支用 RunnablePassthrough 原样保留用户问题。
    # 两个结果合并后依次经过 prompt -> 模型 -> 字符串解析器。
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
    # invoke 结束时会得到纯字符串；中间步骤都可以用相同 Runnable 接口替换。
    print(rag_chain.invoke(question))


if __name__ == "__main__":
    main()
