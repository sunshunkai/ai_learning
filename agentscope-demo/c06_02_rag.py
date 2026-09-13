# -*- coding: utf-8 -*-
"""
RAG：解析 -> 分块 -> 嵌入 -> 向量库 -> 智能体检索。

DeepSeek 不提供 embedding API，因此教学示例实现了一个确定性的本地
哈希嵌入。生产环境可替换为 DashScope、OpenAI 等 EmbeddingModel；
同时用 DeepSeekChatModel 作为 RAGMiddleware 的大模型重排器。

运行:
    python agentscope-demo/c06_02_rag.py
"""

import asyncio
import math
import re

from agentscope.agent import Agent
from agentscope.credential import CredentialBase
from agentscope.embedding import EmbeddingModelBase, EmbeddingResponse, EmbeddingUsage
from agentscope.middleware import RAGMiddleware
from agentscope.rag import (
    ApproxTokenChunker,
    KnowledgeBase,
    QdrantStore,
    TextParser,
)
from agentscope.tool import Toolkit

from common import get_model, make_user


class LocalHashEmbedding(EmbeddingModelBase[str]):
    """Character n-gram feature-hash embedding for teaching RAG flow.

    It makes no network calls and is deterministic. It is deliberately
    NOT a production semantic embedding model.
    """

    supports_multimodal = False

    def __init__(self, dimensions: int = 256) -> None:
        super().__init__(
            credential=CredentialBase(name="local-hash"),
            model="local-hash-v1",
            dimensions=dimensions,
            parameters=None,
            context_size=8192,
            batch_size=16,
            max_retries=0,
            retry_delay=0,
        )

    @staticmethod
    def _features(text: str):
        text = re.sub(r"\s+", " ", text.lower())
        words = re.findall(r"[\w\u4e00-\u9fff]+", text)
        yield from words
        for word in words:
            yield from (word[index : index + 2] for index in range(len(word) - 1))
            yield from (word[index : index + 3] for index in range(len(word) - 2))

    async def _call_api(self, inputs, **kwargs) -> EmbeddingResponse:
        del kwargs
        vectors = []
        tokens = 0
        for item in inputs:
            text = item if isinstance(item, str) else str(item)
            vector = [0.0] * self.dimensions
            tokens += max(1, len(text.encode("utf-8")) // 4)
            for feature in self._features(text):
                index = hash(feature) % self.dimensions
                vector[index] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return EmbeddingResponse(
            embeddings=vectors,
            usage=EmbeddingUsage(tokens=tokens, time=0.0),
        )


DOCUMENTS: dict[str, str] = {
    "agentscope.md": """
# AgentScope 2.0

AgentScope 是面向生产的多智能体开发框架，核心是 Agent 推理-行动循环。

Agent 由 model、toolkit、middleware、permission、context 和 state 组合而成。
AgentScope 支持函数工具、内置文件工具、MCP、Skill 和 ToolGroup。

关键特性包括：流式事件、结构化输出、上下文压缩、长期记忆、RAG、
Human-in-the-loop 和 GoalPipeline。
""".strip(),
    "python_async.md": """
# Python 异步

Python 的 asyncio 使用事件循环调度协程，适合 IO 密集型任务。

async 定义协程，await 挂起当前协程并把控制权交给事件循环。
asyncio.gather 可并发运行多个协程；Task 可以被取消。

CPU 密集任务不会因为 async/await 自动加速，应使用线程池或进程池。
""".strip(),
    "deepseek.md": """
# DeepSeek 接入

AgentScope 提供 DeepSeekChatModel 和 DeepSeekCredential。

deepseek-v4-flash 支持 thinking_enable 与 reasoning_effort。
DeepSeek 使用 OpenAI Chat Completions 兼容接口，但不提供 Embedding API。

学习项目中模型默认使用 deepseek-v4-flash，Base URL 是
https://api.deepseek.com。
""".strip(),
}


async def build_knowledge(knowledge: KnowledgeBase) -> dict[str, str]:
    parser = TextParser()
    chunker = ApproxTokenChunker(
        parameters=ApproxTokenChunker.Parameters(
            chunk_size=180,
            overlap=30,
        )
    )
    document_ids = {}
    for filename, content in DOCUMENTS.items():
        sections = await parser.parse(
            file=content.encode("utf-8"),
            filename=filename,
        )
        chunks = await chunker.chunk(sections)
        document_id = await knowledge.insert_document(
            chunks,
            document_metadata={"filename": filename},
        )
        document_ids[filename] = document_id
        print(f"  已索引 {filename}: {len(chunks)} chunks")
    return document_ids


async def direct_search(knowledge: KnowledgeBase) -> None:
    print("\n=== 1. 直接向量检索 ===")
    results = await knowledge.search(
        ["Python 异步应该怎么学？"],
        top_k=3,
    )
    for rank, result in enumerate(results, 1):
        text = result.chunk.content.text.replace("\n", " ")[:100]
        print(
            f"[{rank}] score={result.score:.3f}, "
            f"source={result.chunk.source}\n     {text}"
        )


async def agentic_rag(knowledge: KnowledgeBase) -> None:
    print("\n=== 2. Agentic RAG + DeepSeek 重排 ===")
    middleware = RAGMiddleware(
        knowledge_bases=[knowledge],
        rerank_model=get_model(stream=False, thinking_enable=False),
        parameters=RAGMiddleware.Parameters(
            mode="agentic",
            top_k=3,
            rerank_candidate_k=6,
        ),
    )
    agent = Agent(
        name="KnowledgeAssistant",
        system_prompt=(
            "回答用户问题时优先检索知识库。只依据检索到的材料回答，"
            "如果检索不到就明确说明。"
        ),
        model=get_model(stream=False, thinking_enable=True),
        toolkit=Toolkit(tools=await middleware.list_tools()),
        middlewares=[middleware],
    )

    reply = await agent.reply(
        make_user(
            "AgentScope 的学习项目里应该如何接入 DeepSeek？"
            "同时简要说明 Python 异步是否适合 CPU 密集任务。"
        )
    )
    print(reply.get_text_content())


async def main() -> None:
    embedding = LocalHashEmbedding(dimensions=256)
    store = QdrantStore(location=":memory:")

    async with store:
        knowledge = KnowledgeBase(
            name="ai-learning-kb",
            description="AgentScope、DeepSeek 与 Python 异步学习资料。",
            embedding_model=embedding,
            vector_store=store,
            collection="ai-learning-demo",
        )
        print("=== 构建知识库 ===")
        await build_knowledge(knowledge)
        await direct_search(knowledge)
        await agentic_rag(knowledge)


if __name__ == "__main__":
    asyncio.run(main())
