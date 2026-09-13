# -*- coding: utf-8 -*-
"""
LangChain 流式事件：astream_events 观察链的执行过程

运行: python langchain-demo/15_stream_events.py
"""

import asyncio

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common import build_model


async def main():
    prompt = ChatPromptTemplate.from_messages(
        [("human", "用三句话解释：{topic}")]
    )
    chain = prompt | build_model(temperature=0.2) | StrOutputParser()

    print("=== 事件与文本流 ===")
    # astream_events 不只返回文本，还报告链、模型等组件的生命周期事件。
    # version="v2" 固定事件结构，便于按稳定的 event 名称做观测和埋点。
    async for event in chain.astream_events(
        {"topic": "LangChain 的 Runnable"},
        version="v2",
    ):
        event_name = event["event"]

        # 每个事件都带有 name、data、run_id 等字段，可按组件或事件类型过滤。
        if event_name == "on_chat_model_start":
            print("\n[model_start] 模型开始生成")

        elif event_name == "on_chat_model_stream":
            # 模型 token 增量位于 data["chunk"]。
            chunk = event["data"]["chunk"]
            if chunk.content:
                print(chunk.content, end="", flush=True)

        elif event_name == "on_chain_end" and event.get("name") == "RunnableSequence":
            # LCEL 用 | 组装的链通常表现为 RunnableSequence。
            print("\n\n[chain_end] 链执行完成")


if __name__ == "__main__":
    asyncio.run(main())
