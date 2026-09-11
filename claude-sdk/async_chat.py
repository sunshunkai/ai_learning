# -*- coding: utf-8 -*-
"""
Claude SDK —— 异步调用与异步流式输出

运行: python claude-sdk/async_chat.py
"""

import asyncio

from common import DEFAULT_MODEL, get_async_client, text_from_message


async def main():
    client = get_async_client()

    print("=== AsyncAnthropic：异步非流式调用 ===")
    response = await client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=512,
        messages=[
            {"role": "user", "content": "用一句话解释异步编程有什么好处。"}
        ],
    )
    print(text_from_message(response), "\n")

    print("=== 异步流式输出 ===")
    print("回复: ", end="", flush=True)
    async with client.messages.stream(
        model=DEFAULT_MODEL,
        max_tokens=512,
        messages=[
            {"role": "user", "content": "用 3 句话介绍 Python 的 asyncio。"}
        ],
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
