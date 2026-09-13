# -*- coding: utf-8 -*-
"""
Agent 事件流：逐个观察并重建最终消息。

reply 只是消费 reply_stream 后返回最终 Msg；前端、调试器和 Console 都可
以直接消费这些增量事件。

运行:
    python agentscope-demo/c03_04_streaming_events.py
"""

import asyncio

from agentscope.agent import Agent
from agentscope.event import (
    ModelCallEndEvent,
    ReplyEndEvent,
    ReplyStartEvent,
    TextBlockDeltaEvent,
    TextBlockEndEvent,
    TextBlockStartEvent,
    ThinkingBlockDeltaEvent,
)
from agentscope.message import AssistantMsg
from agentscope.tool import Toolkit

from common import get_model, make_user


async def main() -> None:
    agent = Agent(
        name="Streamer",
        system_prompt="你是演示事件流的助手，回答控制在三句话以内。",
        model=get_model(stream=True, thinking_enable=True),
        toolkit=Toolkit(),
    )

    final_msg: AssistantMsg | None = None
    thinking_chars = 0
    print("=== 原始事件流 ===")

    async for event in agent.reply_stream(
        make_user("用三句话介绍消息与事件的区别。")
    ):
        if isinstance(event, ReplyStartEvent):
            final_msg = AssistantMsg(
                name=event.name,
                content=[],
                id=event.reply_id,
            )
            print(f"\n[ReplyStart] reply_id={event.reply_id}")

        elif isinstance(event, ThinkingBlockDeltaEvent):
            thinking_chars += len(event.delta)

        elif isinstance(event, TextBlockStartEvent):
            print("\n[text] ", end="", flush=True)

        elif isinstance(event, TextBlockDeltaEvent):
            print(event.delta, end="", flush=True)

        elif isinstance(event, TextBlockEndEvent):
            print()

        elif isinstance(event, ModelCallEndEvent):
            print(
                f"[ModelCallEnd] in={event.input_tokens}, "
                f"out={event.output_tokens}"
            )

        elif isinstance(event, ReplyEndEvent):
            print(f"[ReplyEnd] finished_reason={event.finished_reason}")

        if final_msg is not None:
            final_msg.append_event(event)

    print("\n=== 事件重建出的最终消息 ===")
    assert final_msg is not None
    print(final_msg.get_text_content())
    print(f"thinking_chars={thinking_chars}")
    print(f"finished_reason={final_msg.finished_reason}")
    print(f"usage={final_msg.usage}")


if __name__ == "__main__":
    asyncio.run(main())
