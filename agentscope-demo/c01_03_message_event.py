# -*- coding: utf-8 -*-
"""
AgentScope 消息与事件基础。

Message 是完整会话轮次，Event 是流式增量。本示例演示：
  1. 三种快捷消息与内容块
  2. 手工构造多模态 DataBlock
  3. 用事件重建出一条 AssistantMsg
  4. 观察真实模型响应的 block 类型

运行:
    python agentscope-demo/c01_03_message_event.py
"""

import asyncio
import base64
import json

from agentscope.event import (
    ReplyEndEvent,
    ReplyStartEvent,
    TextBlockDeltaEvent,
    TextBlockEndEvent,
    TextBlockStartEvent,
)
from agentscope.message import (
    AssistantMsg,
    Base64Source,
    DataBlock,
    Msg,
    SystemMsg,
    TextBlock,
    ToolCallBlock,
    ToolCallState,
    ToolResultBlock,
    ToolResultState,
    UserMsg,
)
from agentscope.types import ReplyFinishedReason

from common import get_model, make_user, print_usage, stream_and_collect


def show_message_construction() -> None:
    print("=== 1. Message 工厂与内容块 ===")

    system = SystemMsg(name="system", content="你是中文技术助教。")
    user = UserMsg(
        name="nathan",
        content=[
            TextBlock(text="请分析这张图片。"),
            DataBlock(
                source=Base64Source(
                    data=base64.b64encode(b"fake-image-bytes").decode("ascii"),
                    media_type="image/png",
                ),
                name="diagram.png",
            ),
        ],
    )
    assistant = AssistantMsg(
        name="Friday",
        content=[
            TextBlock(text="我先调用一个工具。"),
            ToolCallBlock(
                id="call-weather-1",
                name="get_weather",
                input=json.dumps({"city": "杭州"}, ensure_ascii=False),
                state=ToolCallState.FINISHED,
            ),
            ToolResultBlock(
                id="call-weather-1",
                name="get_weather",
                output="杭州：晴，26°C",
                state=ToolResultState.SUCCESS,
            ),
        ],
    )

    for msg in (system, user, assistant):
        print(
            f"[{msg.role:9}] name={msg.name:8}, "
            f"text={msg.get_text_content()!r}"
        )

    print("\n按类型取块:")
    print(f"  text blocks: {len(user.get_content_blocks('text'))}")
    print(f"  data blocks: {len(user.get_content_blocks('data'))}")
    print(f"  tool_calls: {assistant.has_content_blocks('tool_call')}")
    print(f"  tool_results: {assistant.has_content_blocks('tool_result')}")


def show_event_reconstruction() -> None:
    print("\n=== 2. Event -> Message 重建 ===")

    reply_id = "demo-reply-001"
    session_id = "demo-session-001"
    block_id = "demo-text-001"

    msg = AssistantMsg(name="Friday", content=[], id=reply_id)
    events = [
        ReplyStartEvent(
            reply_id=reply_id,
            session_id=session_id,
            name="Friday",
        ),
        TextBlockStartEvent(reply_id=reply_id, block_id=block_id),
        TextBlockDeltaEvent(
            reply_id=reply_id,
            block_id=block_id,
            delta="流式事件",
        ),
        TextBlockDeltaEvent(
            reply_id=reply_id,
            block_id=block_id,
            delta="可以无损重建消息。",
        ),
        TextBlockEndEvent(reply_id=reply_id, block_id=block_id),
        ReplyEndEvent(
            reply_id=reply_id,
            session_id=session_id,
            finished_reason=ReplyFinishedReason.COMPLETED,
        ),
    ]

    for event in events:
        msg.append_event(event)

    print(f"msg.id={msg.id}")
    print(f"msg.text={msg.get_text_content()!r}")
    print(f"msg.finished_reason={msg.finished_reason}")
    print(f"msg.finished_at={msg.finished_at}")


async def show_real_model_response() -> None:
    print("\n=== 3. 真实 DeepSeek 响应内容 ===")
    model = get_model(stream=False, thinking_enable=True)
    response = await stream_and_collect(
        model,
        [make_user("9 * 8 + 12 等于多少？")],
    )
    for block in response.content:
        kind = getattr(block, "type", type(block).__name__)
        preview = str(getattr(block, "text", getattr(block, "thinking", "")))[:60]
        print(f"  [{kind}] {preview}")
    print_usage(response)


async def main() -> None:
    show_message_construction()
    show_event_reconstruction()
    await show_real_model_response()


if __name__ == "__main__":
    asyncio.run(main())
