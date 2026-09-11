# -*- coding: utf-8 -*-
"""
Claude SDK —— 底层流事件

stream_chat.py 演示的是 stream.text_stream，适合直接展示文本。
本案例遍历原始事件，观察 message_start、content_block_delta 等结构。

运行: python claude-sdk/stream_events.py
"""

from common import DEFAULT_MODEL, get_client


def main():
    client = get_client()

    print("=== 原始流事件 ===")
    with client.messages.stream(
        model=DEFAULT_MODEL,
        max_tokens=512,
        messages=[
            {"role": "user", "content": "用两句话说明什么是流式输出。"}
        ],
    ) as stream:
        for event in stream:
            print(f"\n[event] {event.type}")

            if event.type == "content_block_start":
                print(f"  block type = {event.content_block.type}")

            elif event.type == "content_block_delta":
                delta = event.delta
                print(f"  delta type = {delta.type}")
                if delta.type == "text_delta":
                    print(f"  text       = {delta.text}")
                elif delta.type == "thinking_delta":
                    print(f"  thinking   = {delta.thinking}")
                elif delta.type == "input_json_delta":
                    print(f"  partial_json = {delta.partial_json}")

        final = stream.get_final_message()
        print(f"\n[最终状态] stop_reason={final.stop_reason}")
        print(f"[Token] in={final.usage.input_tokens}, out={final.usage.output_tokens}")


if __name__ == "__main__":
    main()
