# -*- coding: utf-8 -*-
"""
Claude SDK —— Extended Thinking 思考内容

开启 thinking 后，响应可能同时包含 thinking block 和 text block。
注意：部分中转服务会忽略 budget_tokens，或禁止与特定 tool_choice 组合。

运行: python claude-sdk/thinking_chat.py
"""

from common import DEFAULT_MODEL, get_client


def main():
    client = get_client()

    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=2048,
        thinking={"type": "enabled", "budget_tokens": 1024},
        messages=[
            {
                "role": "user",
                "content": "一个农夫有 17 只羊，除了 9 只以外都跑了，还剩几只？",
            }
        ],
    )

    for block in response.content:
        if block.type == "thinking":
            print("=== Thinking ===")
            print(getattr(block, "thinking", ""))
        elif block.type == "text":
            print("\n=== 最终回答 ===")
            print(block.text)

    print(f"\n[stop_reason] {response.stop_reason}")


if __name__ == "__main__":
    main()
