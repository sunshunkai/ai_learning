# -*- coding: utf-8 -*-
"""
Claude SDK —— 带命令的交互式流式对话

支持命令：
  /exit    退出
  /clear   清空上下文
  /history 查看当前消息历史

运行: python claude-sdk/interactive_chat.py
"""

from common import DEFAULT_MODEL, get_client, text_from_message


SYSTEM_PROMPT = "你是一个简洁、友好的中文 AI 助手。"


def main():
    client = get_client()
    messages = []

    print("输入 /exit 退出，/clear 清空历史，/history 查看历史。")

    while True:
        user_input = input("\n你: ").strip()
        if not user_input:
            continue
        if user_input == "/exit":
            print("再见。")
            break
        if user_input == "/clear":
            messages.clear()
            print("历史已清空。")
            continue
        if user_input == "/history":
            if not messages:
                print("当前没有历史消息。")
            for message in messages:
                print(f"[{message['role']}] {message['content']}")
            continue

        messages.append({"role": "user", "content": user_input})

        print("AI: ", end="", flush=True)
        with client.messages.stream(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            final = stream.get_final_message()
        print()

        messages.append(
            {"role": "assistant", "content": text_from_message(final)}
        )


if __name__ == "__main__":
    main()
