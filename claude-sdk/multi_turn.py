# -*- coding: utf-8 -*-
"""
Claude SDK —— 多轮对话（手动维护上下文）

Claude 本身无状态，每次调用都要把"完整对话历史"传进去。
所以多轮对话 = 自己用 list 累积 messages，每次追加新的一来一回。

运行: python claude-sdk/multi_turn.py
"""

import os
import sys

# 兼容"项目根 .env 文件"的加载（可选，无 .env 时静默跳过）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from tools.load_env import load
    load()
except Exception:
    pass

from anthropic import Anthropic


def main():
    client = Anthropic()

    # 初始化对话历史（role 只有两种：user / assistant）
    messages = [
        {"role": "user", "content": "我叫Nathan，是一名 Java 开发，正在学 Python 和 AI。"},
    ]

    # 用来跟 Claude 对话的函数：传入用户输入，返回 Claude 回复
    def chat(user_input):
        # 把用户的新消息加进历史
        messages.append({"role": "user", "content": user_input})

        # 带着完整历史请求
        resp = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
            messages=messages,
        )

        # 取出文本回复
        reply = "".join(b.text for b in resp.content if b.type == "text")

        # 把 Claude 的回复也加进历史（关键！否则下次它不记得）
        messages.append({"role": "assistant", "content": reply})
        return reply

    # 连续提问，验证它是否"记住"了前面的话
    print("问1:", end=" ")  # noqa
    print(chat("你还记得我刚才介绍过自己吗？我的职业是什么？"))

    print("\n问2:", end=" ")
    print(chat("既然我在学 Python，给我 3 条最适合转行 AI 的学习建议，每条一句话。"))


if __name__ == "__main__":
    main()
