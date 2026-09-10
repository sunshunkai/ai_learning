# -*- coding: utf-8 -*-
"""
Claude SDK 入门 —— 一问一答
运行: python claude-sdk/basic_chat.py

先决条件：
  1. pip install -r claude-sdk/requirements.txt
  2. 配置好环境变量 ANTHROPIC_API_KEY
可先跑 python claude-sdk/env_check.py 自检。
"""

import os

# 兼容"项目根 .env 文件"的加载（可选，无 .env 时静默跳过）
try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from tools.load_env import load  # noqa
    load()
except Exception:
    pass

from anthropic import Anthropic


def main():
    # 1. 创建客户端。SDK 会自动读取 ANTHROPIC_API_KEY 环境变量。
    #    Anthropic() 等价于 Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    client = Anthropic()

    # 2. 调用 messages.create 发消息
    #    对比你熟悉的 HTTP 思路：SDK 把"拼请求 + 鉴权 + 解析响应"都封装好了。
    response = client.messages.create(
        model="claude-3-5-sonnet-latest",   # 模型名
        max_tokens=1024,                     # 回复最大 token 数
        messages=[                           # 对话历史（首轮只需一条 user）
            {"role": "user", "content": "用一句话解释什么是大语言模型(LLM)？"}
        ],
    )

    # 3. 解析响应
    #    response.content 是一个 list，里面每个元素有 type 字段
    #    文本回复的 type 是 "text"，内容在 .text
    print("=== Claude 回复 ===")
    for block in response.content:
        if block.type == "text":
            print(block.text)

    # 附带打印一些元信息，帮助你理解响应结构
    print("\n=== 元信息 ===")
    print(f"stop_reason : {response.stop_reason}")   # 为什么停止(如 end_turn)
    print(f"model       : {response.model}")
    print(f"input_tokens: {response.usage.input_tokens}")
    print(f"output_token: {response.usage.output_tokens}")


if __name__ == "__main__":
    main()
