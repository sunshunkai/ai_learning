# -*- coding: utf-8 -*-
"""
Claude SDK —— 流式输出（打字机效果）

普通调用要等 Claude 全部生成完才一次性返回；
流式调用则把回复内容"逐块"推送给你，适合做聊天界面、长回复体验。

运行: python claude-sdk/stream_chat.py
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

    # 用 client.messages.stream(...) 进入流式上下文
    with client.messages.stream(
        model="claude-3-5-sonnet-latest",
        max_tokens=1024,
        messages=[{"role": "user", "content": "用 5 句话介绍你自己。"}],
    ) as stream:
        # text_stream 逐个吐出文本块，边出边打印（flush=True 保证实时显示）
        print("Claude: ", end="", flush=True)
        for text in stream.text_stream:
            print(text, end="", flush=True)
        print()  # 换行

        # 流结束后，还能拿到完整的元信息（用法、结束原因等）
        final = stream.get_final_message()
        print(f"\n[元信息] model={final.model}, "
              f"in={final.usage.input_tokens}, out={final.usage.output_tokens}")


if __name__ == "__main__":
    main()
