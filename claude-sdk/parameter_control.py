# -*- coding: utf-8 -*-
"""
Claude SDK —— System、temperature、top_p、stop_sequences、metadata

运行: python claude-sdk/parameter_control.py
"""

from common import DEFAULT_MODEL, get_client, text_from_message


def main():
    client = get_client()

    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=512,

        # system 可传字符串，也可传文本块列表
        system=(
            "你是一名严谨的技术讲师。"
            "回答必须控制在三句话以内，不要使用 Markdown。"
        ),

        # 模型一旦生成这个字符串就提前停止
        stop_sequences=["###"],

        # 可用于业务侧标记用户；DeepSeek 兼容端点支持 user_id
        metadata={"user_id": "demo-user-001"},

        # 1.x SDK 不再把 temperature/top_p 放在顶层参数，
        # 第三方兼容端点可通过 extra_body 透传。
        extra_body={"temperature": 0.3, "top_p": 0.9},

        messages=[
            {"role": "user", "content": "解释一下 Python 生成器。"}
        ],
    )

    print("=== 回复 ===")
    print(text_from_message(response))
    print("\n=== 参数与元信息 ===")
    print(f"temperature      : 0.3")
    print(f"top_p            : 0.9")
    print(f"stop_sequences   : ['###']")
    print(f"stop_reason      : {response.stop_reason}")
    print(f"stop_sequence    : {response.stop_sequence}")
    print(f"input_tokens     : {response.usage.input_tokens}")
    print(f"output_tokens    : {response.usage.output_tokens}")


if __name__ == "__main__":
    main()
