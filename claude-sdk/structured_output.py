# -*- coding: utf-8 -*-
"""
Claude SDK —— 用 tool_choice 强制结构化输出

有时我们不希望模型自由回答，而是要求它必须输出固定结构的 JSON。
可以通过给模型一个“提取工具”，再用 tool_choice 强制选择该工具。

运行: python claude-sdk/structured_output.py
"""

import json

from common import DEFAULT_MODEL, get_client


EXTRACT_PROFILE_TOOL = {
    "name": "extract_profile",
    "description": "从自然语言中提取用户资料",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "city": {"type": "string"},
            "occupation": {"type": "string"},
            "skills": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["name", "city", "occupation", "skills"],
    },
}


def main():
    client = get_client()
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=512,
        tools=[EXTRACT_PROFILE_TOOL],
        # DeepSeek 的 thinking 模式不支持指定具体工具名，
        # 但支持 any；这里只有一个工具，因此效果等价。
        tool_choice={"type": "any"},
        messages=[
            {
                "role": "user",
                "content": (
                    "我叫小明，住在杭州，是一名 Java 开发。"
                    "我会 Java、Spring Boot 和一点 Python。"
                ),
            }
        ],
    )

    for block in response.content:
        if block.type == "tool_use":
            print("提取结果:")
            print(json.dumps(block.input, ensure_ascii=False, indent=2))
            return

    print("模型没有返回 tool_use，原始响应:")
    print(response.content)


if __name__ == "__main__":
    main()
