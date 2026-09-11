# -*- coding: utf-8 -*-
"""
Claude SDK —— Prompt Caching 提示缓存

在官方 Anthropic API 中，cache_control 可缓存稳定前缀，减少重复输入成本。
DeepSeek 兼容端点目前会忽略 cache_control，但请求结构可以正常演示。

运行: python claude-sdk/prompt_caching.py
"""

from common import DEFAULT_MODEL, get_client, text_from_message


LONG_SYSTEM_PROMPT = """
你是一名企业知识库助手。回答时只使用给定的规则：
1. 先指出问题所属模块。
2. 再给出不超过三句话的答案。
3. 如果信息不足，明确说资料不足。
4. 不编造内部流程、账号或价格。
5. 对外输出保持专业、简洁。
""".strip()


def main():
    client = get_client()
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": LONG_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": "员工忘记密码后应该走什么流程？",
            }
        ],
    )

    print(text_from_message(response))

    usage = response.usage
    print("\n=== Cache 使用情况 ===")
    for name in (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ):
        print(f"{name}: {getattr(usage, name, None)}")


if __name__ == "__main__":
    main()
