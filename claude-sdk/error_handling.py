# -*- coding: utf-8 -*-
"""
Claude SDK —— 超时、重试和异常处理

运行: python claude-sdk/error_handling.py
"""

import anthropic

from common import DEFAULT_MODEL, get_client, text_from_message


def main():
    # max_retries 由 SDK 自动处理部分临时错误；timeout 控制单次请求超时
    client = get_client(timeout=30.0, max_retries=2)

    try:
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=256,
            messages=[
                {"role": "user", "content": "用一句话说明异常处理的重要性。"}
            ],
        )
        print(text_from_message(response))

    except anthropic.AuthenticationError as exc:
        print("[认证失败] 请检查 ANTHROPIC_API_KEY")
        print(exc)

    except anthropic.RateLimitError as exc:
        print("[触发限流] 可以退避后重试，或检查账户额度")
        print(exc)

    except anthropic.APIStatusError as exc:
        print(f"[API 状态错误] status={exc.status_code}")
        print(exc.response.text[:500])

    except anthropic.APIConnectionError as exc:
        print("[连接失败] 检查网络、代理或 ANTHROPIC_BASE_URL")
        print(exc)

    except anthropic.APIError as exc:
        print("[其他 Claude SDK 错误]")
        print(exc)


if __name__ == "__main__":
    main()
