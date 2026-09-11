# -*- coding: utf-8 -*-
"""
可靠性与降级：with_retry、with_fallbacks

运行: python langchain-demo/11_retry_fallback.py
"""

from langchain_core.runnables import RunnableLambda

from common import build_model


class TemporaryError(Exception):
    pass


def main():
    print("=== Runnable.with_retry：同一个 Runnable 自动重试 ===")
    attempts = {"count": 0}

    def flaky(_):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TemporaryError(f"第 {attempts['count']} 次临时失败")
        return "第 3 次成功"

    retried = RunnableLambda(flaky).with_retry(
        retry_if_exception_type=(TemporaryError,),
        stop_after_attempt=3,
        wait_exponential_jitter=False,
    )
    print(retried.invoke(None))

    print("\n=== with_fallbacks：主链路失败后切换备用链路 ===")
    primary = RunnableLambda(
        lambda _: (_ for _ in ()).throw(RuntimeError("主模型不可用"))
    )
    fallback = RunnableLambda(lambda value: f"备用链路收到: {value}")
    chain = primary.with_fallbacks([fallback])
    print(chain.invoke("测试输入"))

    print("\n=== 给聊天模型增加重试 ===")
    robust_model = build_model(temperature=0).with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )
    result = robust_model.invoke("用一句话说明重试机制。")
    print(result.content)


if __name__ == "__main__":
    main()
