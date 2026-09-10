# -*- coding: utf-8 -*-
"""
async/await 练习 —— 学 AI SDK 前的"最小必要"理解
运行: python basics/async_basics.py

为什么要学？
- Claude SDK、LangChain 的流式输出、并发请求几乎都用 async。
- 你不需要精通，但至少要能"看懂 async 函数"并能运行它。
"""

import asyncio


# async def 定义一个"协程函数"（类似一个异步方法）
# 它不会立即执行，调用后返回一个"协程对象"，要放进事件循环里跑
async def fake_api_call(name, delay):
    """
    模拟一个耗时的 API 调用。
    await asyncio.sleep(delay) 表示"等 delay 秒，期间不阻塞其他任务"。
    """
    print(f"  开始请求 {name}...")
    await asyncio.sleep(delay)            # 模拟网络等待
    print(f"  {name} 完成")
    return f"{name} 的结果"


# async def main() 是异步程序的入口
async def main():
    # 方式一：普通地一个个 await（串行，总耗时 = 各 delay 之和）
    print("===== 串行执行 =====")
    r1 = await fake_api_call("请求A", 1)
    r2 = await fake_api_call("请求B", 1)
    print(r1, "|", r2)

    # 方式二：用 asyncio.gather 并发跑（总耗时 ≈ 最慢的那个）
    print("\n===== 并发执行 (gather) =====")
    results = await asyncio.gather(
        fake_api_call("并发请求1", 2),
        fake_api_call("并发请求2", 2),
        fake_api_call("并发请求3", 2),
    )
    print("并发结果:", results)

    # 方式三：流式/迭代（Claude SDK 的 stream 也是这种思想）
    # 注意：普通异步函数不能直接 for 循环迭代，需要用 async for 迭代异步生成器
    print("\n===== async for 流式迭代 =====")
    async for chunk in fake_stream():
        print(f"  收到块: {chunk}", flush=True)


# async def + yield 定义"异步生成器"，模拟逐块返回的流
async def fake_stream():
    for i in range(1, 6):
        await asyncio.sleep(0.3)          # 每块间隔 0.3 秒
        yield f"第{i}块内容"


# Python 3.7+ 直接用 asyncio.run(main()) 启动异步程序
if __name__ == "__main__":
    asyncio.run(main())
    print("\n所有异步任务执行完毕。")
