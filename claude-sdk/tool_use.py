# -*- coding: utf-8 -*-
"""
Claude SDK —— 工具调用 / Function Calling

场景：让 Claude 能"调用你写的函数"，例如查天气。因为 Claude 自己
不知道天气，它只能返回一个 tool_use 请求，由你执行真实函数后再把
结果喂回给它，它才能给出最终回答。

完整流程（一次"工具往返"）：
  1. 告诉 Claude 有哪些可用工具（函数定义）
  2. Claude 判断需要工具 -> 返回 tool_use（含工具名和参数）
  3. 你执行真实函数，得到结果
  4. 把工具结果作为一条 tool_result 消息回传
  5. Claude 基于工具结果给出最终回答

运行: python claude-sdk/tool_use.py
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


# ---------- 1. 定义"真实业务函数"（你想让 Claude 调用的能力） ----------
def get_weather(city: str) -> str:
    """真实查询天气的函数。演示用 mock 数据，实际可换成 HTTP 调天气 API。"""
    mock_data = {
        "北京": "晴, 22°C",
        "上海": "多云, 25°C",
        "广州": "小雨, 28°C",
    }
    return mock_data.get(city, f"暂无 {city} 的天气数据")


# 用 Claude 要求的 schema 描述这个工具
TOOLS = [
    {
        "name": "get_weather",
        "description": "查询某个中国城市的当前天气",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名，如 北京"}
            },
            "required": ["city"],
        },
    }
]


def main():
    client = Anthropic()

    # 第一轮：把工具定义 + 用户问题发给 Claude
    messages = [
        {"role": "user", "content": "北京今天天气怎么样？"},
    ]

    response = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=1024,
        tools=TOOLS,      # 声明可用工具
        messages=messages,
    )

    # ---------- 2. 检查 Claude 是否请求调用工具 ----------
    # 把 Claude 这次的回复加入历史
    messages.append({"role": "assistant", "content": response.content})

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            tool_name = block.name
            tool_input = block.input        # dict，如 {"city": "北京"}
            tool_use_id = block.id          # 需要原样回传以关联本次调用

            print(f"[Claude 请求调用工具] {tool_name}({tool_input})")

            # ---------- 3. 执行真实函数 ----------
            if tool_name == "get_weather":
                result = get_weather(tool_input.get("city", ""))
            else:
                result = f"未知工具: {tool_name}"

            print(f"[函数执行结果] {result}")

            # ---------- 4. 把结果作为 tool_result 回传 ----------
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result,
            })

    # ---------- 5. 带着工具结果再问一次，让 Claude 给最终回答 ----------
    if tool_results:
        messages.append({"role": "user", "content": tool_results})

        final = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
        print("\n[Claude 最终回答]")
        for block in final.content:
            if block.type == "text":
                print(block.text)
    else:
        print("Claude 没有请求调用工具，直接回复了。")


if __name__ == "__main__":
    main()
