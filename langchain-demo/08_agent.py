# -*- coding: utf-8 -*-
"""
Agent：让模型自动决定调用哪些工具，并循环执行

运行: python langchain-demo/08_agent.py
"""

from langchain.agents import create_agent
from langchain_core.tools import tool

from common import build_model


@tool
def get_weather(city: str) -> str:
    """查询一个城市的当前天气。"""
    mock = {
        "北京": "晴，26°C",
        "上海": "多云，28°C",
        "广州": "小雨，30°C",
    }
    return mock.get(city, f"暂无 {city} 的天气数据")


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。"""
    return a * b


def main():
    model = build_model(temperature=0)
    agent = create_agent(
        model,
        tools=[get_weather, multiply],
        system_prompt="需要查询天气或计算时，请使用对应工具。",
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "北京天气怎么样？顺便帮我算 1234 * 5678。",
                }
            ]
        }
    )

    print("=== Agent 完整消息轨迹 ===")
    for message in result["messages"]:
        print(f"[{message.type}] {message.content}")

    print("\n=== Agent 最终回答 ===")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
