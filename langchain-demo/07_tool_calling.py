# -*- coding: utf-8 -*-
"""
Tool Calling：声明工具 -> 模型请求调用 -> 执行工具 -> 回传结果

运行: python langchain-demo/07_tool_calling.py
"""

from langchain_core.messages import HumanMessage, ToolMessage
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


def main():
    model = build_model(temperature=0).bind_tools([get_weather])
    messages = [HumanMessage("北京今天天气怎么样？")]

    first = model.invoke(messages)
    messages.append(first)
    print(f"模型第一轮 content: {first.content}")
    print(f"tool_calls: {first.tool_calls}\n")

    if not first.tool_calls:
        print("模型没有请求工具，直接回答。")
        return

    for call in first.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        result = get_weather.invoke(tool_args) if tool_name == "get_weather" else "unknown tool"
        print(f"执行 {tool_name}({tool_args}) -> {result}")
        messages.append(
            ToolMessage(content=result, tool_call_id=call["id"])
        )

    final = model.invoke(messages)
    print(f"\n最终回答: {final.content}")


if __name__ == "__main__":
    main()
