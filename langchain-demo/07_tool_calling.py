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
    # @tool 会把函数名、docstring 和类型注解转换成模型可理解的工具 Schema。
    # 这里的返回值是模拟数据，真实项目可替换为 HTTP API 调用。
    mock = {
        "北京": "晴，26°C",
        "上海": "多云，28°C",
        "广州": "小雨，30°C",
    }
    return mock.get(city, f"暂无 {city} 的天气数据")


def main():
    # bind_tools 只把工具 Schema 提供给模型，不会自动执行函数。
    # 模型返回 tool_calls 后，仍由应用代码决定是否执行以及如何执行。
    model = build_model(temperature=0).bind_tools([get_weather])
    messages = [HumanMessage("北京今天天气怎么样？")]

    first = model.invoke(messages)
    # 必须把模型发出的 AIMessage 也写回历史，否则后续请求缺少工具调用依据。
    messages.append(first)
    print(f"模型第一轮 content: {first.content}")
    print(f"tool_calls: {first.tool_calls}\n")

    if not first.tool_calls:
        print("模型没有请求工具，直接回答。")
        return

    for call in first.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        # 生产代码通常按名称查工具注册表，不要只根据模型文本做任意函数调用。
        result = get_weather.invoke(tool_args) if tool_name == "get_weather" else "unknown tool"
        print(f"执行 {tool_name}({tool_args}) -> {result}")
        # ToolMessage 通过 tool_call_id 与本次工具请求精确配对。
        messages.append(
            ToolMessage(content=result, tool_call_id=call["id"])
        )

    # 第二轮把工具结果回传给模型，模型据此生成面向用户的自然语言回答。
    final = model.invoke(messages)
    print(f"\n最终回答: {final.content}")


if __name__ == "__main__":
    main()
