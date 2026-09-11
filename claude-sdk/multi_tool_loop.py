# -*- coding: utf-8 -*-
"""
Claude SDK —— 多工具调用循环与异常处理

这个案例把工具调用封装成可重复循环：
  模型 -> tool_use -> 执行多个工具 -> tool_result -> 模型继续

运行: python claude-sdk/multi_tool_loop.py
"""

from common import DEFAULT_MODEL, get_client, text_from_message


def get_weather(city: str) -> str:
    mock = {
        "北京": "晴，26°C",
        "上海": "多云，28°C",
        "广州": "小雨，30°C",
    }
    return mock.get(city, f"暂无 {city} 的天气数据")


def calculate(a: float, operator: str, b: float) -> float:
    operations = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "/": lambda x, y: x / y,
    }
    if operator not in operations:
        raise ValueError(f"不支持的运算符: {operator}")
    if operator == "/" and b == 0:
        raise ValueError("除数不能为 0")
    return operations[operator](a, b)


def search_docs(query: str) -> str:
    docs = {
        "agent": "Agent 通常由模型、工具和循环控制逻辑组成。",
        "rag": "RAG 先检索资料，再让模型基于资料生成答案。",
    }
    return docs.get(query.lower(), f"没有找到与 {query} 相关的文档")


TOOLS = [
    {
        "name": "get_weather",
        "description": "查询城市天气",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "name": "calculate",
        "description": "计算两个数字的加减乘除",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "operator": {"type": "string", "enum": ["+", "-", "*", "/"]},
                "b": {"type": "number"},
            },
            "required": ["a", "operator", "b"],
        },
    },
    {
        "name": "search_docs",
        "description": "查询本地学习资料，关键词支持 agent、rag",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


def execute_tool(name: str, args: dict):
    if name == "get_weather":
        return get_weather(**args)
    if name == "calculate":
        return calculate(**args)
    if name == "search_docs":
        return search_docs(**args)
    raise ValueError(f"未知工具: {name}")


def main():
    client = get_client()
    messages = [
        {
            "role": "user",
            "content": (
                "请完成三件事：查询北京天气；计算 12 * 8；"
                "查询本地资料里的 agent 是什么。"
            ),
        }
    ]

    for step in range(1, 7):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            print(f"[第 {step} 轮] 调用 {block.name}({block.input})")
            try:
                result = execute_tool(block.name, block.input)
                is_error = False
            except Exception as exc:
                result = f"工具执行失败: {exc}"
                is_error = True

            print(f"          结果: {result}")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                    "is_error": is_error,
                }
            )

        if not tool_results:
            print("\n[最终回答]")
            print(text_from_message(response))
            return

        messages.append({"role": "user", "content": tool_results})

    print("达到最大循环次数，已停止。")


if __name__ == "__main__":
    main()
