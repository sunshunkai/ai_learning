# -*- coding: utf-8 -*-
"""
Output Parser：字符串、逗号分隔列表、JSON

运行: python langchain-demo/04_output_parsers.py
"""

from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser,
    JsonOutputParser,
    StrOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate

from common import build_model


def main():
    model = build_model(temperature=0.2)

    print("=== StrOutputParser：把模型输出转成字符串 ===")
    # Parser 放在链尾，把统一格式的 AIMessage 转换成业务需要的 Python 类型。
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "把下面内容翻译成中文：{text}"),
        ]
    )
    chain = prompt | model | StrOutputParser()
    print(chain.invoke({"text": "Hello, LangChain"}), "\n")

    print("=== CommaSeparatedListOutputParser：解析成 list ===")
    # 解析器能生成格式说明；把它放进 system prompt 可提高输出可解析率。
    list_parser = CommaSeparatedListOutputParser()
    list_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{format_instructions}"),
            ("human", "列出 5 个主流 Python AI 框架名称。"),
        ]
    ).partial(format_instructions=list_parser.get_format_instructions())
    list_chain = list_prompt | model | list_parser
    print(list_chain.invoke({}), "\n")

    print("=== JsonOutputParser：解析成 dict ===")
    # 模型仍负责生成文本，Parser 负责把符合 JSON 语法的文本解析为 dict。
    json_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "只输出 JSON，不要输出 markdown 代码块。"),
            ("human", "介绍 {city} 的基本信息，只返回 JSON 对象，包含 name 和 country 字段。"),
        ]
    )
    json_chain = json_prompt | model | JsonOutputParser()
    print(json_chain.invoke({"city": "北京"}))


if __name__ == "__main__":
    main()
