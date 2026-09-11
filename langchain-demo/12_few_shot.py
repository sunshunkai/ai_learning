# -*- coding: utf-8 -*-
"""
Few-shot Prompt：给模型提供少量输入输出示例

运行: python langchain-demo/12_few_shot.py
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

from common import build_model


def main():
    examples = [
        {"input": "这个需求很复杂", "output": "复杂度：高"},
        {"input": "改一行文案", "output": "复杂度：低"},
        {"input": "增加支付方式和退款流程", "output": "复杂度：高"},
    ]

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "判断需求复杂度，只输出“复杂度：低/中/高”。",
            ),
            few_shot,
            ("human", "{input}"),
        ]
    )

    print("=== 格式化后的消息示例 ===")
    messages = prompt.format_messages(input="增加一个导出 CSV 的按钮")
    for message in messages:
        print(f"[{message.type}] {message.content}")

    print("\n=== 调用模型 ===")
    chain = prompt | build_model(temperature=0) | StrOutputParser()
    print(chain.invoke({"input": "增加一个导出 CSV 的按钮"}))


if __name__ == "__main__":
    main()
