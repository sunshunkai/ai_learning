# -*- coding: utf-8 -*-
"""
Prompt 模板：ChatPromptTemplate、partial、MessagesPlaceholder

运行: python langchain-demo/02_prompt_templates.py
"""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from common import build_model


def main():
    model = build_model(temperature=0.4)

    print("=== ChatPromptTemplate 基础用法 ===")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是资深 AI 讲师，回答要简洁。"),
            ("human", "请解释：{concept}"),
        ]
    )

    # 先不调模型，看一下模板格式化后的消息
    formatted = prompt.format_messages(concept="prompt engineering")
    for msg in formatted:
        print(f"[{msg.type}] {msg.content}")

    print("\n=== prompt | model 组成 chain ===")
    chain = prompt | model
    reply = chain.invoke({"concept": "temperature 参数"})
    print(f"回复: {reply.content}\n")

    print("=== partial：预填部分变量 ===")
    partial_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "请用{style}的风格回复。"),
            ("human", "{question}"),
        ]
    ).partial(style="中文文言文")
    reply = (partial_prompt | model).invoke({"question": "解释一下 Python 的 venv"})
    print(f"回复: {reply.content}\n")

    print("=== MessagesPlaceholder：插入历史消息 ===")
    history_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是客服助手。"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    messages = history_prompt.format_messages(
        history=[
            HumanMessage("我的订单还没到"),
            AIMessage("请稍等，我帮你查一下。"),
        ],
        question="能查到物流信息吗？",
    )
    for msg in messages:
        print(f"[{msg.type}] {msg.content}")


if __name__ == "__main__":
    main()
