# -*- coding: utf-8 -*-
"""
对话记忆：LangGraph checkpointer 持久化会话历史

运行: python langchain-demo/06_memory.py
"""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from common import build_model


def main():
    # create_agent 返回一个带状态图的 Agent；checkpointer 负责按会话保存状态。
    # tools=[] 表示这里只演示多轮消息记忆，不让 Agent 调用外部工具。
    agent = create_agent(
        build_model(temperature=0.4),
        tools=[],
        system_prompt="你是一个友好的 AI 助手。",
        checkpointer=InMemorySaver(),
    )

    # thread_id 是记忆的隔离键：同一 ID 续接历史，不同 ID 从空状态开始。
    config = {"configurable": {"thread_id": "alice"}}

    result1 = agent.invoke(
        # Agent 使用 messages 作为输入/输出状态，新增消息会追加到会话历史。
        {"messages": [HumanMessage("我叫小明，我在学 LangChain。")]},
        config=config,
    )
    reply1 = result1["messages"][-1].content
    print(f"第一轮: {reply1}\n")

    # 第二轮只传新问题；checkpointer 会根据 thread_id 自动恢复前一轮消息。
    result2 = agent.invoke(
        {"messages": [HumanMessage("我叫什么名字？我在学什么？")]},
        config=config,
    )
    reply2 = result2["messages"][-1].content
    print(f"第二轮: {reply2}\n")

    # 不同 thread_id 不共享历史，相当于不同用户
    result3 = agent.invoke(
        {"messages": [HumanMessage("你还记得我叫什么吗？")]},
        config={"configurable": {"thread_id": "bob"}},
    )
    print(f"bob 的第一轮: {result3['messages'][-1].content}")


if __name__ == "__main__":
    main()
