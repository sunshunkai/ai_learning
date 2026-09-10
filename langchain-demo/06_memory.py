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
    agent = create_agent(
        build_model(temperature=0.4),
        tools=[],
        system_prompt="你是一个友好的 AI 助手。",
        checkpointer=InMemorySaver(),
    )

    config = {"configurable": {"thread_id": "alice"}}

    result1 = agent.invoke(
        {"messages": [HumanMessage("我叫小明，我在学 LangChain。")]},
        config=config,
    )
    reply1 = result1["messages"][-1].content
    print(f"第一轮: {reply1}\n")

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
