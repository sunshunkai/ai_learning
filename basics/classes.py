# -*- coding: utf-8 -*-
"""
类练习 —— Python 类与 Java 的差异（入门够用即可）
运行: python basics/classes.py

说明：学 AI SDK 时你会经常看到 class（比如客户端、消息、工具），
但通常只需要"能看懂 + 会实例化调用"，不必深入继承/多态。
"""

class ChatClient:
    """
    模拟一个简单的 AI 客户端类，用来演示 Python 类的写法。
    真正学 Claude SDK 时，Anthropic 也是这种风格。
    """

    # 构造方法：Python 用 __init__(self, ...) 而不是类名
    def __init__(self, api_key, model="default-model"):
        # self 相当于 Java 里的 this，必须显式写在第一个参数
        self.api_key = api_key      # 实例属性
        self.model = model          # 实例属性
        self.history = []           # 用 list 存对话历史

    # 实例方法：第一个参数一定是 self
    def chat(self, user_text):
        """模拟一次对话，返回模型回复"""
        self.history.append({"role": "user", "content": user_text})
        # 这里假装调用了模型，真正接 Claude SDK 时替换成 API 调用
        reply = f"[{self.model} 模拟回复] 我收到了: {user_text}"
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def get_history(self):
        """返回对话历史"""
        return self.history

    # 一个简单方法，展示"类里可以写业务逻辑"
    def switch_model(self, new_model):
        self.model = new_model
        print(f"模型已切换为: {self.model}")


def main():
    # 实例化：Python 不需要 new 关键字
    client = ChatClient(api_key="sk-demo", model="claude-test")

    # 调用方法
    print(client.chat("你好"))
    print(client.chat("你是谁？"))

    print("\n完整对话历史:")
    for msg in client.get_history():
        print(f"  [{msg['role']}] {msg['content']}")

    # 直接访问属性（Python 没有 private，靠命名约定 _下划线表示私有）
    print(f"\n当前模型: {client.model}")
    client.switch_model("claude-sonnet")


if __name__ == "__main__":
    main()
