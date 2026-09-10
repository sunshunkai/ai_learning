# -*- coding: utf-8 -*-
"""
数据类型练习 —— 变量 / list / dict / 流程控制
这是学 AI SDK 最需要先掌握的核心类型。

运行: python basics/data_types.py
"""

def main():
    # ========== 1. 变量与基础类型 ==========
    # Python 变量不用声明类型，直接赋值（动态类型）
    # 对比 Java: int a = 1;  ->  直接 a = 1
    num = 100          # int 整数（Python 的 int 没有长度限制）
    pi = 3.14159       # float 浮点
    text = "hello"     # str 字符串
    is_ok = True       # bool 布尔，注意首字母大写 True/False

    print(f"num={num}, pi={pi}, text={text}, is_ok={is_ok}")

    # 查看变量的实际类型（type() 相当于 Java 的 instanceof 概念）
    print(f"num 的类型是: {type(num)}")

    # ========== 2. list 列表（约等于 Java 的 ArrayList） ==========
    fruits = ["apple", "banana", "cherry"]
    fruits.append("durian")          # 末尾添加元素
    print(f"\nlist 操作: {fruits}")
    print(f"第一个元素: {fruits[0]}")     # 索引从 0 开始
    print(f"长度: {len(fruits)}")

    # for 循环遍历（重点！AI 里经常遍历消息列表）
    print("遍历 fruits:")
    for f in fruits:
        print("  -", f)

    # ========== 3. dict 字典（约等于 Java 的 HashMap） ==========
    # AI 对话消息就是 list[dict] 结构，必须掌握
    person = {"name": "顺凯", "age": 30, "role": "java-developer"}
    print(f"\ndict 操作: {person}")
    print(f"取 name: {person['name']}")     # 通过 key 取值
    person["age"] = 31                      # 修改 value
    person["lang"] = "python"               # 新增 key
    print(f"修改后: {person}")

    # ========== 4. 判断与循环 ==========
    score = 85
    if score >= 90:
        grade = "A"
    elif score >= 80:      # 注意是 elif，不是 Java 的 else if
        grade = "B"
    else:
        grade = "C"
    print(f"\nscore={score} -> 等级 {grade}")

    # while 循环
    i = 0
    while i < 3:
        print(f"while 第 {i} 次")
        i += 1             # Python 没有 i++，要写 i += 1

    # ========== 5. 列表推导式（Python 特色，AI 代码常见） ==========
    # 等价于"new list, for 循环 append"的一行写法
    squares = [x * x for x in range(1, 6)]   # range(1,6) = 1,2,3,4,5
    print(f"\n列表推导式: {squares}")

    # ========== 6. 字符串常用方法 ==========
    msg = "  hello world  "
    print(f"\n去空格: '{msg.strip()}'")
    print(f"转大写: {msg.strip().upper()}")
    print(f"是否包含 world: {'world' in msg}")


if __name__ == "__main__":
    main()
