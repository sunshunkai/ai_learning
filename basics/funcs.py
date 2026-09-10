# -*- coding: utf-8 -*-
"""
函数练习 —— Python 函数与 Java 的差异点
运行: python basics/funcs.py
"""

# ========== 1. 基本函数定义 ==========
# Python 用 def 定义函数，没有访问修饰符(public/private)、没有返回类型声明
def add(a, b):
    """求两个数的和（三引号是文档字符串 docstring，类似 Javadoc）"""
    return a + b


def greet(name):
    return f"你好, {name}!"


# ========== 2. 默认参数值（Java 用重载，Python 直接给默认值） ==========
# Java 里你可能写两个重载方法；Python 只需要一个带默认值的函数
def say_hello(name="同学"):
    """不传 name 就用默认值 '同学'"""
    return f"Hello, {name}"


# ========== 3. 不定长参数 *args / **kwargs（Python 特色） ==========
# *args 收集多余的位置参数，变成一个 tuple
def sum_all(*nums):
    total = 0
    for n in nums:
        total += n
    return total

# **kwargs 收集多余的命名参数，变成一个 dict
def describe_person(name, **info):
    print(f"姓名: {name}")
    for k, v in info.items():
        print(f"  {k} = {v}")


def main():
    # 调用基本函数
    print(f"add(3, 5) = {add(3, 5)}")
    print(greet("顺凯"))

    # 默认参数：带与不带都行
    print(say_hello())            # -> Hello, 同学
    print(say_hello("小王"))      # -> Hello, 小王

    # *args：想传几个传几个
    print(f"sum_all(1,2,3,4) = {sum_all(1, 2, 3, 4)}")

    # **kwargs
    describe_person("顺凯", age=30, city="北京", job="Java开发")


if __name__ == "__main__":
    main()
