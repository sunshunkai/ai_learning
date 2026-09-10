# -*- coding: utf-8 -*-
"""
第一个可运行脚本 —— 认识 Python 的"入口"写法
运行: python basics/hello.py  （在项目根目录）
"""

def main():
    # print 是 Python 里最常用的输出函数（等价于 Java 的 System.out.println）
    name = "顺凯"                 # 字符串变量（Python 变量无需声明类型）
    print("Hello, Python! 我是", name)

    # f-string：格式化字符串，类似 Java 的字符串拼接但更优雅
    print(f"我今年想学会在代码里调用 AI 大模型。")

    # Python 的缩进就是"代码块"，用 4 个空格表示，不要用 tab
    # （这一点和 Java 的花括号 {} 完全不同，是新手最容易踩的坑）
    if name == "顺凯":
        print("条件成立：name 确实是顺凯")


# 这个 if __name__ == "__main__" 是 Python 的标准入口写法
# 直接运行本文件时，才会执行 main()
if __name__ == "__main__":
    main()
