# -*- coding: utf-8 -*-
"""
环境自检脚本 —— 学 Claude SDK 之前先跑这个
作用：确认 anthropic 是否安装、API Key 是否配置。

运行: python claude-sdk/env_check.py
"""

import os
import sys

# 和 basic_chat.py 等脚本保持一致，先读取项目根 .env
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.load_env import load  # noqa: E402
load()


def main():
    # 1. 检查 anthropic 库是否安装
    try:
        import anthropic
        print(f"[OK] anthropic 已安装，版本: {anthropic.__version__}")
    except ImportError:
        print("[FAIL] 未找到 anthropic 库。请先执行:")
        print("       pip install -r claude-sdk/requirements.txt")
        sys.exit(1)

    # 2. 检查 API Key 是否配置（SDK 默认读 ANTHROPIC_API_KEY 环境变量）
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n[WARN] 未检测到环境变量 ANTHROPIC_API_KEY。")
        print("       Claude 调用需要 API Key。配置方法见 claude-sdk/README.md。")
        # 没有 Key 时也尝试看看有没有 base_url 中转配置
    else:
        # 只显示掩码，不打印完整 Key，避免泄露
        masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "(过短?)"
        print(f"[OK] 检测到 ANTHROPIC_API_KEY（掩码）: {masked}")

    # 3. 检查是否配置了中转 base_url（可选，用于国内第三方中转）
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if base_url:
        print(f"[INFO] 检测到 ANTHROPIC_BASE_URL（自定义端点）: {base_url}")
    else:
        print("[INFO] 未配置 ANTHROPIC_BASE_URL，将使用官方默认端点 https://api.anthropic.com")

    # 4. 给出下一步提示
    print("\n下一步：")
    if api_key:
        print("  Key 已就绪，可运行 python claude-sdk/basic_chat.py 开始第一个调用。")
    else:
        print("  请先配置 ANTHROPIC_API_KEY，然后运行 basic_chat.py。")


if __name__ == "__main__":
    main()
