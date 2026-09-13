# -*- coding: utf-8 -*-
"""
AgentScope + DeepSeek 环境自检。

运行:
    python agentscope-demo/c01_01_env_check.py
"""

import os
import sys

from common import DEFAULT_BASE_URL, DEFAULT_MODEL, mask_key


def main() -> None:
    print("=== AgentScope 环境检查 ===\n")

    try:
        import agentscope

        print(f"[OK] agentscope={agentscope.__version__}")
    except ImportError:
        print("[FAIL] 未安装 agentscope")
        print("       venv/bin/pip install -r agentscope-demo/requirements.txt")
        sys.exit(1)

    try:
        from agentscope.model import DeepSeekChatModel

        cards = DeepSeekChatModel.list_models()
        names = [getattr(card, "name", str(card)) for card in cards]
        print(f"[OK] DeepSeekChatModel 可用，模型卡片: {', '.join(names)}")
    except Exception as exc:
        print(f"[WARN] 读取 DeepSeek 模型卡片失败: {exc}")

    key = os.environ.get("DEEPSEEK_API_KEY")
    print(f"[{'OK' if key else 'FAIL'}] DEEPSEEK_API_KEY={mask_key(key)}")
    print(f"[INFO] DEEPSEEK_BASE_URL={DEFAULT_BASE_URL}")
    print(f"[INFO] 默认模型={DEFAULT_MODEL}")

    print("\n下一步:")
    if not key:
        print("  在项目根目录 .env 配置 DEEPSEEK_API_KEY。")
    else:
        print("  运行 python agentscope-demo/c01_02_basic_model.py")


if __name__ == "__main__":
    main()
