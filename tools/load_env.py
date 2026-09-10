# -*- coding: utf-8 -*-
"""
共享小工具 —— 从 .env 文件加载环境变量。

为什么需要它？
  每个终端手动 export 环境变量很麻烦，容易漏。有了它，只要在项目根目录
  放一个 .env 文件（把 .env.example 复制改名而来并填上你的 Key），
  各脚本 import 本工具后即可自动读取，无需手动 export。

用法（在任何脚本顶部）：
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # 指向项目根
    import tools.load_env as load_env
    load_env.load()          # 加载项目根目录的 .env

说明：
  - python-dotenv 负责真正解析 .env。
  - 若未安装 python-dotenv，则静默跳过（提示你先 pip install）。
  - 若系统里已设同名环境变量，以系统里为准（不覆盖），方便临时调试。
"""

import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_PATH = os.path.join(_PROJECT_ROOT, ".env")
_loaded = False


def load() -> bool:
    """加载项目根 .env 到 os.environ。已加载过则直接返回 True。"""
    global _loaded
    if _loaded:
        return True
    try:
        from dotenv import load_dotenv
        # override=False：不覆盖已存在的系统环境变量
        _loaded = load_dotenv(dotenv_path=_ENV_PATH, override=False)
        if _loaded:
            print(f"[tools.load_env] 已从 {_ENV_PATH} 加载环境变量")
        else:
            print(f"[tools.load_env] 未找到 {_ENV_PATH}（可复制 .env.example 为 .env）")
        return True
    except ImportError:
        print("[tools.load_env] 未安装 python-dotenv。请先: pip install python-dotenv")
        return False


def require(key: str) -> str:
    """取一个必填环境变量，缺失则抛错并给出友好提示。"""
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(
            f"缺少环境变量 {key}。请复制项目根目录的 .env.example 为 .env 并填入你的值，"
            f"或在终端 export {key}=... "
        )
    return value


if __name__ == "__main__":
    load()
