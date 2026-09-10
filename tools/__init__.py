# -*- coding: utf-8 -*-
"""公用小工具包：跨 module 共享的脚本（如从 .env 读取环境变量）。"""

from .load_env import load, require  # noqa: F401
