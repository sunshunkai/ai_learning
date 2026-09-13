# -*- coding: utf-8 -*-
"""c01_01：检查 LangGraph 学习环境。

这个示例不调用模型，只确认 Python 版本、依赖包版本，以及 DeepSeek 在线
模式所需的配置是否齐全。运行 ``--offline`` 时不会因为缺少 Key 而失败。
"""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from common import banner, parser_for, print_json


def package_version(package: str) -> str:
    """读取已安装包版本；包不存在时返回 not-installed。"""

    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def collect_environment(offline: bool) -> dict[str, Any]:
    """收集环境检查结果，不暴露任何密钥。"""

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    return {
        "python": sys.version.split()[0],
        "langgraph": package_version("langgraph"),
        "langgraph_checkpoint": package_version("langgraph-checkpoint"),
        "langchain": package_version("langchain"),
        "langchain_openai": package_version("langchain-openai"),
        "mode": "offline" if offline else "live",
        "deepseek_api_key": "configured" if api_key else "missing",
        "deepseek_base_url": os.environ.get(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com",
        ),
        "deepseek_model": os.environ.get(
            "DEEPSEEK_MODEL",
            "deepseek-chat",
        ),
    }


def main() -> None:
    parser = parser_for(__doc__ or "LangGraph environment check")
    args = parser.parse_args()

    banner("c01_01 LangGraph 环境检查")
    result = collect_environment(args.offline)
    print_json(result)

    print("\n知识点：")
    print("- langgraph 提供图编排运行时，langgraph-checkpoint 负责状态持久化。")
    print("- langchain-openai 只负责模型适配，base_url 决定实际访问 DeepSeek。")
    print("- 离线模式使用本地确定性模型，不需要 API Key 或网络。")

    if not args.offline and result["deepseek_api_key"] != "configured":
        raise SystemExit("在线模式缺少 DEEPSEEK_API_KEY，请配置根目录 .env。")


if __name__ == "__main__":
    main()

