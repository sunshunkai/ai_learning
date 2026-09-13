# -*- coding: utf-8 -*-
"""c10_04：应用目录、配置和本地 Server/Studio 接入检查。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from common import banner, parser_for, print_json


def application_status() -> dict:
    return {
        "recommended_layout": [
            "app/graph.py",
            "app/state.py",
            "app/tools.py",
            "langgraph.json",
            ".env",
        ],
        "fastapi_installed": importlib.util.find_spec("fastapi") is not None,
        "uvicorn_installed": importlib.util.find_spec("uvicorn") is not None,
        "langgraph_cli_installed": (
            importlib.util.find_spec("langgraph_cli") is not None
        ),
    }


def main() -> None:
    parser = parser_for(__doc__ or "Application local server")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
    )
    args = parser.parse_args()

    banner("c10_04 应用结构与本地服务")
    status = application_status()
    status["root"] = args.root
    print_json(status)
    if not status["langgraph_cli_installed"]:
        print("提示：安装 langgraph-cli[inmem] 后可运行 langgraph dev。")
    print("\n知识点：图定义、状态、工具、配置和部署入口应保持清晰边界。")


if __name__ == "__main__":
    main()

