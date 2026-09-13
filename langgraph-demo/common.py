# -*- coding: utf-8 -*-
"""LangGraph demo 公共工具。

这里集中处理项目路径、环境变量、命令行参数和 DeepSeek 模型创建，
避免每个教学示例重复实现相同的基础设施。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.load_env import load as load_env  # noqa: E402

load_env()


def parser_for(description: str) -> argparse.ArgumentParser:
    """创建一个带统一选项的 ArgumentParser。"""

    parser = argparse.ArgumentParser(description=description)
    add_common_args(parser)
    return parser


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """为示例添加离线、模型和详细输出选项。"""

    parser.add_argument(
        "--offline",
        action="store_true",
        help="使用确定性本地 FakeModel，不读取 Key，也不访问网络。",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat",
        help="在线模式使用的 DeepSeek 模型，默认读取 DEEPSEEK_MODEL。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印更详细的中间状态。",
    )


def build_model(
    *,
    offline: bool,
    model: str = "deepseek-chat",
    temperature: float = 0.0,
    structured_responses: dict[type, object] | None = None,
    tool_responses: list[dict[str, Any]] | None = None,
    text_responses: list[str] | None = None,
    **kwargs: Any,
):
    """创建 DeepSeek 模型或确定性离线模型。"""

    if offline:
        from fixtures.fake_models import DeterministicFakeChatModel

        return DeterministicFakeChatModel(
            structured_responses=structured_responses,
            tool_responses=tool_responses,
            text_responses=text_responses,
        )

    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    if not api_key:
        raise RuntimeError(
            "缺少 DEEPSEEK_API_KEY。请在项目根目录 .env 中配置，"
            "或使用 --offline 运行本地示例。"
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        **kwargs,
    )


def banner(title: str) -> None:
    """打印清晰的章节标题。"""

    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_json(value: object) -> None:
    """以 UTF-8 JSON 打印状态或结果。"""

    print(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=repr,
        )
    )


def ensure_project_root_on_path() -> None:
    """确保从任意工作目录运行脚本时都能导入项目根模块。"""

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

