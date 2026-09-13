# -*- coding: utf-8 -*-
"""
Claude SDK 加载 Skill 的两种方式。

1. local：把本地 SKILL.md 解析后注入 system prompt，任何兼容 Messages API
   的端点都可以运行，不依赖 Skills API。

2. hosted：先通过 ``client.skills.create`` 上传本地 Skill，再在
   ``messages.create(container={"skills": [...]})`` 中加载，并绑定代码执行
   工具。这是 Anthropic SDK 1.5 的托管 Skill / 容器能力，DeepSeek 兼容端点
   通常不支持。

运行:
    python claude-sdk/skill_loading.py --mode local
    python claude-sdk/skill_loading.py --mode local --inspect
    python claude-sdk/skill_loading.py --mode hosted
    python claude-sdk/skill_loading.py --mode hosted --inspect
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.load_env import load  # noqa: E402
from tools.skill_loader import (  # noqa: E402
    SkillLoadError,
    SkillNotFoundError,
    discover_skills,
    find_skill,
    render_full_context,
    upload_files,
)

load()

from common import DEFAULT_MODEL, get_client, text_from_message  # noqa: E402

SKILL_ROOT = PROJECT_ROOT / "skills"
SKILL_MODEL = os.environ.get("ANTHROPIC_SKILL_MODEL", "claude-opus-5")
CODE_EXECUTION_TOOL = {
    "type": "code_execution_20260521",
    "name": "code_execution",
}


def inspect_local(skill, prompt: str, model: str, max_tokens: int) -> None:
    print("=== 已加载本地 Skill ===")
    print(f"name       : {skill.name}")
    print(f"description: {skill.description}")
    print(f"path       : {skill.path}")
    print()
    print("=== 注入 system prompt 的内容 ===")
    print(render_full_context([skill]))
    print()
    print("=== 将发送的 Messages 参数 ===")
    print(
        json.dumps(
            {
                "model": model,
                "max_tokens": max_tokens,
                "system": render_full_context([skill]),
                "messages": [{"role": "user", "content": prompt}],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_local(args: argparse.Namespace) -> None:
    skill = find_skill(discover_skills(SKILL_ROOT), args.skill)
    model = args.model or DEFAULT_MODEL
    if args.inspect:
        inspect_local(skill, args.prompt, model, args.max_tokens)
        return

    client = get_client()
    response = client.messages.create(
        model=model,
        max_tokens=args.max_tokens,
        system=render_full_context([skill]),
        messages=[{"role": "user", "content": args.prompt}],
    )

    print("=== Claude 回复 ===")
    print(text_from_message(response))
    print(f"\nmodel       : {response.model}")
    print(f"stop_reason : {response.stop_reason}")
    print(f"input_tokens: {response.usage.input_tokens}")
    print(f"output_tokens: {response.usage.output_tokens}")


def inspect_hosted(skill, prompt: str, model: str, max_tokens: int) -> None:
    files = upload_files(skill)
    print("=== 准备上传的 Skill 文件 ===")
    for filename, content, media_type in files:
        print(f"filename   : {filename}")
        print(f"media_type : {media_type}")
        print(f"size       : {len(content)} bytes")
    print()
    print("=== Skills API 调用 ===")
    print("POST /v1/skills")
    print("files: SKILL.md")
    print()
    print("=== container.skills 加载参数 ===")
    print(
        json.dumps(
            {
                "model": model,
                "max_tokens": max_tokens,
                "container": {
                    "skills": [
                        {
                            "type": "custom",
                            "skill_id": "<skills.create 返回的 skill.id>",
                            "version": "latest",
                        }
                    ]
                },
                "tools": [CODE_EXECUTION_TOOL],
                "messages": [{"role": "user", "content": prompt}],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_hosted(args: argparse.Namespace) -> None:
    skill = find_skill(discover_skills(SKILL_ROOT), args.skill)
    model = args.model or SKILL_MODEL
    if args.inspect:
        inspect_hosted(skill, args.prompt, model, args.max_tokens)
        return

    client = get_client()
    if not hasattr(client, "skills"):
        raise RuntimeError(
            "当前 anthropic 版本不支持 client.skills，请安装 anthropic>=1.5.0"
        )

    created = client.skills.create(files=upload_files(skill))
    print("=== 已创建托管 Skill ===")
    print(f"id               : {created.id}")
    print(f"display_name     : {created.display_name}")
    print(f"latest_version_id: {created.latest_version_id}")
    print(f"source           : {created.source}")

    try:
        response = client.messages.create(
            model=model,
            max_tokens=args.max_tokens,
            container={
                "skills": [
                    {
                        "type": "custom",
                        "skill_id": created.id,
                        "version": "latest",
                    }
                ]
            },
            tools=[CODE_EXECUTION_TOOL],
            messages=[{"role": "user", "content": args.prompt}],
        )
        print("\n=== Claude 回复 ===")
        print(text_from_message(response))
        container = getattr(response, "container", None)
        if container is not None:
            print("\n=== 本次请求使用的容器 ===")
            print(f"id        : {container.id}")
            print(f"expires_at: {container.expires_at}")
            print(f"skills    : {container.skills}")
    finally:
        if not args.keep:
            try:
                deleted = client.skills.delete(created.id)
                print(f"\n=== 已清理托管 Skill ===\nid      : {deleted.id}")
                print(f"deleted : {deleted.deleted}")
            except Exception as cleanup_error:  # noqa: BLE001
                print(f"\n清理 Skill 失败，请手动删除 {created.id}: {cleanup_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Claude SDK Skill 加载示例")
    parser.add_argument(
        "--mode",
        choices=("local", "hosted"),
        default="local",
        help="local: SKILL.md 注入 Prompt；hosted: Anthropic Skills API",
    )
    parser.add_argument("--skill", default="text-reversal")
    parser.add_argument(
        "--prompt",
        default="请使用 text-reversal skill 反转文本：AI Learning",
    )
    parser.add_argument("--model", help="默认使用 ANTHROPIC_MODEL 或 SKILL_MODEL")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--inspect", action="store_true", help="只打印加载结果和参数")
    parser.add_argument("--keep", action="store_true", help="hosted 模式不删除测试 Skill")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.mode == "local":
            run_local(args)
        else:
            run_hosted(args)
    except (SkillLoadError, SkillNotFoundError) as exc:
        print(f"[Skill 加载失败] {exc}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"[调用失败] {exc}")
        print(
            "hosted 模式需要官方 Anthropic Skills/容器能力；"
            "DeepSeek 兼容端点请先使用 --mode local。"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
