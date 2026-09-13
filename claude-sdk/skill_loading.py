# -*- coding: utf-8 -*-
"""
Claude SDK 加载 Skill 的三种方式。

1. session：按会话自动路由和渐进加载 Skill，再把当前会话已加载的完整
   SKILL.md 注入 system prompt。

2. local：显式指定一个 Skill，把完整 SKILL.md 注入 system prompt。

3. hosted：按会话选中的 Skill 懒上传，再在
   ``messages.create(container={"skills": [...]})`` 中加载，并绑定代码执行
   工具。这是 Anthropic SDK 1.5 的托管 Skill / 容器能力，DeepSeek 兼容端点
   通常不支持。

运行:
    python claude-sdk/skill_loading.py --mode session --conversation --inspect
    python claude-sdk/skill_loading.py --mode session --conversation
    python claude-sdk/skill_loading.py --mode local
    python claude-sdk/skill_loading.py --mode local --inspect
    python claude-sdk/skill_loading.py --mode hosted --conversation --inspect
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.load_env import load  # noqa: E402
from tools.skill_loader import (  # noqa: E402
    Skill,
    SkillLoadError,
    SkillNotFoundError,
    discover_skills,
    find_skill,
    render_full_context,
    upload_files,
)
from tools.skill_session import (  # noqa: E402
    HybridSkillRouter,
    RouteDecision,
    RuleBasedSkillRouter,
    SkillSessionManager,
)

load()

from common import DEFAULT_MODEL, get_client, text_from_message  # noqa: E402

SKILL_ROOT = PROJECT_ROOT / "skills"
SKILL_MODEL = os.environ.get("ANTHROPIC_SKILL_MODEL", "claude-opus-5")
CODE_EXECUTION_TOOL = {
    "type": "code_execution_20260521",
    "name": "code_execution",
}
SKILL_ROUTER_TOOL = {
    "name": "select_skills",
    "description": (
        "Select only the newly needed Skills from the provided candidate list. "
        "Return an empty array when no Skill is needed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "skill_names": {
                "type": "array",
                "items": {"type": "string"},
            },
            "reason": {"type": "string"},
        },
        "required": ["skill_names", "reason"],
    },
}
SESSION_SCENARIOS = (
    (
        "session-alpha",
        ("developer",),
        "请反转文本：AI Learning",
    ),
    (
        "session-alpha",
        ("developer",),
        "现在分析这个报错：ValueError: invalid literal for int()",
    ),
    (
        "session-alpha",
        ("developer",),
        "再帮我规范化这段 JSON：{'Name': 'Ada', 'age': '36'}",
    ),
    (
        "session-beta",
        ("developer",),
        "请把这些改动整理成 release notes：新增导出功能，修复登录超时。",
    ),
)


class AnthropicSkillRouter:
    """Use a structured model call to route when deterministic rules miss."""

    def __init__(self, client, model: str, max_tokens: int = 256) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def select(
        self,
        query: str,
        candidates: Sequence[Skill],
        loaded_names: Sequence[str],
    ) -> RouteDecision:
        # 没有新的候选时不需要调用模型。
        if not candidates:
            return RouteDecision(
                selected_names=(),
                strategy="model",
                reason="没有新的候选 Skill",
            )

        # 路由模型只看轻量目录，避免在选 Skill 阶段读取所有正文。
        catalog = "\n".join(
            f"- {skill.name}: {skill.description}"
            for skill in candidates
        )
        loaded = ", ".join(loaded_names) or "无"
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=(
                "你是 Skill 路由器。只能从候选列表选择当前任务真正需要的 "
                "Skill。信息不足时返回空数组，不要为了调用而调用。"
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"当前已加载 Skill：{loaded}\n"
                        f"候选 Skill：\n{catalog}\n\n"
                        f"用户消息：{query}"
                    ),
                }
            ],
            tools=[SKILL_ROUTER_TOOL],
            tool_choice={"type": "any"},
        )

        for block in response.content:
            if (
                getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", None) == "select_skills"
            ):
                # 模型返回值先过滤到候选名称，防止越权或拼写错误。
                payload = block.input or {}
                candidate_names = {skill.name for skill in candidates}
                selected = tuple(
                    name
                    for name in payload.get("skill_names", [])
                    if name in candidate_names
                )
                return RouteDecision(
                    selected_names=selected,
                    strategy="model",
                    reason=str(payload.get("reason", "")),
                )

        return RouteDecision(
            selected_names=(),
            strategy="model",
            reason="模型没有返回有效工具调用",
        )


def build_session_manager(router) -> SkillSessionManager:
    # 这里是会话级安全边界：限制最多加载数量和完整正文总字符数。
    return SkillSessionManager(
        discover_skills(SKILL_ROOT),
        router=router,
        max_loaded_skills=4,
        max_loaded_chars=40_000,
    )


def get_scenarios(args: argparse.Namespace):
    if args.conversation:
        return SESSION_SCENARIOS
    roles = tuple(args.role or ("developer",))
    return ((args.session_id, roles, args.prompt),)


def print_session_state(
    session_id: str,
    query: str,
    context,
) -> None:
    print(f"\n=== 会话 {session_id} ===")
    print(f"用户: {query}")
    print(f"路由策略: {context.decision.strategy}")
    print(f"路由原因: {context.decision.reason}")
    print(f"本轮加载: {', '.join(context.selected_names) or '无'}")
    print(f"会话已加载: {', '.join(context.loaded_names) or '无'}")
    print(f"候选目录: {', '.join(context.catalog_names)}")


def inspect_session(args: argparse.Namespace) -> None:
    # inspect 只使用规则路由，不访问网络，方便先观察加载决策。
    manager = build_session_manager(RuleBasedSkillRouter())
    last_context = None
    for session_id, roles, prompt in get_scenarios(args):
        session = manager.session(session_id, roles=roles)
        last_context = session.route(prompt)
        print_session_state(session_id, prompt, last_context)

    if last_context is not None:
        print("\n=== 最后一个会话的最终 system prompt ===")
        print(last_context.system_prompt)


def run_session(args: argparse.Namespace) -> None:
    if args.inspect:
        inspect_session(args)
        return

    client = get_client()
    # 先用规则快速判断；规则无法判断时才调用模型做语义路由。
    router = HybridSkillRouter(
        primary=RuleBasedSkillRouter(),
        fallback=AnthropicSkillRouter(
            client,
            args.model or DEFAULT_MODEL,
        ),
    )
    manager = build_session_manager(router)

    for session_id, roles, prompt in get_scenarios(args):
        session = manager.session(session_id, roles=roles)
        # context.system_prompt = 全量允许目录 + 当前会话已加载完整正文。
        context = session.route(prompt)
        print_session_state(session_id, prompt, context)

        response = client.messages.create(
            model=args.model or DEFAULT_MODEL,
            max_tokens=args.max_tokens,
            system=context.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        print("\n=== Claude 回复 ===")
        print(text_from_message(response))


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


def inspect_hosted(args: argparse.Namespace) -> None:
    manager = build_session_manager(RuleBasedSkillRouter())
    all_skills = {skill.name: skill for skill in discover_skills(SKILL_ROOT)}
    for session_id, roles, prompt in get_scenarios(args):
        session = manager.session(session_id, roles=roles)
        context = session.route(prompt)
        print_session_state(session_id, prompt, context)

        if context.selected_names:
            # inspect 只展示本轮新增 Skill 需要上传哪些文件。
            print("\n=== 本轮需要懒上传的 Skill 文件 ===")
            for name in context.selected_names:
                print(f"Skill: {name}")
                for filename, content, media_type in upload_files(
                    all_skills[name]
                ):
                    print(f"  {filename} | {media_type} | {len(content)} bytes")

        print("\n=== 将发送的 container.skills ===")
        print(
            json.dumps(
                {
                    "model": args.model or SKILL_MODEL,
                    "container": {
                        "skills": [
                            {
                                "type": "custom",
                                "skill_id": f"<{name}-id>",
                                "version": "latest",
                            }
                            for name in context.loaded_names
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
    model = args.model or SKILL_MODEL
    if args.inspect:
        inspect_hosted(args)
        return

    client = get_client()
    if not hasattr(client, "skills"):
        raise RuntimeError(
            "当前 anthropic 版本不支持 client.skills，请安装 anthropic>=1.5.0"
        )

    router = HybridSkillRouter(
        primary=RuleBasedSkillRouter(),
        fallback=AnthropicSkillRouter(client, model),
    )
    manager = build_session_manager(router)
    all_skills = {skill.name: skill for skill in discover_skills(SKILL_ROOT)}
    # 缓存 name -> 远端 skill_id，后续轮次只上传本轮新增项。
    remote_ids: dict[str, str] = {}

    try:
        for session_id, roles, prompt in get_scenarios(args):
            session = manager.session(session_id, roles=roles)
            context = session.route(prompt)
            print_session_state(session_id, prompt, context)

            for name in context.selected_names:
                created = client.skills.create(
                    files=upload_files(all_skills[name])
                )
                remote_ids[name] = created.id
                print(
                    f"\n=== 已创建托管 Skill ===\n"
                    f"name             : {name}\n"
                    f"id               : {created.id}\n"
                    f"latest_version_id: {created.latest_version_id}"
                )

            request_args = {
                "model": model,
                "max_tokens": args.max_tokens,
                "tools": [CODE_EXECUTION_TOOL],
                "messages": [{"role": "user", "content": prompt}],
            }
            if context.loaded_names:
                request_args["container"] = {
                    "skills": [
                        {
                            "type": "custom",
                            "skill_id": remote_ids[name],
                            "version": "latest",
                        }
                        for name in context.loaded_names
                    ]
                }

            response = client.messages.create(
                **request_args,
            )
            print("\n=== Claude 回复 ===")
            print(text_from_message(response))
    finally:
        if not args.keep:
            for name, skill_id in remote_ids.items():
                try:
                    deleted = client.skills.delete(skill_id)
                    print(
                        f"\n=== 已清理托管 Skill ===\n"
                        f"name    : {name}\n"
                        f"id      : {deleted.id}\n"
                        f"deleted : {deleted.deleted}"
                    )
                except Exception as cleanup_error:  # noqa: BLE001
                    print(
                        f"\n清理 Skill 失败，请手动删除 {skill_id}: "
                        f"{cleanup_error}"
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Claude SDK Skill 加载示例")
    parser.add_argument(
        "--mode",
        choices=("session", "local", "hosted"),
        default="session",
        help=(
            "session: 自动路由和渐进加载；"
            "local: 显式注入；hosted: Anthropic Skills API"
        ),
    )
    parser.add_argument("--skill", default="text-reversal")
    parser.add_argument(
        "--prompt",
        default="请使用 text-reversal skill 反转文本：AI Learning",
    )
    parser.add_argument("--session-id", default="session-alpha")
    parser.add_argument(
        "--role",
        action="append",
        help="会话角色，可重复传入；默认 developer",
    )
    parser.add_argument(
        "--conversation",
        action="store_true",
        help="运行多会话、多轮次的渐进加载演示",
    )
    parser.add_argument("--model", help="默认使用 ANTHROPIC_MODEL 或 SKILL_MODEL")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--inspect", action="store_true", help="只打印加载结果和参数")
    parser.add_argument("--keep", action="store_true", help="hosted 模式不删除测试 Skill")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.mode == "session":
            run_session(args)
        elif args.mode == "local":
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
