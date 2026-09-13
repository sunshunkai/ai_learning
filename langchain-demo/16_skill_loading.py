# -*- coding: utf-8 -*-
"""
LangChain 加载 Skill 的三种模式。

agent    : 先用规则路由，未命中时默认通过 LangChain 结构化输出让模型选择
          Skill 并预加载；主 Agent 仍可通过 search_skills 查找候选，再调用
          read_skill 渐进加载完整指令。会话状态相互隔离，默认使用 DeepSeek。

prompt   : 把完整 SKILL.md 直接放进 SystemMessage，模型无需工具即可遵循。

anthropic: 使用 langchain-anthropic 的 container.skills，让 Anthropic 在代码
          执行容器中加载托管 Skill。该模式需要官方 Anthropic 能力。

运行:
    python langchain-demo/16_skill_loading.py --mode agent --router hybrid --conversation
    python langchain-demo/16_skill_loading.py --mode agent --router rule --conversation
    python langchain-demo/16_skill_loading.py --mode agent --router hybrid --inspect
    python langchain-demo/16_skill_loading.py --mode prompt --inspect
    python langchain-demo/16_skill_loading.py --mode anthropic --inspect
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

from langchain.agents import create_agent  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from common import build_model  # noqa: E402
from llm_skill_router import LangChainSkillRouter  # noqa: E402
from tools.skill_loader import (  # noqa: E402
    SkillLoadError,
    SkillNotFoundError,
    discover_skills,
    find_skill,
    load_skill_resource,
    render_full_context,
)
from tools.skill_session import (  # noqa: E402
    HybridSkillRouter,
    RuleBasedSkillRouter,
    SkillSession,
    SkillSessionManager,
)

SKILL_ROOT = PROJECT_ROOT / "skills"
DEFAULT_SKILL_MODEL = os.environ.get("ANTHROPIC_SKILL_MODEL", "claude-opus-5")
CODE_EXECUTION_TOOL = {
    "type": "code_execution_20260521",
    "name": "code_execution",
}
DEFAULT_LOCAL_PROMPT = "请使用 text-reversal skill 反转文本：AI Learning"
DEFAULT_ANTHROPIC_PROMPT = (
    "请使用 PPTX skill 创建一页标题为 'AI Learning' 的幻灯片，并总结完成结果。"
)
SESSION_SCENARIOS = (
    (
        "langchain-alpha",
        ("developer",),
        "请反转文本：AI Learning",
    ),
    (
        "langchain-alpha",
        ("developer",),
        "现在分析这个报错：ValueError: invalid literal for int()",
    ),
    (
        "langchain-beta",
        ("developer",),
        "请把本次上线中用户能感知到的变化整理成一份面向客户的说明。",
    ),
)


def make_agent_tools(session: SkillSession):
    @tool
    def search_skills(query: str) -> str:
        """Search the allowed Skill catalog without loading full instructions."""
        # 搜索工具只返回 name/description，不把完整 SKILL.md 放进上下文。
        normalized = query.casefold().strip()
        ranked = []
        for skill in session.allowed_skills:
            haystack = " ".join(
                (
                    skill.name,
                    skill.description,
                    skill.metadata.get("triggers", ""),
                )
            ).casefold()
            triggers = skill.metadata.get("triggers", "").split(",")
            score = sum(
                1
                for trigger in triggers
                if trigger.strip() and trigger.strip().casefold() in normalized
            )
            if normalized and normalized in haystack:
                score += 1
            ranked.append((score, skill))

        ranked.sort(key=lambda item: (-item[0], item[1].name))
        matches = [skill for score, skill in ranked if score > 0]
        if not matches:
            matches = list(session.allowed_skills)
        return "\n".join(
            f"- {skill.name}: {skill.description}"
            for skill in matches
        )

    @tool
    def read_skill(name: str) -> str:
        """Load one allowed Skill and return its complete SKILL.md."""
        try:
            # 模型给的是名称；权限、去重和预算检查都封装在 session.load()。
            return session.load(name).markdown
        except (SkillLoadError, SkillNotFoundError, PermissionError, ValueError) as exc:
            return f"ERROR: {exc}"

    @tool
    def read_skill_resource(name: str, path: str) -> str:
        """Load one referenced resource after its parent Skill is loaded."""
        try:
            # 资源属于二级加载，只有模型明确需要时才读取。
            skill = session.load(name)
            return load_skill_resource(skill, path)
        except (SkillLoadError, SkillNotFoundError, PermissionError, ValueError) as exc:
            return f"ERROR: {exc}"

    return [search_skills, read_skill, read_skill_resource]


def get_agent_scenarios(args: argparse.Namespace):
    if args.conversation:
        return SESSION_SCENARIOS
    roles = tuple(args.role or ("developer",))
    return ((args.session_id, roles, args.prompt),)


def build_agent_system_prompt(session: SkillSession) -> str:
    # render() 只带上当前会话已经加载的完整正文，未加载项仅保留目录。
    return (
        session.render().system_prompt
        + "\n\n执行要求：\n"
        "1. 先用 search_skills 查找相关 Skill。\n"
        "2. 需要执行某个 Skill 时，调用 read_skill(name) 加载完整正文。\n"
        "3. Skill 引用的资源只在真正需要时通过 read_skill_resource 读取。\n"
        "4. 不要假装已经读取未加载的 Skill。"
    )


def build_agent_router(
    args: argparse.Namespace,
    model=None,
):
    primary = RuleBasedSkillRouter()
    if args.router == "rule" or model is None:
        return primary
    return HybridSkillRouter(
        primary=primary,
        fallback=LangChainSkillRouter(model),
    )


def run_agent(args: argparse.Namespace) -> None:
    skills = discover_skills(SKILL_ROOT)

    if args.inspect:
        manager = SkillSessionManager(
            skills,
            router=RuleBasedSkillRouter(),
            max_loaded_skills=4,
            max_loaded_chars=40_000,
        )
        for session_id, roles, prompt in get_agent_scenarios(args):
            session = manager.session(session_id, roles=roles)
            preview = session.route(prompt)
            tools = make_agent_tools(session)
            print(f"\n=== 会话 {session_id} ===")
            print(f"用户: {prompt}")
            print(f"路由模式: {args.router}（inspect 不调用兜底模型）")
            print(f"规则预览加载: {', '.join(preview.selected_names) or '无'}")
            print(f"会话已加载: {', '.join(session.loaded_names) or '无'}")
            print("\n=== Agent tools ===")
            for item in tools:
                schema = item.get_input_schema().model_json_schema()
                print(f"{item.name}: {json.dumps(schema, ensure_ascii=False)}")
            print("\n=== 渐进式 system prompt ===")
            print(build_agent_system_prompt(session))
        return

    model = build_model(
        model=args.model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=0,
    )
    manager = SkillSessionManager(
        skills,
        router=build_agent_router(args, model),
        max_loaded_skills=4,
        max_loaded_chars=40_000,
    )
    for session_id, roles, prompt in get_agent_scenarios(args):
        session = manager.session(session_id, roles=roles)
        context = session.route(prompt)
        # 每轮重新构造 Agent prompt，让规则或模型预加载的 Skill
        # 在下一轮继续出现在 system prompt 中。
        agent = create_agent(
            model,
            tools=make_agent_tools(session),
            system_prompt=build_agent_system_prompt(session),
        )
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            }
        )

        messages = result["messages"]
        print(f"\n=== 会话 {session_id} ===")
        print(f"用户: {prompt}")
        print(f"路由策略: {context.decision.strategy}")
        print(f"路由原因: {context.decision.reason}")
        print(f"本轮加载: {', '.join(context.selected_names) or '无'}")
        print(f"会话已加载: {', '.join(session.loaded_names) or '无'}")
        print("\n=== Agent 消息轨迹 ===")
        for message in messages:
            print(f"[{message.type}] {message.content}")
        print("\n=== Agent 最终回答 ===")
        print(messages[-1].content)


def run_prompt(args: argparse.Namespace) -> None:
    skill = find_skill(discover_skills(SKILL_ROOT), args.skill)
    system_content = render_full_context([skill])
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=args.prompt),
    ]

    if args.inspect:
        print("=== SystemMessage ===")
        print(system_content)
        print()
        print("=== HumanMessage ===")
        print(args.prompt)
        return

    model = build_model(
        model=args.model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=0,
    )
    answer = model.invoke(messages)
    print("=== 模型回复 ===")
    print(answer.content)


def build_anthropic_model(args: argparse.Namespace):
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise RuntimeError(
            "缺少 langchain-anthropic，请先执行: "
            "venv/bin/pip install -r langchain-demo/requirements.txt"
        ) from exc

    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    return ChatAnthropic(
        model=args.model or DEFAULT_SKILL_MODEL,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        base_url=base_url,
        max_tokens=args.max_tokens,
        container={
            "skills": [
                {
                    "type": args.skill_type,
                    "skill_id": args.skill_id,
                    "version": "latest",
                }
            ]
        },
    )


def run_anthropic(args: argparse.Namespace) -> None:
    model = build_anthropic_model(args)
    bound_model = model.bind_tools([CODE_EXECUTION_TOOL])

    if args.inspect:
        print("=== ChatAnthropic container 配置 ===")
        print(json.dumps(model.container, ensure_ascii=False, indent=2))
        print()
        print("=== 绑定的代码执行工具 ===")
        print(json.dumps(CODE_EXECUTION_TOOL, indent=2))
        print()
        print("=== 用户任务 ===")
        print(args.prompt)
        return

    result = bound_model.invoke(args.prompt)
    print("=== Claude 回复 ===")
    print(result.content)
    container = result.response_metadata.get("container")
    if container:
        print("\n=== 本次请求使用的容器 ===")
        print(json.dumps(container, ensure_ascii=False, indent=2, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangChain Skill 加载示例")
    parser.add_argument(
        "--mode",
        choices=("agent", "prompt", "anthropic"),
        default="agent",
    )
    parser.add_argument(
        "--router",
        choices=("rule", "hybrid"),
        default="hybrid",
        help="agent 模式的路由方式：规则或规则未命中时的模型兜底",
    )
    parser.add_argument("--skill", default="text-reversal")
    parser.add_argument("--session-id", default="langchain-alpha")
    parser.add_argument(
        "--role",
        action="append",
        help="会话角色，可重复传入；默认 developer",
    )
    parser.add_argument(
        "--conversation",
        action="store_true",
        help="运行多会话渐进加载演示",
    )
    parser.add_argument("--skill-id", default="pptx")
    parser.add_argument(
        "--skill-type",
        choices=("anthropic", "custom"),
        default="anthropic",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_LOCAL_PROMPT,
    )
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--inspect", action="store_true", help="只打印加载结果和参数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "anthropic" and args.prompt == DEFAULT_LOCAL_PROMPT:
        args.prompt = DEFAULT_ANTHROPIC_PROMPT
    try:
        if args.mode == "agent":
            run_agent(args)
        elif args.mode == "prompt":
            run_prompt(args)
        else:
            run_anthropic(args)
    except (SkillLoadError, SkillNotFoundError) as exc:
        print(f"[Skill 加载失败] {exc}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"[调用失败] {exc}")
        print(
            "anthropic 模式需要官方 Skills/容器能力和 langchain-anthropic；"
            "DeepSeek 端点请使用 --mode agent 或 --mode prompt。"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
