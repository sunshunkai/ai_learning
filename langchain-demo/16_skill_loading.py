# -*- coding: utf-8 -*-
"""
LangChain 加载 Skill 的三种模式。

agent    : 只注入 Skill 目录，Agent 通过 read_skill 工具按需读取完整指令。
          这是最接近 Agent Skill 语义的通用做法，默认使用 DeepSeek。

prompt   : 把完整 SKILL.md 直接放进 SystemMessage，模型无需工具即可遵循。

anthropic: 使用 langchain-anthropic 的 container.skills，让 Anthropic 在代码
          执行容器中加载托管 Skill。该模式需要官方 Anthropic 能力。

运行:
    python langchain-demo/16_skill_loading.py --mode agent
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
from tools.skill_loader import (  # noqa: E402
    SkillLoadError,
    SkillNotFoundError,
    discover_skills,
    find_skill,
    render_catalog,
    render_full_context,
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


def make_read_skill_tool(skills):
    @tool
    def read_skill(name: str) -> str:
        """读取一个已加载 Skill 的完整 SKILL.md 指令。"""
        return find_skill(skills, name).markdown

    return read_skill


def run_agent(args: argparse.Namespace) -> None:
    skills = discover_skills(SKILL_ROOT)
    find_skill(skills, args.skill)
    read_skill = make_read_skill_tool(skills)
    catalog = render_catalog(skills)

    if args.inspect:
        print("=== 注入 Agent 的 Skill 目录 ===")
        print(catalog)
        print()
        print("=== read_skill 工具输入 Schema ===")
        print(json.dumps(read_skill.get_input_schema().model_json_schema(), indent=2))
        print()
        print("=== 用户任务 ===")
        print(args.prompt)
        return

    model = build_model(
        model=args.model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=0,
    )
    agent = create_agent(
        model,
        tools=[read_skill],
        system_prompt=catalog,
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": args.prompt,
                }
            ]
        }
    )

    messages = result["messages"]
    print("=== Agent 消息轨迹 ===")
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
    parser.add_argument("--skill", default="text-reversal")
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
