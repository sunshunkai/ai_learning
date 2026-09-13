# -*- coding: utf-8 -*-
"""
共享 Skill 加载工具。

Skill 在本学习项目中的统一含义是：一个目录包含一份 ``SKILL.md``，其中
frontmatter 声明 name 和 description，正文描述何时使用、按什么步骤执行。
工具本身通常不直接拥有执行能力，而是告诉 Agent 或应用如何组合已有能力。

本文件只做本地发现、解析和文本渲染，不依赖 Anthropic、LangChain 或第三方
YAML 库，因此可以在不同模块中复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SKILL_FILENAME = "SKILL.md"


class SkillLoadError(ValueError):
    """Skill 文件缺失、frontmatter 不合法或找不到 Skill 时抛出。"""


class SkillNotFoundError(LookupError):
    """按名称找不到指定 Skill 时抛出。"""


@dataclass(frozen=True)
class Skill:
    """一份已加载到进程内的本地 Skill。"""

    name: str
    description: str
    directory: Path
    path: Path
    markdown: str
    body: str


def _coerce_roots(roots: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(roots, (str, Path)):
        return [Path(roots)]
    return [Path(root) for root in roots]


def _parse_frontmatter(markdown: str, directory: Path) -> tuple[str, str, str]:
    """解析 SKILL.md 最简 frontmatter，返回 name、description 和正文。"""
    lines = markdown.splitlines()
    if not lines or lines[0].strip() not in {"---", "..."}:
        raise SkillLoadError(f"{directory / SKILL_FILENAME} 缺少起始 frontmatter 分隔符")

    end = 1
    while end < len(lines) and lines[end].strip() not in {"---", "..."}:
        end += 1
    if end >= len(lines):
        raise SkillLoadError(f"{directory / SKILL_FILENAME} 的 frontmatter 没有结束分隔符")

    metadata: dict[str, str] = {}
    for raw_line in lines[1:end]:
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip().lower()] = value.strip().strip("\"'")

    body = "\n".join(lines[end + 1 :]).strip()
    name = metadata.get("name") or directory.name
    if not name:
        raise SkillLoadError(f"{directory / SKILL_FILENAME} 没有可用的 Skill name")

    description = metadata.get("description")
    if not description:
        for line in body.splitlines():
            candidate = line.strip()
            if candidate and not candidate.startswith("#"):
                description = candidate
                break
    if not description:
        description = f"执行 {directory / SKILL_FILENAME} 中描述的流程"

    return name, description, body


def discover_skills(
    roots: str | Path | Iterable[str | Path],
    *,
    recursive: bool = True,
) -> list[Skill]:
    """从若干根目录递归发现并加载 ``SKILL.md``。"""
    candidates: list[Path] = []
    for root in _coerce_roots(roots):
        if root.is_file():
            if root.name != SKILL_FILENAME:
                raise SkillLoadError(f"{root} 不是 {SKILL_FILENAME}")
            candidates.append(root)
            continue
        if not root.is_dir():
            raise SkillLoadError(f"Skill 根目录不存在: {root}")
        candidates.extend(
            sorted(
                root.rglob(SKILL_FILENAME)
                if recursive
                else root.glob(SKILL_FILENAME)
            )
        )

    seen: set[Path] = set()
    skills: list[Skill] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)

        markdown = resolved.read_text(encoding="utf-8")
        name, description, body = _parse_frontmatter(markdown, resolved.parent)
        skills.append(
            Skill(
                name=name,
                description=description,
                directory=resolved.parent,
                path=resolved,
                markdown=markdown,
                body=body,
            )
        )

    if not skills:
        formatted = ", ".join(str(root) for root in _coerce_roots(roots))
        raise SkillLoadError(f"在 {formatted} 中没有发现任何 {SKILL_FILENAME}")
    return sorted(skills, key=lambda skill: skill.name)


def find_skill(skills: Iterable[Skill], name: str) -> Skill:
    """按精确名称查找 Skill；同名不同大小写时要求结果唯一。"""
    loaded = list(skills)
    exact = [skill for skill in loaded if skill.name == name]
    if exact:
        return exact[0]

    lowered = name.casefold()
    matches = [skill for skill in loaded if skill.name.casefold() == lowered]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        available = ", ".join(skill.name for skill in loaded)
        raise SkillNotFoundError(f"未找到 Skill '{name}'。可用 Skill: {available}")
    raise SkillNotFoundError(f"Skill 名称 '{name}' 不唯一，请使用完整名称")


def render_catalog(skills: Iterable[Skill]) -> str:
    """渲染 Skill 目录，供 Agent 先判断相关性，再按需读取完整指令。"""
    lines = [
        "当前应用已经加载以下 Skill。每个 Skill 封装了一套可复用流程：",
        "",
        "1. 先根据 description 判断是否与用户任务相关。",
        "2. 需要执行某个 Skill 时，调用 read_skill(name) 读取完整 SKILL.md。",
        "3. 严格按读取到的步骤执行，不要把目录文字当作完整实现。",
        "",
        "可用 Skill:",
    ]
    for index, skill in enumerate(skills, start=1):
        lines.append(f"{index}. {skill.name}: {skill.description}")
    return "\n".join(lines)


def render_full_context(skills: Iterable[Skill]) -> str:
    """把一份或多份 Skill 的完整 Markdown 直接注入 system prompt。"""
    blocks = [
        "以下是已经加载并可以直接遵循的 Skill 指令：",
        "",
    ]
    for skill in skills:
        blocks.append(f'<skill name="{skill.name}">')
        blocks.append(skill.markdown)
        blocks.append("</skill>")
        blocks.append("")
    return "\n".join(blocks).rstrip()


def upload_files(skill: Skill) -> list[tuple[str, bytes, str]]:
    """生成 Anthropic Skills API ``files`` 参数接受的单文件上传列表。"""
    return [(SKILL_FILENAME, skill.markdown.encode("utf-8"), "text/markdown")]
