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

import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

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
    # markdown 保留完整原文件，注入 system prompt 时使用。
    markdown: str
    # body 去掉 frontmatter，适合只展示或处理 Skill 正文。
    body: str
    # metadata 保留 triggers、allowed_roles、version 等扩展字段。
    metadata: Mapping[str, str]


def _coerce_roots(roots: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(roots, (str, Path)):
        return [Path(roots)]
    return [Path(root) for root in roots]


def _parse_frontmatter(
    markdown: str,
    directory: Path,
) -> tuple[str, str, str, dict[str, str]]:
    """解析 SKILL.md 最简 frontmatter，返回名称、描述、正文和元数据。"""
    lines = markdown.splitlines()
    if not lines or lines[0].strip() not in {"---", "..."}:
        raise SkillLoadError(f"{directory / SKILL_FILENAME} 缺少起始 frontmatter 分隔符")

    end = 1
    while end < len(lines) and lines[end].strip() not in {"---", "..."}:
        end += 1
    if end >= len(lines):
        raise SkillLoadError(f"{directory / SKILL_FILENAME} 的 frontmatter 没有结束分隔符")

    # 这里只解析简单的 key: value frontmatter，复杂 YAML 应换正式解析库。
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

    return name, description, body, metadata


def discover_skills(
    roots: str | Path | Iterable[str | Path],
    *,
    recursive: bool = True,
) -> list[Skill]:
    """从若干根目录递归发现并加载 ``SKILL.md``。"""
    candidates: list[Path] = []
    for root in _coerce_roots(roots):
        # 允许直接传一个 SKILL.md，也允许传包含多个 Skill 的目录。
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

    # 多个 root 可能指向同一个文件，用 resolve 后的路径去重。
    seen: set[Path] = set()
    skills: list[Skill] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)

        markdown = resolved.read_text(encoding="utf-8")
        name, description, body, metadata = _parse_frontmatter(
            markdown,
            resolved.parent,
        )
        skills.append(
            Skill(
                name=name,
                description=description,
                directory=resolved.parent,
                path=resolved,
                markdown=markdown,
                body=body,
                metadata=metadata,
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


def load_skill_resource(skill: Skill, relative_path: str) -> str:
    """Load one Skill resource without allowing path traversal."""
    resource_path = Path(relative_path)
    # 模型只能提供 Skill 内部的相对路径，不能访问任意文件。
    if resource_path.is_absolute():
        raise SkillLoadError("Skill 资源路径必须是相对路径")

    # resolve 后再检查父目录，能够同时挡住 ../ 和符号链接逃逸。
    root = skill.directory.resolve()
    target = (root / resource_path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise SkillLoadError("Skill 资源路径不能越过 Skill 目录") from exc

    if not target.is_file():
        raise SkillLoadError(f"Skill 资源不存在: {relative_path}")
    return target.read_text(encoding="utf-8")


def upload_files(skill: Skill) -> list[tuple[str, bytes, str]]:
    """生成 Anthropic Skills API 接受的完整 Skill 文件上传列表。"""
    root = skill.directory.resolve()
    # SKILL.md 会被单独放在首位，这里收集它引用的其他资源。
    resources = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name == SKILL_FILENAME:
            continue
        relative = path.relative_to(root)
        # 不把隐藏文件、缓存文件或虚拟环境内容上传到远端 Skill。
        if any(
            part.startswith(".") or part == "__pycache__"
            for part in relative.parts
        ):
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise SkillLoadError(
                f"Skill 资源不能位于 Skill 目录之外: {relative}"
            ) from exc
        resources.append((relative, resolved))

    uploaded: list[tuple[str, bytes, str]] = [
        (SKILL_FILENAME, skill.markdown.encode("utf-8"), "text/markdown")
    ]
    for relative, path in sorted(resources, key=lambda item: str(item[0])):
        media_type = (
            mimetypes.guess_type(path.name)[0]
            or "application/octet-stream"
        )
        uploaded.append(
            (str(relative), path.read_bytes(), media_type)
        )
    return uploaded
