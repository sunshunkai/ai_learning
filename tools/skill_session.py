# -*- coding: utf-8 -*-
"""Production-oriented Skill routing and per-conversation loading helpers."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence

from tools.skill_loader import (
    Skill,
    SkillNotFoundError,
    find_skill,
    render_full_context,
)


class SkillPermissionError(PermissionError):
    """The current principal cannot use the requested Skill."""


class SkillLimitError(ValueError):
    """Loading the Skill would exceed a session budget."""


@dataclass(frozen=True)
class RouteDecision:
    """A router decision made against one candidate catalog."""

    # selected_names 只包含本轮新增加载的 Skill，不包含上一轮已加载项。
    selected_names: tuple[str, ...]
    # strategy 便于审计：explicit、rule、model 或 render。
    strategy: str
    reason: str = ""


class SkillRouter(Protocol):
    """Select zero or more new Skills for the current user turn."""

    def select(
        self,
        query: str,
        candidates: Sequence[Skill],
        loaded_names: Sequence[str],
    ) -> RouteDecision:
        """Return candidate Skill names that should be loaded."""


@dataclass(frozen=True)
class RoleSkillPolicy:
    """Filter Skills by the frontmatter ``allowed_roles`` metadata."""

    def allows(self, skill: Skill, roles: Iterable[str]) -> bool:
        # allowed_roles 支持逗号分隔的多个角色，* 表示所有角色可用。
        required = _split_metadata(skill.metadata.get("allowed_roles", "*"))
        if "*" in required:
            return True
        principal_roles = {role.casefold() for role in roles}
        return bool(required & principal_roles)


@dataclass(frozen=True)
class SkillContext:
    """The progressive context prepared for one model invocation."""

    session_id: str
    selected_names: tuple[str, ...]
    loaded_names: tuple[str, ...]
    catalog_names: tuple[str, ...]
    system_prompt: str
    decision: RouteDecision


@dataclass
class _SessionState:
    loaded_names: list[str] = field(default_factory=list)
    route_history: list[RouteDecision] = field(default_factory=list)


class RuleBasedSkillRouter:
    """Fast deterministic routing for explicit mentions and trigger terms."""

    _EXPLICIT_PATTERN = re.compile(r"@skill:([A-Za-z0-9_-]+)")

    def select(
        self,
        query: str,
        candidates: Sequence[Skill],
        loaded_names: Sequence[str],
    ) -> RouteDecision:
        # 只允许从服务端已经过滤过的 candidates 中选择。
        by_name = {skill.name.casefold(): skill for skill in candidates}
        loaded = {name.casefold() for name in loaded_names}

        # 第一优先级：用户显式写出的 @skill:name。
        explicit_names: list[str] = []
        explicit_loaded = False
        for match in self._EXPLICIT_PATTERN.finditer(query):
            requested = match.group(1).casefold()
            skill = by_name.get(requested)
            if skill is not None and skill.name not in explicit_names:
                explicit_names.append(skill.name)
            elif requested in loaded:
                explicit_loaded = True
        if explicit_names:
            return RouteDecision(
                selected_names=tuple(explicit_names),
                strategy="explicit",
                reason="用户显式指定了 Skill",
            )
        if explicit_loaded:
            return RouteDecision(
                selected_names=(),
                strategy="explicit",
                reason="用户显式指定的 Skill 已在本会话加载",
            )

        # 第二优先级：根据 frontmatter triggers 做确定性关键词匹配。
        folded_query = query.casefold()
        scores: list[tuple[int, int, Skill]] = []
        for index, skill in enumerate(candidates):
            triggers = _split_metadata(skill.metadata.get("triggers", ""))
            score = sum(
                1
                for trigger in triggers
                if trigger and trigger in folded_query
            )
            if score:
                scores.append((score, index, skill))

        if not scores:
            return RouteDecision(
                selected_names=(),
                strategy="rule",
                reason="没有命中显式名称或 trigger",
            )

        best_score = max(item[0] for item in scores)
        selected = tuple(
            skill.name
            for score, _, skill in sorted(scores)
            if score == best_score
        )
        return RouteDecision(
            selected_names=selected,
            strategy="rule",
            reason=f"命中 trigger，最高分 {best_score}",
        )


class HybridSkillRouter:
    """Use deterministic rules first, then a model or retrieval fallback."""

    def __init__(self, *, primary: SkillRouter, fallback: SkillRouter) -> None:
        self._primary = primary
        self._fallback = fallback

    def select(
        self,
        query: str,
        candidates: Sequence[Skill],
        loaded_names: Sequence[str],
    ) -> RouteDecision:
        # 规则能确定时直接结束，避免为了选 Skill 多调用一次模型。
        decision = self._primary.select(query, candidates, loaded_names)
        if decision.selected_names or decision.strategy == "explicit":
            return decision
        return self._fallback.select(query, candidates, loaded_names)


class SkillSessionManager:
    """Own allowed Skills and in-memory state for each conversation."""

    def __init__(
        self,
        skills: Iterable[Skill],
        *,
        router: SkillRouter,
        policy: RoleSkillPolicy | None = None,
        max_loaded_skills: int = 3,
        max_loaded_chars: int = 30_000,
    ) -> None:
        self._skills = tuple(skills)
        self._router = router
        self._policy = policy or RoleSkillPolicy()
        self._max_loaded_skills = max_loaded_skills
        self._max_loaded_chars = max_loaded_chars
        self._states: dict[str, _SessionState] = {}
        self._lock = threading.RLock()

    def session(
        self,
        session_id: str,
        *,
        roles: Iterable[str] = ("*",),
    ) -> "SkillSession":
        if not session_id:
            raise ValueError("session_id 不能为空")
        # 每个 session_id 对应独立状态，防止不同会话互相污染。
        with self._lock:
            self._states.setdefault(session_id, _SessionState())
        return SkillSession(self, session_id, roles=frozenset(roles))

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._states.pop(session_id, None)

    def audit(self, session_id: str) -> tuple[RouteDecision, ...]:
        with self._lock:
            state = self._states.get(session_id)
            return tuple(state.route_history) if state else ()


class SkillSession:
    """One principal's view of a conversation-scoped Skill lifecycle."""

    def __init__(
        self,
        manager: SkillSessionManager,
        session_id: str,
        *,
        roles: frozenset[str],
    ) -> None:
        self._manager = manager
        self.session_id = session_id
        self._roles = roles

    @property
    def allowed_skills(self) -> tuple[Skill, ...]:
        return tuple(
            skill
            for skill in self._manager._skills
            if self._manager._policy.allows(skill, self._roles)
        )

    @property
    def catalog_names(self) -> tuple[str, ...]:
        return tuple(skill.name for skill in self.allowed_skills)

    @property
    def loaded_names(self) -> tuple[str, ...]:
        allowed = set(self.catalog_names)
        with self._manager._lock:
            state = self._manager._states[self.session_id]
            # 如果角色权限发生变化，自动移除已经无权使用的 Skill。
            state.loaded_names = [
                name for name in state.loaded_names if name in allowed
            ]
            return tuple(state.loaded_names)

    @property
    def loaded_skills(self) -> tuple[Skill, ...]:
        return tuple(find_skill(self.allowed_skills, name) for name in self.loaded_names)

    def load(self, name: str) -> Skill:
        """Validate and load one Skill, enforcing session budgets."""
        try:
            skill = find_skill(self._manager._skills, name)
        except SkillNotFoundError:
            raise

        # 模型只能给出名称，真正读取和授权校验必须在应用侧完成。
        if not self._manager._policy.allows(skill, self._roles):
            raise SkillPermissionError(
                f"当前角色无权加载 Skill '{name}'"
            )

        with self._manager._lock:
            state = self._manager._states[self.session_id]
            # 重复加载视为幂等操作，不重复消耗上下文预算。
            if skill.name in state.loaded_names:
                return skill
            if len(state.loaded_names) >= self._manager._max_loaded_skills:
                raise SkillLimitError(
                    f"会话最多只能加载 {self._manager._max_loaded_skills} 个 Skill"
                )

            # 预算是按完整 markdown 计算的，因为最终会整段进入 system prompt。
            loaded_chars = sum(
                len(find_skill(self._manager._skills, loaded_name).markdown)
                for loaded_name in state.loaded_names
            )
            if loaded_chars + len(skill.markdown) > self._manager._max_loaded_chars:
                raise SkillLimitError(
                    "加载 Skill 后将超过会话上下文预算"
                )
            state.loaded_names.append(skill.name)
        return skill

    def route(self, query: str) -> SkillContext:
        """Select and load only Skills not already active in this session."""
        # 已加载 Skill 会在本轮继续复用，因此不再作为候选。
        loaded = set(self.loaded_names)
        candidates = tuple(
            skill for skill in self.allowed_skills if skill.name not in loaded
        )
        decision = self._manager._router.select(query, candidates, tuple(loaded))

        # 路由器可能来自模型，必须再次确认返回名称属于候选集合。
        candidate_names = {skill.name for skill in candidates}
        invalid = [
            name for name in decision.selected_names if name not in candidate_names
        ]
        if invalid:
            joined = ", ".join(invalid)
            raise SkillPermissionError(
                f"路由器选择了无效或不可用的 Skill: {joined}"
            )

        # 只有通过 load() 的权限、去重和预算检查后才算真正加载。
        selected: list[str] = []
        for name in decision.selected_names:
            self.load(name)
            selected.append(name)

        with self._manager._lock:
            self._manager._states[self.session_id].route_history.append(decision)

        return self.render(
            selected_names=tuple(selected),
            decision=decision,
        )

    def render(
        self,
        *,
        selected_names: tuple[str, ...] = (),
        decision: RouteDecision | None = None,
    ) -> SkillContext:
        """Build the catalog and the currently loaded full Skill bodies."""
        loaded_skills = self.loaded_skills
        # 未加载 Skill 只暴露 name/description，已加载 Skill 才暴露完整正文。
        catalog_lines = [
            "当前会话允许以下 Skill。",
            "完整 Skill 只会渐进加载到当前会话，不要假设未加载 Skill 的细节。",
            "",
        ]
        loaded_names = {skill.name for skill in loaded_skills}
        for skill in self.allowed_skills:
            marker = "loaded" if skill.name in loaded_names else "available"
            catalog_lines.append(
                f"- [{marker}] {skill.name}: {skill.description}"
            )

        sections = ["\n".join(catalog_lines)]
        if loaded_skills:
            sections.append(render_full_context(loaded_skills))

        return SkillContext(
            session_id=self.session_id,
            selected_names=selected_names,
            loaded_names=tuple(skill.name for skill in loaded_skills),
            catalog_names=self.catalog_names,
            system_prompt="\n\n".join(sections),
            decision=decision
            or RouteDecision(
                selected_names=selected_names,
                strategy="render",
            ),
        )


def _split_metadata(value: str) -> set[str]:
    return {
        item.strip().casefold()
        for item in value.split(",")
        if item.strip()
    }
