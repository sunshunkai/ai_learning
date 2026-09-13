# -*- coding: utf-8 -*-
"""Use LangChain structured output to route when deterministic rules miss."""

from __future__ import annotations

from typing import Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from tools.skill_loader import Skill
from tools.skill_session import RouteDecision


class SkillSelection(BaseModel):
    """Structured routing result returned by the LangChain model."""

    skill_names: list[str] = Field(
        default_factory=list,
        description="需要加载的候选 Skill 名称；没有匹配时返回空数组。",
    )
    reason: str = Field(
        default="",
        description="选择这些 Skill 的简短原因。",
    )


class LangChainSkillRouter:
    """Model fallback that only selects from the supplied candidate catalog."""

    def __init__(self, model) -> None:
        self._selector = model.with_structured_output(SkillSelection)

    def select(
        self,
        query: str,
        candidates: Sequence[Skill],
        loaded_names: Sequence[str],
    ) -> RouteDecision:
        if not candidates:
            return RouteDecision(
                selected_names=(),
                strategy="model",
                reason="没有新的候选 Skill",
            )

        catalog = "\n".join(
            f"- {skill.name}: {skill.description}"
            for skill in candidates
        )
        loaded = ", ".join(loaded_names) or "无"
        selection = self._selector.invoke(
            [
                SystemMessage(
                    content=(
                        "你是 Skill 路由器。只能从候选列表中选择当前任务"
                        "真正需要的 Skill；信息不足时返回空数组。"
                    )
                ),
                HumanMessage(
                    content=(
                        f"当前已加载 Skill：{loaded}\n"
                        f"候选 Skill：\n{catalog}\n\n"
                        f"用户消息：{query}"
                    )
                ),
            ]
        )

        by_folded_name = {
            skill.name.casefold(): skill.name
            for skill in candidates
        }
        selected_names: list[str] = []
        seen: set[str] = set()
        for raw_name in selection.skill_names:
            actual_name = by_folded_name.get(raw_name.casefold())
            if actual_name is None or actual_name in seen:
                continue
            seen.add(actual_name)
            selected_names.append(actual_name)

        if selected_names:
            reason = selection.reason or "模型选择了相关 Skill"
        elif selection.skill_names:
            reason = "模型返回的 Skill 均不在候选列表中"
        else:
            reason = selection.reason or "模型判断无需加载 Skill"

        return RouteDecision(
            selected_names=tuple(selected_names),
            strategy="model",
            reason=reason,
        )
