from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LANGCHAIN_DEMO_ROOT = PROJECT_ROOT / "langchain-demo"
if str(LANGCHAIN_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(LANGCHAIN_DEMO_ROOT))

from llm_skill_router import LangChainSkillRouter, SkillSelection  # noqa: E402
from tools.skill_loader import Skill  # noqa: E402
from tools.skill_session import (  # noqa: E402
    HybridSkillRouter,
    RuleBasedSkillRouter,
)


def make_skill(name: str, description: str, triggers: str = "") -> Skill:
    root = Path("/tmp") / name
    return Skill(
        name=name,
        description=description,
        directory=root,
        path=root / "SKILL.md",
        markdown=f"# {name}\n",
        body=f"# {name}\n",
        metadata={"triggers": triggers},
    )


class FakeStructuredModel:
    def __init__(self, selection: SkillSelection) -> None:
        self.selection = selection
        self.schema = None
        self.calls: list[list[object]] = []

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.selection


class LangChainSkillRouterTests(unittest.TestCase):
    def test_no_candidates_skips_model_call(self) -> None:
        model = FakeStructuredModel(
            SkillSelection(skill_names=[], reason="should not run")
        )
        router = LangChainSkillRouter(model)

        decision = router.select("任意问题", (), ())

        self.assertEqual(decision.selected_names, ())
        self.assertEqual(decision.strategy, "model")
        self.assertEqual(decision.reason, "没有新的候选 Skill")
        self.assertEqual(model.calls, [])

    def test_prompt_contains_candidate_catalog_and_returns_selection(self) -> None:
        model = FakeStructuredModel(
            SkillSelection(
                skill_names=["release-notes"],
                reason="用户需要整理发布说明",
            )
        )
        router = LangChainSkillRouter(model)
        candidates = (
            make_skill("release-notes", "整理发布说明"),
            make_skill("text-reversal", "反转文本"),
        )

        decision = router.select("整理这次上线的用户变化", candidates, ())

        self.assertEqual(decision.selected_names, ("release-notes",))
        self.assertEqual(decision.strategy, "model")
        self.assertEqual(decision.reason, "用户需要整理发布说明")
        self.assertEqual(len(model.calls), 1)
        prompt_text = "\n".join(
            str(getattr(message, "content", message))
            for message in model.calls[0]
        )
        self.assertIn("release-notes: 整理发布说明", prompt_text)
        self.assertIn("text-reversal: 反转文本", prompt_text)

    def test_unknown_and_duplicate_names_are_filtered(self) -> None:
        model = FakeStructuredModel(
            SkillSelection(
                skill_names=[
                    "release-notes",
                    "missing-skill",
                    "release-notes",
                    "text-reversal",
                ],
                reason="模型结果",
            )
        )
        router = LangChainSkillRouter(model)
        candidates = (
            make_skill("release-notes", "整理发布说明"),
            make_skill("text-reversal", "反转文本"),
        )

        decision = router.select("处理文本", candidates, ())

        self.assertEqual(
            decision.selected_names,
            ("release-notes", "text-reversal"),
        )
        self.assertEqual(decision.strategy, "model")

    def test_all_unknown_names_return_empty_decision(self) -> None:
        model = FakeStructuredModel(
            SkillSelection(
                skill_names=["missing-skill"],
                reason="模型结果",
            )
        )
        router = LangChainSkillRouter(model)

        decision = router.select(
            "处理文本",
            (make_skill("text-reversal", "反转文本"),),
            (),
        )

        self.assertEqual(decision.selected_names, ())
        self.assertEqual(decision.strategy, "model")
        self.assertEqual(decision.reason, "模型返回的 Skill 均不在候选列表中")


class HybridLangChainRouterTests(unittest.TestCase):
    def test_model_runs_only_when_rules_miss(self) -> None:
        model = FakeStructuredModel(
            SkillSelection(
                skill_names=["release-notes"],
                reason="语义判断",
            )
        )
        fallback = LangChainSkillRouter(model)
        router = HybridSkillRouter(
            primary=RuleBasedSkillRouter(),
            fallback=fallback,
        )
        candidates = (
            make_skill(
                "release-notes",
                "整理发布说明",
                "release notes, 发布说明",
            ),
        )

        missed = router.select("整理上线变化", candidates, ())
        matched = router.select("请生成 release notes", candidates, ())

        self.assertEqual(missed.selected_names, ("release-notes",))
        self.assertEqual(missed.strategy, "model")
        self.assertEqual(matched.selected_names, ("release-notes",))
        self.assertEqual(matched.strategy, "rule")
        self.assertEqual(len(model.calls), 1)

    def test_explicit_loaded_skill_does_not_call_model(self) -> None:
        model = FakeStructuredModel(
            SkillSelection(skill_names=["release-notes"], reason="语义判断")
        )
        router = HybridSkillRouter(
            primary=RuleBasedSkillRouter(),
            fallback=LangChainSkillRouter(model),
        )

        decision = router.select(
            "@skill:release-notes",
            (),
            ("release-notes",),
        )

        self.assertEqual(decision.selected_names, ())
        self.assertEqual(decision.strategy, "explicit")
        self.assertEqual(model.calls, [])


class LangChainSkillLoadingCliTests(unittest.TestCase):
    def test_hybrid_inspect_prints_mode_without_calling_model(self) -> None:
        import os
        import subprocess

        script = PROJECT_ROOT / "langchain-demo" / "16_skill_loading.py"
        env = os.environ.copy()
        env.pop("DEEPSEEK_API_KEY", None)
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--mode",
                "agent",
                "--router",
                "hybrid",
                "--inspect",
                "--prompt",
                "普通文本",
            ],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("路由模式: hybrid（inspect 不调用兜底模型）", result.stdout)


if __name__ == "__main__":
    unittest.main()
