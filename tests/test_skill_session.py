from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.skill_loader import discover_skills
from tools.skill_session import (
    HybridSkillRouter,
    RuleBasedSkillRouter,
    RouteDecision,
    SkillLimitError,
    SkillPermissionError,
    SkillSessionManager,
)


def write_skill(
    root: Path,
    name: str,
    description: str,
    triggers: str,
    roles: str = "*",
) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {name}
description: {description}
triggers: {triggers}
allowed_roles: {roles}
version: 1.0.0
---

# {name}

Use this skill for {description}
""",
        encoding="utf-8",
    )


class SkillSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        write_skill(
            self.root,
            "text-reversal",
            "Reverse arbitrary text.",
            "reverse, 反转, 倒序",
        )
        write_skill(
            self.root,
            "incident-triage",
            "Triage errors and stack traces.",
            "error, exception, stack trace, 报错",
            roles="developer, operator",
        )
        self.skills = discover_skills(self.root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_manager(self, **kwargs) -> SkillSessionManager:
        return SkillSessionManager(
            self.skills,
            router=RuleBasedSkillRouter(),
            **kwargs,
        )

    def test_rule_router_loads_only_matching_skill(self) -> None:
        manager = self.make_manager()
        session = manager.session("session-a", roles={"developer"})

        result = session.route("请帮我反转这段文本：AI Learning")

        self.assertEqual(result.selected_names, ("text-reversal",))
        self.assertEqual(session.loaded_names, ("text-reversal",))
        self.assertIn("<skill name=\"text-reversal\">", result.system_prompt)
        self.assertNotIn("<skill name=\"incident-triage\">", result.system_prompt)

    def test_same_session_reuses_loaded_skill(self) -> None:
        manager = self.make_manager()
        session = manager.session("session-a", roles={"developer"})

        first = session.route("reverse the text ABC")
        second = session.route("reverse the text XYZ")

        self.assertEqual(first.selected_names, ("text-reversal",))
        self.assertEqual(second.selected_names, ())
        self.assertEqual(session.loaded_names, ("text-reversal",))
        self.assertIn("<skill name=\"text-reversal\">", second.system_prompt)

    def test_sessions_are_isolated(self) -> None:
        manager = self.make_manager()
        session_a = manager.session("session-a", roles={"developer"})
        session_b = manager.session("session-b", roles={"developer"})

        session_a.route("reverse ABC")

        self.assertEqual(session_a.loaded_names, ("text-reversal",))
        self.assertEqual(session_b.loaded_names, ())
        self.assertNotIn("<skill name=\"text-reversal\">", session_b.render().system_prompt)

    def test_permission_policy_blocks_disallowed_skill(self) -> None:
        manager = self.make_manager()
        session = manager.session("session-a", roles={"viewer"})

        with self.assertRaises(SkillPermissionError):
            session.load("incident-triage")

        self.assertNotIn("incident-triage", session.catalog_names)

    def test_explicit_skill_selection_can_load_multiple_skills(self) -> None:
        manager = self.make_manager(max_loaded_skills=3)
        session = manager.session("session-a", roles={"operator"})

        result = session.route(
            "@skill:text-reversal @skill:incident-triage 一起处理"
        )

        self.assertEqual(
            result.selected_names,
            ("text-reversal", "incident-triage"),
        )
        self.assertEqual(
            session.loaded_names,
            ("text-reversal", "incident-triage"),
        )

    def test_loaded_skill_limit_is_enforced(self) -> None:
        manager = self.make_manager(max_loaded_skills=1)
        session = manager.session("session-a", roles={"operator"})
        session.load("text-reversal")

        with self.assertRaises(SkillLimitError):
            session.load("incident-triage")

    def test_context_character_limit_is_enforced(self) -> None:
        manager = self.make_manager(max_loaded_chars=40)
        session = manager.session("session-a", roles={"developer"})

        with self.assertRaises(SkillLimitError):
            session.load("text-reversal")


class HybridSkillRouterTests(unittest.TestCase):
    class FixedRouter:
        def __init__(self, decision: RouteDecision) -> None:
            self.decision = decision
            self.calls = 0

        def select(self, query, candidates, loaded_names) -> RouteDecision:
            self.calls += 1
            return self.decision

    def test_uses_fallback_only_when_rules_do_not_match(self) -> None:
        fallback = self.FixedRouter(
            RouteDecision(("incident-triage",), strategy="model")
        )
        router = HybridSkillRouter(
            primary=RuleBasedSkillRouter(),
            fallback=fallback,
        )
        candidates = (
            type(
                "Candidate",
                (),
                {
                    "name": "incident-triage",
                    "metadata": {"triggers": "报错, exception"},
                },
            )(),
        )

        unmatched = router.select("服务为什么挂了", candidates, ())
        matched = router.select("这里有一个 exception", candidates, ())

        self.assertEqual(unmatched.selected_names, ("incident-triage",))
        self.assertEqual(matched.selected_names, ("incident-triage",))
        self.assertEqual(fallback.calls, 1)

    def test_explicit_loaded_skill_does_not_trigger_fallback(self) -> None:
        fallback = self.FixedRouter(
            RouteDecision(("incident-triage",), strategy="model")
        )
        router = HybridSkillRouter(
            primary=RuleBasedSkillRouter(),
            fallback=fallback,
        )

        decision = router.select("@skill:incident-triage", (), ("incident-triage",))

        self.assertEqual(decision.selected_names, ())
        self.assertEqual(decision.strategy, "explicit")
        self.assertEqual(fallback.calls, 0)


if __name__ == "__main__":
    unittest.main()
