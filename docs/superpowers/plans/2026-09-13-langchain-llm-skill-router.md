# LangChain LLM Skill Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a LangChain-native structured-output router that selects Skills only when deterministic rules miss.

**Architecture:** `LangChainSkillRouter` adapts `ChatOpenAI.with_structured_output()` to the existing `SkillRouter` protocol. `HybridSkillRouter` runs `RuleBasedSkillRouter` first and invokes the model fallback only after a rule miss; `SkillSession` remains the authorization and budget boundary.

**Tech Stack:** Python 3.13, LangChain 1.4, langchain-core 1.6, Pydantic 2, unittest, DeepSeek via OpenAI-compatible `ChatOpenAI`

---

### Task 1: Define the LangChain router contract with failing tests

**Files:**
- Create: `tests/test_langchain_skill_router.py`

- [ ] **Step 1: Add the failing router tests**

Create `tests/test_langchain_skill_router.py` with:

```python
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify RED**

Run:

```bash
venv/bin/python -m unittest tests.test_langchain_skill_router -v
```

Expected: FAIL because `langchain-demo/llm_skill_router.py` does not exist.

### Task 2: Implement the minimal structured-output router

**Files:**
- Create: `langchain-demo/llm_skill_router.py`
- Test: `tests/test_langchain_skill_router.py`

- [ ] **Step 1: Add the minimal implementation**

Create `langchain-demo/llm_skill_router.py` with:

```python
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
```

- [ ] **Step 2: Run the focused tests to verify GREEN**

Run:

```bash
venv/bin/python -m unittest tests.test_langchain_skill_router -v
```

Expected: all tests PASS.

- [ ] **Step 3: Run the existing router tests**

Run:

```bash
venv/bin/python -m unittest tests.test_skill_session -v
```

Expected: all tests PASS.

### Task 3: Integrate hybrid routing into the LangChain Agent example

**Files:**
- Modify: `langchain-demo/16_skill_loading.py`
- Modify: `tests/test_langchain_skill_router.py`

- [ ] **Step 1: Add a failing CLI inspection test**

Append this class before `if __name__ == "__main__":` in
`tests/test_langchain_skill_router.py`:

```python
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
```

- [ ] **Step 2: Run the CLI test to verify RED**

Run:

```bash
venv/bin/python -m unittest tests.test_langchain_skill_router.LangChainSkillLoadingCliTests -v
```

Expected: FAIL because `--router` is not defined.

- [ ] **Step 3: Import the hybrid router and LangChain fallback**

In `langchain-demo/16_skill_loading.py`, add:

```python
from llm_skill_router import LangChainSkillRouter
```

Update the existing `tools.skill_session` import to include
`HybridSkillRouter`:

```python
from tools.skill_session import (
    HybridSkillRouter,
    RuleBasedSkillRouter,
    SkillSession,
    SkillSessionManager,
)
```

- [ ] **Step 4: Add the router builder**

Add this function before `run_agent()`:

```python
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
```

- [ ] **Step 5: Route before creating the Agent**

Replace the manager setup and execution section in `run_agent()` with:

```python
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
```

- [ ] **Step 6: Add the CLI option**

In `parse_args()`, add this argument after `--mode`:

```python
    parser.add_argument(
        "--router",
        choices=("rule", "hybrid"),
        default="hybrid",
        help="agent 模式的路由方式：规则或规则未命中时的模型兜底",
    )
```

- [ ] **Step 7: Add a rule-miss scenario**

Replace the third entry in `SESSION_SCENARIOS` with:

```python
    (
        "langchain-beta",
        ("developer",),
        "请把本次上线中用户能感知到的变化整理成一份面向客户的说明。",
    ),
```

- [ ] **Step 8: Run the CLI and router tests to verify GREEN**

Run:

```bash
venv/bin/python -m unittest tests.test_langchain_skill_router -v
```

Expected: all tests PASS, including the CLI inspection test.

- [ ] **Step 9: Verify the inspect command manually**

Run:

```bash
venv/bin/python langchain-demo/16_skill_loading.py \
  --mode agent --router hybrid --inspect
```

Expected: exits successfully without an API request and prints
`路由模式: hybrid（inspect 不调用兜底模型）`.

### Task 4: Document the two routing layers

**Files:**
- Modify: `langchain-demo/README.md`
- Modify: `langchain-demo/KNOWLEDGE_GUIDE.md`

- [ ] **Step 1: Update the LangChain README**

In `langchain-demo/README.md`, replace the existing run examples with:

```markdown
venv/bin/python langchain-demo/16_skill_loading.py \
  --mode agent --router hybrid --conversation
venv/bin/python langchain-demo/16_skill_loading.py \
  --mode agent --router rule --conversation
venv/bin/python langchain-demo/16_skill_loading.py \
  --mode agent --router hybrid --inspect
venv/bin/python langchain-demo/16_skill_loading.py --mode prompt
venv/bin/python langchain-demo/16_skill_loading.py --mode anthropic --inspect
```

Replace the `16_skill_loading.py` description with:

```markdown
`16_skill_loading.py` 的 `agent` 模式使用 `search_skills`、`read_skill` 和
`read_skill_resource` 三个工具实现渐进加载，并按会话隔离状态。默认
`--router hybrid` 先用 trigger 和显式名称做确定性路由，规则未命中时才通过
LangChain 结构化输出让模型从候选目录中选择 Skill；`--router rule` 保留纯规则
对照。主 Agent 仍可在后续步骤中通过工具自行搜索和加载 Skill。`prompt` 模式
保留完整注入对照；`anthropic` 模式依赖 `langchain-anthropic` 的
`container.skills` 和代码执行工具。
```

- [ ] **Step 2: Update the knowledge guide**

In `langchain-demo/KNOWLEDGE_GUIDE.md`, add this subsection after the existing
Agent-mode data flow section:

````markdown
### 2.1 规则兜底与主 Agent 工具选择

`agent` 模式包含两个不同层次的模型参与方式：

1. `--router hybrid` 在进入 Agent 前执行一次预路由。`RuleBasedSkillRouter`
   先处理 `@skill:name` 和 trigger；只有规则未命中时，
   `LangChainSkillRouter` 才通过 `with_structured_output()` 让模型返回
   `skill_names` 和 `reason`。
2. 主 Agent 仍拥有 `search_skills`、`read_skill` 和
   `read_skill_resource` 工具，可以在执行任务时继续发现和加载 Skill。

预路由模型只看到候选 Skill 的 name 和 description，不读取完整
`SKILL.md`。模型返回的名称会再次与候选集合比对，随后交给
`SkillSession.load()` 执行权限、去重和上下文预算检查。

```bash
venv/bin/python langchain-demo/16_skill_loading.py \
  --mode agent --router hybrid --conversation
venv/bin/python langchain-demo/16_skill_loading.py \
  --mode agent --router rule --conversation
```

`--router rule` 延迟最低、行为可重复，但无法理解 trigger 之外的语义表达。
`--router hybrid` 在规则未命中时增加一次模型调用，换取更强的语义泛化能力。
````

- [ ] **Step 3: Run the complete test suite**

Run:

```bash
venv/bin/python -m unittest discover -s tests -v
```

Expected: all tests PASS.

- [ ] **Step 4: Check the final diff**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only the intended source, test, and documentation
files are modified.

- [ ] **Step 5: Commit the implementation**

Run:

```bash
git add langchain-demo/16_skill_loading.py \
  langchain-demo/llm_skill_router.py \
  langchain-demo/README.md \
  langchain-demo/KNOWLEDGE_GUIDE.md \
  tests/test_langchain_skill_router.py
git commit -m "feat(skills): add LangChain LLM skill router"
```
