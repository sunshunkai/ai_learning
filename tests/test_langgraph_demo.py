from __future__ import annotations

import sys
import unittest
import asyncio
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LANGGRAPH_DEMO_ROOT = PROJECT_ROOT / "langgraph-demo"
if str(LANGGRAPH_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(LANGGRAPH_DEMO_ROOT))

from common import build_model, parser_for  # noqa: E402
from fixtures.fake_models import DeterministicFakeChatModel  # noqa: E402


class Route(BaseModel):
    route: Literal["math", "general"]


class CommonHelperTests(unittest.TestCase):
    def test_common_parser_accepts_offline_model_and_verbose(self) -> None:
        parser = parser_for("demo")
        args = parser.parse_args(
            ["--offline", "--model", "deepseek-reasoner", "--verbose"]
        )

        self.assertTrue(args.offline)
        self.assertEqual(args.model, "deepseek-reasoner")
        self.assertTrue(args.verbose)

    def test_offline_model_uses_structured_override(self) -> None:
        model = build_model(
            offline=True,
            structured_responses={Route: Route(route="math")},
        )

        result = model.with_structured_output(Route).invoke("2+2")

        self.assertEqual(result.route, "math")


class FakeModelTests(unittest.TestCase):
    def test_tool_response_precedes_text_response(self) -> None:
        model = DeterministicFakeChatModel(
            tool_responses=[
                {"name": "add", "args": {"a": 1, "b": 2}},
            ],
            text_responses=["结果是 3"],
        )

        tool_message = model.invoke("计算 1+2")
        final_message = model.invoke("继续")

        self.assertEqual(tool_message.tool_calls[0]["name"], "add")
        self.assertEqual(final_message.content, "结果是 3")

    def test_stream_returns_deterministic_chunks(self) -> None:
        model = DeterministicFakeChatModel(text_responses=["abc"])

        chunks = list(model.stream("hi"))

        self.assertEqual("".join(chunk.content for chunk in chunks), "abc")


class GraphFoundationTests(unittest.TestCase):
    def test_hello_graph_returns_greeting(self) -> None:
        from c01_02_hello_state_graph import run_hello_graph

        result = run_hello_graph("LangGraph")

        self.assertEqual(result["message"], "hello LangGraph")

    def test_add_messages_reducer_accumulates_history(self) -> None:
        from c02_02_reducers_messages import build_message_graph

        graph = build_message_graph()
        result = graph.invoke(
            {
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "second"},
                ]
            }
        )

        self.assertEqual(len(result["messages"]), 2)
        self.assertEqual(result["messages"][1].content, "second")

    def test_counter_reducer_adds_updates(self) -> None:
        from c02_02_reducers_messages import build_counter_graph

        graph = build_counter_graph()
        first = graph.invoke({"count": 0})
        second = graph.invoke({"count": 1})

        self.assertEqual(first["count"], 1)
        self.assertEqual(second["count"], 2)

    def test_private_state_is_not_in_public_output(self) -> None:
        from c02_03_input_output_private_state import build_private_graph

        result = build_private_graph().invoke({"question": "hello"})

        self.assertEqual(result["answer"], "HELLO")
        self.assertNotIn("draft", result)


class AdvancedControlFlowTests(unittest.TestCase):
    def test_parallel_branches_merge_with_reducer(self) -> None:
        from c03_01_branch_parallel import build_branch_graph

        result = build_branch_graph().invoke({"branches": []})

        self.assertEqual(set(result["branches"]), {"left", "right"})

    def test_conditional_graph_selects_expected_route(self) -> None:
        from c03_02_conditional_routing import build_conditional_graph

        graph = build_conditional_graph()

        self.assertEqual(graph.invoke({"value": 8})["result"], "large")
        self.assertEqual(graph.invoke({"value": 2})["result"], "small")

    def test_loop_stops_at_limit(self) -> None:
        from c03_03_loops_recursion import build_loop_graph

        result = build_loop_graph().invoke({"count": 0, "limit": 3})

        self.assertEqual(result["count"], 3)

    def test_send_processes_all_items(self) -> None:
        from c03_04_send_map_reduce import build_send_graph

        result = build_send_graph().invoke({"items": ["a", "b", "c"]})

        self.assertEqual(set(result["results"]), {"A", "B", "C"})

    def test_command_updates_state_and_routes(self) -> None:
        from c03_05_command import build_command_graph

        result = build_command_graph().invoke({"value": 1})

        self.assertEqual(result["value"], 3)
        self.assertEqual(result["path"], ["first", "second"])

    def test_reliability_graph_retries_and_uses_context(self) -> None:
        from c03_06_runtime_retry_timeout_cache import (
            DemoContext,
            build_reliability_graph,
        )

        result = asyncio.run(
            build_reliability_graph().ainvoke(
                {"value": 3},
                context=DemoContext(multiplier=2),
            )
        )

        self.assertEqual(result["result"], 6)
        self.assertEqual(result["attempts"], 2)


class FunctionalApiTests(unittest.TestCase):
    def test_entrypoint_calls_task(self) -> None:
        from c04_01_entrypoint_task import double_workflow

        self.assertEqual(double_workflow.invoke(5), 10)

    def test_parallel_futures_are_aggregated(self) -> None:
        from c04_02_futures_parallel import parallel_square_workflow

        self.assertEqual(parallel_square_workflow.invoke(4), 24)

    def test_retry_workflow_recovers(self) -> None:
        from c04_03_retry_resume import run_retry_workflow

        result = run_retry_workflow(7)

        self.assertEqual(result["value"], 14)
        self.assertEqual(result["attempts"], 2)

    def test_graph_and_functional_api_interoperate(self) -> None:
        from c04_04_graph_functional_interop import interop_workflow

        self.assertEqual(interop_workflow.invoke(" hi "), "HI-GRAPH")


class WorkflowAgentTests(unittest.TestCase):
    def test_prompt_chaining_runs_in_order(self) -> None:
        from c05_02_prompt_chaining import build_prompt_chain_graph

        model = DeterministicFakeChatModel(
            text_responses=["outline", "draft", "final"]
        )

        result = build_prompt_chain_graph(model).invoke({"topic": "LangGraph"})

        self.assertEqual(result["final"], "final")

    def test_parallelization_combines_all_answers(self) -> None:
        from c05_03_parallelization import build_parallel_graph

        model = DeterministicFakeChatModel(text_responses=["A", "B"])

        result = build_parallel_graph(model).invoke({"question": "如何学习？"})

        self.assertEqual(set(result["answers"]), {"A", "B"})

    def test_routing_uses_structured_decision(self) -> None:
        from c05_04_routing import RouteSelection, build_routing_graph

        model = DeterministicFakeChatModel(
            structured_responses={
                RouteSelection: RouteSelection(route="math"),
            }
        )

        result = build_routing_graph(model).invoke({"question": "2+2"})

        self.assertEqual(result["route"], "math")

    def test_orchestrator_worker_collects_drafts(self) -> None:
        from c05_05_orchestrator_worker import (
            Plan,
            build_orchestrator_graph,
        )

        planner = DeterministicFakeChatModel(
            structured_responses={
                Plan: Plan(subtasks=["背景", "示例"]),
            }
        )
        worker = DeterministicFakeChatModel(text_responses=["draft-a", "draft-b"])

        result = build_orchestrator_graph(planner, worker).invoke(
            {"topic": "Agent"}
        )

        self.assertEqual(set(result["drafts"]), {"draft-a", "draft-b"})

    def test_evaluator_optimizer_stops_on_pass(self) -> None:
        from c05_06_evaluator_optimizer import build_evaluator_graph

        generator = DeterministicFakeChatModel(
            text_responses=["draft-1", "draft-2"]
        )
        evaluator = DeterministicFakeChatModel(
            text_responses=["REVISE", "PASS"]
        )

        result = build_evaluator_graph(generator, evaluator).invoke(
            {"task": "写一句介绍", "rounds": 0}
        )

        self.assertEqual(result["draft"], "draft-2")
        self.assertEqual(result["rounds"], 2)

    def test_react_agent_executes_tool_and_returns_final_answer(self) -> None:
        from c05_07_react_tool_agent import build_react_agent

        model = DeterministicFakeChatModel(
            tool_responses=[
                {
                    "name": "calculate_shipping",
                    "args": {"weight_kg": 2, "distance_km": 10},
                }
            ],
            text_responses=["运费为 30 元。"],
        )

        result = build_react_agent(model).invoke(
            {"messages": [{"role": "user", "content": "计算运费"}]}
        )

        self.assertEqual(result["messages"][-1].content, "运费为 30 元。")


if __name__ == "__main__":
    unittest.main()
