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


if __name__ == "__main__":
    unittest.main()
