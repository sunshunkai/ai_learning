from __future__ import annotations

import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()
