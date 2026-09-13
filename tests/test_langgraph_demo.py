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
    def test_verification_manifest_contains_53_examples(self) -> None:
        from verify_examples import all_examples

        self.assertEqual(len(all_examples()), 53)

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


class PersistenceMemoryTests(unittest.TestCase):
    def test_thread_ids_isolate_message_history(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        from c06_01_thread_checkpointer import build_thread_graph

        graph = build_thread_graph(InMemorySaver())
        first_config = {"configurable": {"thread_id": "first"}}
        second_config = {"configurable": {"thread_id": "second"}}

        graph.invoke(
            {"messages": [{"role": "user", "content": "one"}]},
            config=first_config,
        )
        first = graph.invoke(
            {"messages": [{"role": "user", "content": "two"}]},
            config=first_config,
        )
        second = graph.invoke(
            {"messages": [{"role": "user", "content": "other"}]},
            config=second_config,
        )

        self.assertEqual(len(first["messages"]), 2)
        self.assertEqual(len(second["messages"]), 1)

    def test_state_history_contains_checkpoints(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        from c06_02_state_history_replay import build_history_graph

        graph = build_history_graph(InMemorySaver())
        config = {"configurable": {"thread_id": "history"}}
        graph.invoke({"count": 0}, config=config)
        graph.invoke({"count": 10}, config=config)

        history = list(graph.get_state_history(config))

        self.assertGreaterEqual(len(history), 3)
        self.assertEqual(history[0].values["count"], 12)

    def test_time_travel_fork_changes_only_new_branch(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        from c06_03_fork_time_travel import (
            build_time_travel_graph,
            fork_before_write,
        )

        graph = build_time_travel_graph(InMemorySaver())
        config = {"configurable": {"thread_id": "fork-test"}}
        original = graph.invoke({"topic": "A"}, config=config)
        original_snapshot = graph.get_state(config)
        fork_config = fork_before_write(graph, config, "B")
        forked = graph.invoke(None, config=fork_config)
        current = graph.get_state(original_snapshot.config)

        self.assertEqual(original["result"], "A-draft")
        self.assertEqual(forked["result"], "B-draft")
        self.assertEqual(current.values["result"], "A-draft")

    def test_store_namespaces_isolate_users(self) -> None:
        from c06_05_store_long_term_memory import build_memory_store

        store = build_memory_store()
        store.put(("user", "alice", "memories"), "food", {"text": "喜欢面食"})
        store.put(("user", "bob", "memories"), "food", {"text": "喜欢米饭"})

        alice = store.get(("user", "alice", "memories"), "food")
        bob = store.get(("user", "bob", "memories"), "food")

        self.assertEqual(alice.value["text"], "喜欢面食")
        self.assertEqual(bob.value["text"], "喜欢米饭")

    def test_memory_management_removes_old_messages(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        from c06_06_memory_management import (
            build_memory_graph,
            trim_oldest_messages,
        )

        graph = build_memory_graph(InMemorySaver())
        config = {"configurable": {"thread_id": "trim"}}
        graph.invoke(
            {
                "messages": [
                    {"role": "user", "content": "1"},
                    {"role": "assistant", "content": "2"},
                    {"role": "user", "content": "3"},
                ]
            },
            config=config,
        )

        trim_oldest_messages(graph, config, keep=1)

        self.assertEqual(len(graph.get_state(config).values["messages"]), 1)


class HumanInTheLoopReliabilityTests(unittest.TestCase):
    def test_interrupt_resumes_with_human_value(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.types import Command

        from c07_01_interrupt_resume import build_interrupt_graph

        graph = build_interrupt_graph(InMemorySaver())
        config = {"configurable": {"thread_id": "interrupt"}}
        first = graph.invoke({"value": "start"}, config=config)
        resumed = graph.invoke(Command(resume="Alice"), config=config)

        self.assertIn("__interrupt__", first)
        self.assertEqual(resumed["value"], "Alice")

    def test_approve_and_reject_choose_different_branches(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.types import Command

        from c07_02_approve_reject import build_approval_graph

        approved_graph = build_approval_graph(InMemorySaver())
        approved_config = {"configurable": {"thread_id": "approved"}}
        approved_graph.invoke({"action": "部署"}, approved_config)
        approved = approved_graph.invoke(
            Command(resume=True),
            approved_config,
        )

        rejected_graph = build_approval_graph(InMemorySaver())
        rejected_config = {"configurable": {"thread_id": "rejected"}}
        rejected_graph.invoke({"action": "部署"}, rejected_config)
        rejected = rejected_graph.invoke(
            Command(resume=False),
            rejected_config,
        )

        self.assertEqual(approved["result"], "approved:部署")
        self.assertEqual(rejected["result"], "rejected:部署")

    def test_review_edit_uses_modified_payload(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.types import Command

        from c07_03_review_edit_state import build_review_graph

        graph = build_review_graph(InMemorySaver())
        config = {"configurable": {"thread_id": "review"}}
        graph.invoke({"draft": "旧内容"}, config=config)
        result = graph.invoke(
            Command(resume="修改后的内容"),
            config=config,
        )

        self.assertEqual(result["draft"], "修改后的内容")

    def test_retry_graph_recovers(self) -> None:
        from c07_05_retries_timeouts_errors import build_retry_graph

        result = build_retry_graph().invoke({"value": 2})

        self.assertEqual(result["value"], 4)
        self.assertEqual(result["attempts"], 2)

    def test_error_handler_recovers(self) -> None:
        from c07_05_retries_timeouts_errors import (
            build_error_handler_graph,
        )

        result = build_error_handler_graph().invoke({"status": "running"})

        self.assertEqual(result["status"], "recovered")

    def test_graceful_drain_pauses_and_resumes(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        from c07_06_graceful_drain import run_drain_demo

        paused, resumed = run_drain_demo(InMemorySaver())

        self.assertEqual(paused, 1)
        self.assertEqual(resumed["count"], 2)


class StreamingObservabilityTests(unittest.TestCase):
    def test_values_and_updates_streams(self) -> None:
        from c08_01_stream_values_updates import (
            build_stream_graph,
            collect_modes,
        )

        parts = collect_modes(build_stream_graph(), {"value": 0})
        types = {part["type"] for part in parts}

        self.assertIn("values", types)
        self.assertIn("updates", types)

    def test_message_stream_contains_ai_content(self) -> None:
        from c08_02_stream_messages import build_message_stream_graph

        model = DeterministicFakeChatModel(text_responses=["hello stream"])
        graph = build_message_stream_graph(model)
        contents = [
            part["data"][0].content
            for part in graph.stream(
                {"messages": [{"role": "user", "content": "hi"}]},
                stream_mode="messages",
                version="v2",
            )
        ]

        self.assertIn("hello stream", contents)

    def test_custom_stream_receives_progress(self) -> None:
        from c08_03_custom_tasks_checkpoints import build_custom_graph

        events = list(
            build_custom_graph().stream(
                {"value": 0},
                stream_mode="custom",
                version="v2",
            )
        )

        self.assertEqual(events[0]["data"]["progress"], 50)

    def test_multiple_modes_include_subgraph_custom_event(self) -> None:
        from c08_04_multiple_modes_subgraphs import build_parent_graph

        parts = list(
            build_parent_graph().stream(
                {"text": "hello"},
                stream_mode=["updates", "custom"],
                version="v2",
            )
        )

        self.assertTrue(any(part["type"] == "custom" for part in parts))
        self.assertTrue(any(part["type"] == "updates" for part in parts))

    def test_event_stream_contains_lifecycle_events(self) -> None:
        from c08_05_event_streaming import (
            build_event_graph,
            collect_event_methods,
        )

        methods = collect_event_methods(build_event_graph())

        self.assertIn("values", methods)


class SubgraphMultiAgentTests(unittest.TestCase):
    def test_compiled_subgraph_runs_as_parent_node(self) -> None:
        from c09_01_subgraph_as_node import build_parent_graph

        result = build_parent_graph().invoke({"text": "hello"})

        self.assertEqual(result["result"], "HELLO")

    def test_subgraph_inside_node_translates_state(self) -> None:
        from c09_02_subgraph_in_node import build_parent_graph

        result = build_parent_graph().invoke({"text": "hello"})

        self.assertEqual(result["child_output"], "HELLO")
        self.assertNotIn("private_child_note", result)

    def test_subgraph_stream_exposes_child_progress(self) -> None:
        from c09_03_subgraph_persistence_stream import build_parent_graph

        parts = list(
            build_parent_graph().stream(
                {"text": "hello"},
                config={"configurable": {"thread_id": "subgraph-stream-test"}},
                stream_mode="custom",
                subgraphs=True,
                version="v2",
            )
        )

        self.assertTrue(any(part.get("ns") for part in parts))

    def test_supervisor_routes_to_selected_worker(self) -> None:
        from c09_04_multi_agent_patterns import (
            WorkerChoice,
            build_supervisor_graph,
        )

        model = DeterministicFakeChatModel(
            structured_responses={
                WorkerChoice: WorkerChoice(worker="researcher"),
            }
        )

        result = build_supervisor_graph(model).invoke({"topic": "LangGraph"})

        self.assertEqual(result["worker"], "researcher")
        self.assertEqual(result["result"], "研究结果：LangGraph")


class ApplicationCapstoneTests(unittest.TestCase):
    def test_partial_execution_resumes_after_first_node(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver

        from c10_01_testing_partial_execution import build_pipeline_graph

        graph = build_pipeline_graph(InMemorySaver())
        config = {"configurable": {"thread_id": "partial"}}
        paused = graph.invoke(
            {"value": 1},
            config=config,
            interrupt_after=["step_one"],
        )
        resumed = graph.invoke(None, config=config)

        self.assertEqual(paused["value"], 2)
        self.assertEqual(resumed["value"], 4)

    def test_rag_retrieval_finds_relevant_document(self) -> None:
        from c10_02_agentic_rag import retrieve_documents
        from fixtures.documents import load_documents

        results = retrieve_documents("state graph reducers", load_documents())

        self.assertEqual(results[0]["id"], "langgraph-state")

    def test_sql_guard_rejects_write_statement(self) -> None:
        from c10_03_sql_agent_hitl import validate_readonly_sql

        with self.assertRaises(ValueError):
            validate_readonly_sql("DELETE FROM orders")
        self.assertEqual(
            validate_readonly_sql("SELECT * FROM orders"),
            "SELECT * FROM orders",
        )

    def test_capstone_runs_tool_and_human_approval(self) -> None:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.types import Command

        from c10_05_capstone_research_assistant import (
            RouteDecision,
            build_capstone_graph,
        )

        model = DeterministicFakeChatModel(
            structured_responses={
                RouteDecision: RouteDecision(route="research"),
            },
            text_responses=["最终研究摘要：LangGraph 适合有状态编排。"],
        )
        graph = build_capstone_graph(model, InMemorySaver())
        config = {"configurable": {"thread_id": "capstone-test"}}
        paused = graph.invoke({"question": "介绍 LangGraph"}, config=config)
        final = graph.invoke(Command(resume=True), config=config)

        self.assertIn("__interrupt__", paused)
        self.assertTrue(final["approved"])
        self.assertIn("最终研究摘要", final["answer"])


if __name__ == "__main__":
    unittest.main()
