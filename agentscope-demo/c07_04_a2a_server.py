# -*- coding: utf-8 -*-
"""
把 AgentScope + DeepSeek Agent 暴露为 A2A 1.0 Server。

先启动本脚本，再运行 c07_05_a2a_client.py。

运行:
    python agentscope-demo/c07_04_a2a_server.py [--port 9999]
"""

import argparse
import asyncio

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Part,
    Task,
    TaskState,
    TaskStatus,
)
from starlette.applications import Starlette

from agentscope.agent import Agent
from agentscope.event import TextBlockDeltaEvent
from agentscope.message import UserMsg
from agentscope.tool import Toolkit

from common import get_model


class AgentScopeExecutor(AgentExecutor):
    """One AgentScope agent per A2A context_id."""

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def _new_agent(self) -> Agent:
        return Agent(
            name="Friday",
            system_prompt="You are a concise Chinese assistant named Friday.",
            model=get_model(stream=True),
            toolkit=Toolkit(),
        )

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        if context.task_id is None or context.context_id is None:
            raise RuntimeError("A2A server did not assign IDs.")

        agent = self._agents.setdefault(context.context_id, self._new_agent())
        await event_queue.enqueue_event(
            Task(
                id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
            )
        )
        updater = TaskUpdater(
            event_queue,
            task_id=context.task_id,
            context_id=context.context_id,
        )
        await updater.start_work()

        artifact_id = f"{context.task_id}-reply"
        pending = None
        started = False
        async for event in agent.reply_stream(
            UserMsg(name="user", content=context.get_user_input()),
        ):
            if not isinstance(event, TextBlockDeltaEvent):
                continue
            if pending is not None:
                await updater.add_artifact(
                    [Part(text=pending)],
                    artifact_id=artifact_id,
                    append=started,
                    last_chunk=False,
                )
                started = True
            pending = event.delta

        if pending is not None:
            await updater.add_artifact(
                [Part(text=pending)],
                artifact_id=artifact_id,
                append=started,
                last_chunk=True,
            )
        await updater.complete()

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        if context.task_id is None or context.context_id is None:
            return
        await TaskUpdater(
            event_queue,
            task_id=context.task_id,
            context_id=context.context_id,
        ).cancel()


def create_app(base_url: str) -> Starlette:
    card = AgentCard(
        name="Friday",
        description="An AgentScope agent backed by DeepSeek over A2A 1.0.",
        version="1.0.0",
        supported_interfaces=[
            AgentInterface(
                url=base_url,
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            ),
        ],
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="chat",
                name="Chat",
                description="Answer questions in a multi-turn conversation.",
                tags=["chat"],
            ),
        ],
    )
    handler = DefaultRequestHandler(
        agent_executor=AgentScopeExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    return Starlette(
        routes=[
            *create_agent_card_routes(card),
            *create_jsonrpc_routes(handler, rpc_url="/"),
        ],
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    args = parser.parse_args()

    app = create_app(f"http://{args.host}:{args.port}")
    config = uvicorn.Config(app, host=args.host, port=args.port)
    await uvicorn.Server(config).serve()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nA2A server stopped.")
