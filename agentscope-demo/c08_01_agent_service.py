# -*- coding: utf-8 -*-
"""
最小 AgentScope Agent Service。

使用 SQLite 作为存储、InMemoryMessageBus 作为消息总线、
LocalWorkspaceManager 管理工作区，不需要本地 Redis / Docker。

运行:
    python agentscope-demo/c08_01_agent_service.py [--port 8000]

服务启动后可用 c08_02_service_smoke.py 走通完整 API 流程。
"""

import argparse
import os
from pathlib import Path

import uvicorn

from agentscope.app import create_app
from agentscope.app.message_bus import InMemoryMessageBus
from agentscope.app.storage import AsyncSQLAlchemyStorage
from agentscope.app.workspace_manager import LocalWorkspaceManager
from agentscope.credential import DeepSeekCredential


MODULE_DIR = Path(__file__).resolve().parent
SERVICE_DIR = Path(
    os.environ.get("AGENTSCOPE_DEMO_SERVICE_DIR", str(MODULE_DIR))
)


def build_app():
    storage = AsyncSQLAlchemyStorage(
        f"sqlite+aiosqlite:///{SERVICE_DIR / 'agent_service.db'}",
        create_tables=True,
        auto_migrate=False,
    )
    return create_app(
        storage=storage,
        message_bus=InMemoryMessageBus(),
        workspace_manager=LocalWorkspaceManager(
            basedir=str(SERVICE_DIR / "service_workspaces"),
        ),
        extra_credentials=[DeepSeekCredential],
        enable_scheduler=False,
        title="AgentScope DeepSeek Learning Service",
    )


app = build_app()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(
        "c08_01_agent_service:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
