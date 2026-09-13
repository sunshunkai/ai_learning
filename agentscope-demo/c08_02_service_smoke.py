# -*- coding: utf-8 -*-
"""
Agent Service HTTP 冒烟测试。

启动 c08_01_agent_service.py 后运行本脚本，完整走通：
  健康检查 -> 创建 DeepSeek 凭据 -> 创建 Agent -> 创建 Session
  -> 触发 chat -> 轮询消息。

运行:
    python agentscope-demo/c08_02_service_smoke.py [--url http://127.0.0.1:8000]
"""

import argparse
import asyncio
import os
import time

import httpx

from agentscope.message import UserMsg

from common import DEFAULT_BASE_URL, DEFAULT_MODEL, require


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", default="learner-001")
    args = parser.parse_args()

    headers = {"X-User-ID": args.user_id}
    async with httpx.AsyncClient(
        base_url=args.url,
        headers=headers,
        timeout=30,
    ) as client:
        print("=== 1. Health ===")
        health = await client.get("/health")
        health.raise_for_status()
        print(health.json())

        print("\n=== 2. 创建 DeepSeek 凭据 ===")
        credential_response = await client.post(
            "/credential/",
            json={
                "data": {
                    "type": "deepseek_credential",
                    "api_key": require("DEEPSEEK_API_KEY"),
                    "base_url": DEFAULT_BASE_URL,
                }
            },
        )
        credential_response.raise_for_status()
        credential_id = credential_response.json()["credential_id"]
        print(f"credential_id={credential_id}")

        print("\n=== 3. 创建 Agent ===")
        agent_response = await client.post(
            "/agent/",
            json={
                "name": "DeepSeek Friday",
                "system_prompt": "你是简洁的中文助手 Friday。",
            },
        )
        agent_response.raise_for_status()
        agent_id = agent_response.json()["agent_id"]
        print(f"agent_id={agent_id}")

        print("\n=== 4. 创建 Session ===")
        session_response = await client.post(
            "/sessions/",
            json={
                "agent_id": agent_id,
                "chat_model_config": {
                    "type": "deepseek_credential",
                    "credential_id": credential_id,
                    "model": DEFAULT_MODEL,
                    "parameters": {
                        "thinking_enable": False,
                    },
                },
            },
        )
        session_response.raise_for_status()
        session_id = session_response.json()["session_id"]
        print(f"session_id={session_id}")

        print("\n=== 5. 触发 Chat ===")
        user_msg = UserMsg(
            name="user",
            content="请用一句话介绍 AgentScope Agent Service。",
        )
        chat_response = await client.post(
            "/chat/",
            json={
                "agent_id": agent_id,
                "session_id": session_id,
                "input": user_msg.model_dump(mode="json"),
            },
        )
        chat_response.raise_for_status()
        print(chat_response.json())

        print("\n=== 6. 轮询会话消息 ===")
        messages = []
        for attempt in range(20):
            message_response = await client.get(
                f"/sessions/{session_id}/messages",
                params={"agent_id": agent_id},
            )
            message_response.raise_for_status()
            payload = message_response.json()
            messages = payload.get("messages", [])
            if payload.get("is_running") is False and len(messages) >= 2:
                break
            await asyncio.sleep(1)

        print(f"最终消息数: {len(messages)}")
        for message in messages:
            name = message.get("name", message.get("role", ""))
            text = message.get("content")
            if isinstance(text, list):
                text = "\n".join(
                    block.get("text", "")
                    for block in text
                    if block.get("type") == "text"
                )
            print(f"\n[{name}] {text}")


if __name__ == "__main__":
    asyncio.run(main())
