# -*- coding: utf-8 -*-
"""供离线示例和测试使用的确定性聊天模型。"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk


class DeterministicStructuredModel:
    """包装结构化输出响应，行为与 LangChain 的 Runnable 子集一致。"""

    def __init__(
        self,
        schema: type,
        responses: dict[type, object] | None = None,
    ) -> None:
        self.schema = schema
        self.responses = responses or {}

    def invoke(self, input: Any, config: Any = None, **kwargs: Any):
        response = self.responses.get(self.schema)
        if response is not None:
            return response

        try:
            return self.schema()
        except Exception:
            fields = getattr(self.schema, "model_fields", {})
            fallback = {
                name: (
                    [] if getattr(field, "default_factory", None) else None
                )
                for name, field in fields.items()
            }
            return fallback


class DeterministicFakeChatModel:
    """按预设顺序返回消息、工具调用和结构化输出。

    它不是一个完整的 ChatModel 实现，只覆盖教学示例需要的 Runnable 子集：
    ``invoke``、``stream``、``bind_tools`` 和 ``with_structured_output``。
    """

    def __init__(
        self,
        *,
        structured_responses: dict[type, object] | None = None,
        tool_responses: list[dict[str, Any]] | None = None,
        text_responses: list[str] | None = None,
    ) -> None:
        self.structured_responses = structured_responses or {}
        self.tool_responses = list(tool_responses or [])
        self.text_responses = list(text_responses or [])
        self.bound_tools: list[Any] = []
        self.calls: list[Any] = []

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        self.bound_tools = list(tools)
        return self

    def with_structured_output(self, schema: type, **kwargs: Any):
        return DeterministicStructuredModel(schema, self.structured_responses)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        self.calls.append(input)
        if self.tool_responses:
            tool_call = dict(self.tool_responses.pop(0))
            tool_call.setdefault("id", f"offline-call-{len(self.calls)}")
            tool_call.setdefault("type", "tool_call")
            return AIMessage(content="", tool_calls=[tool_call])

        if self.text_responses:
            content = self.text_responses.pop(0)
        else:
            content = "离线示例回答：已按确定性规则完成。"
        return AIMessage(content=content)

    async def ainvoke(
        self,
        input: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> AIMessage:
        return self.invoke(input, config=config, **kwargs)

    def stream(
        self,
        input: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunk]:
        message = self.invoke(input, config=config, **kwargs)
        content = str(message.content)
        for character in content:
            yield AIMessageChunk(content=character)

    async def astream(
        self,
        input: Any,
        config: Any = None,
        **kwargs: Any,
    ):
        for chunk in self.stream(input, config=config, **kwargs):
            yield chunk

