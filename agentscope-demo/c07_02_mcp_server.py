# -*- coding: utf-8 -*-
"""
本模块自带的 MCP STDIO Server，供 c07_03_mcp_tool.py 连接。

不会单独对外监听端口，而是由 MCPClient 通过标准输入输出启动本脚本。
"""

from datetime import datetime

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("agentscope-demo-tools")


@mcp.tool()
def current_time() -> str:
    """Return the current local time in ISO 8601 format."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


@mcp.tool()
def repeat_text(text: str, count: int = 2) -> str:
    """Repeat a text several times.

    Args:
        text: Text to repeat.
        count: How many times to repeat, between 1 and 5.
    """
    if not 1 <= count <= 5:
        return "count must be between 1 and 5"
    return " ".join([text] * count)


if __name__ == "__main__":
    mcp.run(transport="stdio")
