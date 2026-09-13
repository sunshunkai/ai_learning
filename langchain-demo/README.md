# LangChain 学习模块

本目录学习 LangChain 1.x，并通过 OpenAI 兼容接口接入 DeepSeek。

逐文件知识扫盲见 [`KNOWLEDGE_GUIDE.md`](./KNOWLEDGE_GUIDE.md)。

> 目录故意命名为 `langchain-demo/`，因为如果叫 `langchain/`，本地目录会
> 遮蔽 pip 安装的 `langchain` 包，导致 `import langchain` 导入错误。

## 安装与配置

```bash
venv/bin/pip install -r langchain-demo/requirements.txt
```

项目根目录 `.env` 需要配置：

```text
DEEPSEEK_API_KEY=你的DeepSeek Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

`common.py` 会统一读取配置并创建 `ChatOpenAI`，模型默认是 `deepseek-chat`。

## Demo 清单

### 基础

1. `quickstart.py`：invoke、stream、messages 三种基础调用
2. `01_chat_models.py`：SystemMessage/HumanMessage、batch、stream、usage
3. `02_prompt_templates.py`：ChatPromptTemplate、partial、MessagesPlaceholder
4. `03_lcel_chains.py`：LCEL、RunnablePassthrough、RunnableParallel
5. `04_output_parsers.py`：字符串、列表和 JSON 输出解析
6. `05_structured_output.py`：Pydantic 校验和 with_structured_output

### 记忆、工具和 Agent

7. `06_memory.py`：LangGraph checkpointer 按 thread_id 保存会话
8. `07_tool_calling.py`：bind_tools、tool_calls 和 ToolMessage 回传
9. `08_agent.py`：create_agent 自动执行多步工具循环
10. `09_rag.py`：切分、向量检索、上下文注入的 RAG 最小闭环
11. `10_async_stream.py`：ainvoke、astream、abatch 和异步链

### 可靠性、控制流和可观测性

12. `11_retry_fallback.py`：with_retry、with_fallbacks
13. `12_few_shot.py`：FewShotChatMessagePromptTemplate
14. `13_message_trimming.py`：trim_messages 控制上下文长度
15. `14_agent_hitl.py`：Agent 执行工具前暂停并等待人工批准
16. `15_stream_events.py`：astream_events 观察链和模型事件
17. `16_skill_loading.py`：Agent 搜索/读取/资源渐进加载，以及容器 Skill

## 运行方式

```bash
venv/bin/python langchain-demo/quickstart.py
venv/bin/python langchain-demo/01_chat_models.py
venv/bin/python langchain-demo/08_agent.py
venv/bin/python langchain-demo/14_agent_hitl.py
venv/bin/python langchain-demo/16_skill_loading.py --mode agent --conversation --inspect
venv/bin/python langchain-demo/16_skill_loading.py --mode agent --conversation
venv/bin/python langchain-demo/16_skill_loading.py --mode prompt
venv/bin/python langchain-demo/16_skill_loading.py --mode anthropic --inspect
```

Agent 人工审批案例支持自动测试：

```bash
CLAUDE_DEMO_APPROVAL=yes venv/bin/python langchain-demo/14_agent_hitl.py
CLAUDE_DEMO_APPROVAL=no  venv/bin/python langchain-demo/14_agent_hitl.py
```

## 版本说明

网上很多教程仍是 LangChain 0.x（LLMChain、旧 Memory API），本目录使用
LangChain 1.x 的 Runnable、LCEL、create_agent 和 LangGraph checkpointer。

DeepSeek 不提供 embedding API，因此 RAG 示例使用本地 HashEmbeddings 演示流程；
生产环境可替换为 OpenAI、HuggingFace 或 BGE 等 embedding 服务。

`16_skill_loading.py` 的 `agent` 模式使用 `search_skills`、`read_skill` 和
`read_skill_resource` 三个工具实现渐进加载，并按会话隔离状态；`prompt`
模式保留完整注入对照；`anthropic` 模式依赖 `langchain-anthropic` 的
`container.skills` 和代码执行工具，需使用支持这些能力的官方 Anthropic 端点。
