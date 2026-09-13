# LangChain Demo 逐文件知识扫盲

这份文档面向刚开始学习 LangChain 的读者，按仓库中的 Python 文件逐个解释：

- 这个文件在整条学习路线中解决什么问题
- 代码中出现了哪些核心概念和 API
- 一次完整调用是怎样流转的
- 哪些地方最容易产生误解
- 学完后应该能够回答哪些问题

当前目录使用 LangChain 1.x。阅读网上文章时要注意，很多教程仍然使用
`LLMChain`、旧版 Memory 等 LangChain 0.x API，不能直接照搬。

## 一、先统一几个大概念

### 1. Chat Model

聊天模型接收一组消息，返回一条 AI 消息。它不是简单地把字符串传进去，
而是在模拟“系统、用户、助手”之间的对话。

常见消息角色：

- `SystemMessage`：系统规则、角色设定、输出要求
- `HumanMessage`：用户输入
- `AIMessage`：模型历史回复，也可能携带工具调用
- `ToolMessage`：应用程序执行工具后回传给模型的结果

### 2. Prompt Template

Prompt Template 是可复用、可参数化的消息模板。它负责把用户变量安全地
插入消息结构，而不是靠 Python 字符串随意拼接。

### 3. Runnable 与 LCEL

LangChain 1.x 中，许多组件都实现为 Runnable。Runnable 通常支持：

- `invoke()`：调用一次
- `batch()`：批量调用
- `stream()`：流式输出
- `ainvoke()`、`abatch()`、`astream()`：异步版本

LCEL 使用 `|` 把 Runnable 连起来：

```python
chain = prompt | model | parser
```

它表达的是一条数据流水线，前一个组件的输出会成为后一个组件的输入。

### 4. Output Parser

模型本质上输出文本或消息。Output Parser 负责把输出转换成程序更容易使用的
字符串、列表、字典或 Pydantic 对象。

### 5. Tool Calling

工具调用不是让模型直接执行 Python 函数。模型只负责生成“想调用哪个工具、
参数是什么”的结构化请求，真正执行工具的是应用程序。

### 6. Agent

Agent 在 Tool Calling 之上增加了循环控制：

1. 模型判断是否需要调用工具
2. 程序执行工具
3. 工具结果回传给模型
4. 模型继续判断或输出最终答案

### 7. RAG

RAG 是 Retrieval-Augmented Generation，检索增强生成。典型流程是：

```text
文档 -> 切分 -> 向量化 -> 存入向量库 -> 检索相关片段
     -> 把片段放进 Prompt -> 模型基于资料回答
```

RAG 主要解决模型不知道私有知识、知识过时以及无法引用资料的问题。

### 8. Checkpointer 与 thread_id

LangGraph Checkpointer 保存 Agent 或对话图的状态。`thread_id` 是会话编号：

- 同一个 `thread_id` 共享历史
- 不同 `thread_id` 相互隔离
- `InMemorySaver` 只保存在当前 Python 进程内存中

## 二、基础设施文件

### common.py

文件链接：`common.py`

### 文件定位

这是所有 Demo 的公共模型工厂。其他脚本通常只调用：

```python
from common import build_model
```

因此其他示例不需要重复处理 API Key、Base URL 和模型配置。

### 涉及的知识点

1. 环境变量与配置管理

   `tools.load_env.load()` 从项目根目录 `.env` 读取配置。必须了解：

   - `.env` 不应该提交真实密钥
   - 已存在的系统环境变量默认不会被 `.env` 覆盖
   - API Key 属于敏感信息，不能写死在代码里

2. OpenAI 兼容接口

   DeepSeek 提供兼容 OpenAI Chat Completions 风格的接口，因此可以通过
   `langchain_openai.ChatOpenAI` 接入。

   这里的 `ChatOpenAI` 是 LangChain 的模型适配器，并不代表后端一定是
   OpenAI 官方服务。真正访问哪个服务由 `base_url` 决定。

3. 模型工厂模式

   `build_model()` 把模型创建逻辑集中在一个函数中，统一设置：

   - `model`：默认 `deepseek-chat`
   - `api_key`：从环境变量读取
   - `base_url`：DeepSeek API 地址
   - `temperature`：生成随机性
   - `**kwargs`：允许调用者追加模型参数

4. temperature

   `temperature` 越低，输出通常越稳定；越高，输出更多样但更不稳定。

   大致可以这样理解：

   - `0`：抽取、分类、结构化输出、工具选择
   - `0.2 - 0.4`：普通问答、教学解释
   - `0.7 - 1.0`：创意写作、头脑风暴

   它不是严格确定性的开关，也不会改变模型能力上限。

5. 导入路径处理

   `PROJECT_ROOT` 被加入 `sys.path`，这样从项目根目录或子目录运行时，
   都能稳定导入 `tools.load_env`。

### 学完后应该知道

- 为什么业务代码通常不应该到处创建 `ChatOpenAI`
- `base_url` 和官方 OpenAI 服务之间的关系
- 为什么 API Key 要放进环境变量
- 温度参数应该如何按任务选择

### quickstart.py

文件链接：`quickstart.py`

### 文件定位

这是最小可运行入口，用来建立“LangChain 调用聊天模型”的第一印象。

### 三种调用方式

1. `invoke()`：一次性调用

```python
answer = model.invoke("问题")
```

调用返回 `AIMessage`，通常通过 `answer.content` 取得文本。

适合：

- 后端同步请求
- 不需要持续显示生成过程
- 逻辑简单的问答

2. `stream()`：流式调用

```python
for chunk in model.stream("问题"):
    print(chunk.content, end="")
```

模型每生成一小段就返回一个 chunk。用户体验上能更早看到内容，但不代表
总生成时间一定更短，也不代表服务费用更低。

适合：

- 聊天界面
- 长文本生成
- 需要降低“首字等待时间”的场景

3. 消息列表调用

```python
messages = [
    SystemMessage("你是简短回答问题的助手。"),
    HumanMessage("1+1 等于几？"),
]
```

这里要理解：

- 消息列表代表当前一次请求的上下文
- `SystemMessage` 是行为约束，不是用户问题
- 模型本身并不会自动永久记住上一轮聊天

### 学完后应该知道

- `invoke`、`stream` 的使用场景
- 为什么模型输入常常是一组消息而不是单个字符串
- `AIMessage.content` 代表什么

## 三、基础组件

### 01_chat_models.py

文件链接：`01_chat_models.py`

### 文件定位

系统学习 Chat Model 的三种执行模式和返回信息。

### 知识点一：消息上下文

示例传入：

```python
[
    SystemMessage("你是 Java 转 AI 学习的导师，回复保持简洁。"),
    HumanMessage("为什么 LLM 每次调用都要传入完整历史消息？"),
]
```

聊天模型本身是无状态的。一次 HTTP 请求通常只认识这次请求中携带的内容。
所谓“记住历史”，本质上是应用每次把相关历史重新放进请求。

后面 `06_memory.py` 和 `13_message_trimming.py` 会继续解决历史如何保存、
如何裁剪的问题。

### 知识点二：invoke

`invoke()` 是最基础的同步单次调用：

```python
response = model.invoke(messages)
```

返回的对象不仅包含文本，还可能包含：

- `content`：模型回答
- `usage_metadata`：Token 使用统计
- `response_metadata`：供应商返回的元信息
- `tool_calls`：结构化工具请求

### 知识点三：batch

```python
answers = model.batch([...])
```

Batch 把多个独立输入一起提交给 Runnable。它表达的是批量接口，具体并发方式
取决于实现。适合批量分类、批量摘要、离线数据处理。

不要误解为“把多个问题拼成同一个上下文”。在示例中，每个问题相互独立。

### 知识点四：stream

每个 chunk 也是 `AIMessageChunk`。流式输出时要考虑：

- 某些 chunk 的 `content` 可能为空
- 工具调用参数也可能分片到达
- 前端应逐块追加显示，不要假设每个 chunk 都对应完整句子

### 知识点五：usage_metadata

Token 统计可以帮助你：

- 估算费用
- 判断上下文是否过长
- 判断输入和输出哪部分占主要成本
- 为日志和监控提供指标

### 学完后应该知道

- 无状态模型如何实现多轮对话
- `invoke`、`batch`、`stream` 的区别
- Token 用量从哪里读取

### 02_prompt_templates.py

文件链接：`02_prompt_templates.py`

### 文件定位

学习如何用模板稳定地组织系统提示、用户变量和历史消息。

### 知识点一：ChatPromptTemplate

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是资深 AI 讲师，回答要简洁。"),
    ("human", "请解释：{concept}"),
])
```

`{concept}` 是模板变量。调用 `format_messages()` 后才会变成真正的消息列表。

模板的价值：

- 把职责、输出要求和变量分开
- 避免不稳定、难维护的字符串拼接
- 同一模板可复用于不同输入
- 可以嵌入 Chain

### 知识点二：先格式化，不调用模型

```python
formatted = prompt.format_messages(concept="prompt engineering")
```

这是调试 Prompt 的常用技巧。先看真正发送给模型的消息内容，再决定是否调用
模型，可以节省调试成本。

### 知识点三：prompt | model

```python
chain = prompt | model
```

输入是 `{"concept": "..."}`，Prompt 将它转换成消息列表，模型再接收消息列表。

这里已经出现了 LCEL 的基本思想：每个组件负责一个转换步骤。

### 知识点四：partial

```python
prompt.partial(style="中文文言文")
```

`partial()` 预先填好部分变量，生成偏函数式的 Prompt。后续调用只需要传剩余
变量。

适合：

- 固定的角色或格式要求
- 根据用户配置创建专用 Prompt
- 多租户场景中的固定模板参数

### 知识点五：MessagesPlaceholder

普通变量只能插入字符串。需要插入一整套 Human/AI 历史消息时，使用：

```python
MessagesPlaceholder(variable_name="history")
```

传入的 `history` 必须是消息对象列表，例如 `HumanMessage` 和 `AIMessage`。

注意：`MessagesPlaceholder` 只负责把历史放进 Prompt，本身不负责保存历史。

### 学完后应该知道

- Prompt 模板和模型调用的边界
- 如何调试格式化后的消息
- partial 与历史占位符分别解决什么问题

### 03_lcel_chains.py

文件链接：`03_lcel_chains.py`

### 文件定位

掌握 LangChain 1.x 最重要的组合语法 LCEL。

### 知识点一：管道组合

```python
joke_chain = joke_prompt | model | parser
```

数据流向：

```text
dict -> ChatPromptTemplate -> ChatModel -> StrOutputParser -> str
```

每个组件都遵循 Runnable 接口，因此可以像积木一样组合。

### 知识点二：StrOutputParser 的作用

模型返回 `AIMessage`，而后续业务代码通常希望直接得到字符串。
`StrOutputParser` 负责取消息的文本内容。

如果不需要字符串，也可以保留 `AIMessage`，从而继续读取 Token 信息或工具调用。

### 知识点三：RunnablePassthrough

```python
{"topic": RunnablePassthrough()}
```

它表示把当前输入原样透传。

示例中，chain 收到字符串 `"加班"`，`RunnablePassthrough()` 将其作为
`topic` 传给 Prompt，因此没写 `invoke({"topic": "加班"})` 也能运行。

常见组合是：

```python
{
    "context": retriever | format_docs,
    "question": RunnablePassthrough(),
}
```

### 知识点四：RunnableParallel

```python
parallel_chain = RunnableParallel(
    joke=joke_chain,
    summary=summary_chain,
)
```

同一份输入被分配给多个分支，最终返回：

```python
{
    "joke": "...",
    "summary": "...",
}
```

需要区分“并行分支”和“同一个模型内部的 batch”。前者是链层面的编排，
后者是 Runnable 的批量调用能力。

### 知识点五：链也是 Runnable

组合出来的 chain 仍然支持 `invoke`、`batch`、`stream` 和对应异步方法。
所以链可以继续嵌套进新的链。

### 学完后应该知道

- 每个 `|` 前后的数据类型是什么
- `AIMessage` 为什么有时需要转换为字符串
- `RunnablePassthrough` 和 `RunnableParallel` 的区别
- 如何调试一条 LCEL 链

### 04_output_parsers.py

文件链接：`04_output_parsers.py`

### 文件定位

学习把模型输出转换成 Python 常用结构。

### 知识点一：StrOutputParser

将 `AIMessage` 转成 `str`。它不负责校验内容正确性，也不保证回答符合业务规则。

### 知识点二：CommaSeparatedListOutputParser

该 Parser 会把模型返回的逗号分隔文本转换成 Python 列表。

关键点是：

```python
list_parser.get_format_instructions()
```

它生成一段格式说明，并注入系统 Prompt。模型收到“应该按什么格式回答”的
要求后，Parser 才有更高概率成功解析。

常见风险：

- 模型没有严格遵守格式
- 项目名称本身包含逗号
- 输出中带编号、解释或 Markdown

生产代码还需要异常处理、重试或结构化输出。

### 知识点三：JsonOutputParser

它把模型输出的 JSON 文本解析成 Python `dict`。

使用时要明确：

- 模型可能输出 Markdown 代码块
- JSON 字段可能缺失或类型错误
- `JsonOutputParser` 主要是解析，不等同于完整业务校验

示例通过“只输出 JSON”来降低格式违规概率，但真正关键场景应配合 Pydantic。

### 学完后应该知道

- Parser 的输入和输出类型
- `format_instructions` 为什么重要
- 解析成功和业务数据合法不是一回事

### 05_structured_output.py

文件链接：`05_structured_output.py`

### 文件定位

解决“模型输出必须是稳定结构化数据”的问题。

### 知识点一：Pydantic BaseModel

```python
class Movie(BaseModel):
    title: str
    director: str
    year: int
    genres: list[str]
```

Pydantic 负责：

- 字段定义
- 类型检查
- 必填校验
- 默认值与约束
- JSON 序列化和反序列化

`Field(description=...)` 的说明会进入 JSON Schema，能够帮助模型理解字段含义。

### 知识点二：JSON Schema

```python
Movie.model_json_schema()
```

JSON Schema 是描述 JSON 数据结构的标准。它告诉模型应该生成哪些字段、字段
是什么类型、有哪些约束。

### 知识点三：JsonOutputParser + model_validate

这条路线分两步：

1. Parser 把模型文本解析成字典
2. `Movie.model_validate(raw)` 再做 Pydantic 校验并产生对象

优点是控制清晰；缺点是格式提示和校验是分开的。

### 知识点四：with_structured_output

```python
structured_model = model.with_structured_output(Movie, method="json_mode")
```

它把模型包装成“输入普通请求，输出 Pydantic 对象”的 Runnable。

常见实现机制包括：

- 原生结构化输出
- JSON mode
- 工具调用或函数调用

不同模型提供商支持能力不同，因此这里显式指定了 `json_mode`。切换模型时
需要验证该模式是否受支持。

### 知识点五：什么时候使用

适合：

- 信息抽取
- 表单填写
- 分类结果
- 路由决策
- 需要进入数据库或业务代码的对象

不适合：

- 自由创作
- 开放式解释
- 结构本身频繁变化的内容

### 学完后应该知道

- Pydantic 模型和 JSON Schema 的关系
- Parser 与 Validator 的区别
- `with_structured_output` 为什么能减少手工处理

## 四、记忆、工具和 Agent

### 06_memory.py

文件链接：`06_memory.py`

### 文件定位

理解 LangChain 1.x 中对话记忆的新实现方式。

### 知识点一：Agent 也是有状态的图

`create_agent()` 创建的 Agent 可以使用 Checkpointer 保存状态。虽然示例没有
业务工具，但它仍然可以用来演示多轮消息保存。

### 知识点二：InMemorySaver

`InMemorySaver` 把状态保存在内存中：

- 运行速度快，适合学习
- Python 进程退出后数据丢失
- 不能用于多进程共享
- 不能替代生产数据库

生产环境常见选择包括 SQLite、PostgreSQL 等持久化 Checkpointer。

### 知识点三：thread_id

```python
config = {"configurable": {"thread_id": "alice"}}
```

`thread_id` 相当于会话主键。第一次调用时指定 `alice`，第二次仍指定 `alice`，
Agent 就能读取此前保存的消息。

`bob` 使用另一个 `thread_id`，所以看不到 Alice 的历史。

### 知识点四：消息会不断累积

历史变长会带来：

- Token 成本上升
- 延迟增加
- 超出模型上下文上限
- 无关内容干扰答案

这也是 `13_message_trimming.py` 存在的原因。

### 学完后应该知道

- Checkpointer 保存什么
- `thread_id` 如何隔离用户
- 为什么内存记忆不能直接用于生产
- 为什么需要上下文裁剪或摘要

### 07_tool_calling.py

文件链接：`07_tool_calling.py`

### 文件定位

手动拆解 Tool Calling 的完整协议，理解 Agent 内部到底发生了什么。

### 知识点一：定义工具

```python
@tool
def get_weather(city: str) -> str:
    """查询一个城市的当前天气。"""
```

`@tool` 会读取：

- 函数名：工具名
- 类型标注：参数 Schema
- docstring：工具用途说明

函数类型标注和 docstring 不是装饰，它们会参与工具描述，影响模型能否正确选工具。

### 知识点二：绑定工具

```python
model = build_model().bind_tools([get_weather])
```

`bind_tools()` 把可用工具定义放进模型请求上下文。它不会执行工具，只是告诉
模型“你可以请求这些工具”。

### 知识点三：模型返回 tool_calls

模型第一次返回的可能是：

```python
AIMessage(
    content="",
    tool_calls=[
        {
            "name": "get_weather",
            "args": {"city": "北京"},
            "id": "call_xxx",
        }
    ],
)
```

核心认知：

- 模型只提出请求
- 参数来自模型生成，必须视为不可信输入
- 真实应用要做权限校验、参数校验和审计
- 不应让模型生成任意 Python 代码后直接执行

### 知识点四：应用程序执行工具

```python
result = get_weather.invoke(tool_args)
```

工具真正执行发生在应用进程。

### 知识点五：ToolMessage 回传

```python
ToolMessage(content=result, tool_call_id=call["id"])
```

`tool_call_id` 必须和模型请求中的 ID 对应，模型才能知道每个结果对应哪个调用。

最后再次调用模型，模型基于工具结果生成最终回答。

### 知识点六：手动循环 vs Agent

本文件只是手动处理一次工具调用。真实任务可能需要多轮：

```text
模型 -> 工具 -> 模型 -> 另一个工具 -> 模型 -> 最终答案
```

`08_agent.py` 会使用 `create_agent()` 自动管理这个循环。

### 学完后应该知道

- 模型和应用程序在工具调用中的职责边界
- `bind_tools`、`tool_calls`、`ToolMessage` 的关系
- 为什么工具参数必须校验
- Agent 为什么可以看成自动工具循环

### 08_agent.py

文件链接：`08_agent.py`

### 文件定位

从手工 Tool Calling 进入自动 Agent。

### 知识点一：create_agent

```python
agent = create_agent(
    model,
    tools=[get_weather, multiply],
    system_prompt="...",
)
```

`create_agent()` 负责构建 LangGraph 状态图，并自动处理：

- 调用模型
- 检查 `tool_calls`
- 执行工具
- 生成 ToolMessage
- 继续调用模型
- 直到模型给出最终回答

### 知识点二：Agent 输入输出

输入：

```python
{"messages": [{"role": "user", "content": "..."}]}
```

输出：

```python
{"messages": [HumanMessage, AIMessage, ToolMessage, AIMessage, ...]}
```

最后一条消息通常是最终答案，但完整消息轨迹对调试非常重要。

### 知识点三：一个请求可以调用多个工具

示例问题同时要求查天气和计算乘法。模型可能：

1. 先请求天气工具
2. 执行并回传
3. 再请求乘法工具
4. 汇总最终答案

也可能一次返回多个 `tool_calls`。控制流由 Agent 图管理。

### 知识点四：Agent 的关键约束

- 工具描述要清楚
- 工具权限要最小化
- 工具参数必须验证
- 要有最大步骤数或终止条件
- 要记录每一步用于审计
- 高风险工具应增加人工审批

### 学完后应该知道

- Agent 和普通 Chain 的区别
- Agent 为什么需要循环
- 如何查看 Agent 的完整执行轨迹
- 为什么 Agent 不等于可以随意执行任何代码

### 09_rag.py

文件链接：`09_rag.py`

### 文件定位

用一个小型中文知识库串起 RAG 的全部核心步骤。

### 知识点一：Document

```python
Document(page_content=content, metadata={"source": "demo"})
```

`Document` 包含：

- `page_content`：正文
- `metadata`：来源、标题、页码、权限等附加信息

生产 RAG 中 metadata 很重要，可用于引用、过滤和权限控制。

### 知识点二：文本切分

```python
RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
)
```

切分的原因：

- 模型上下文有限
- 检索颗粒度需要适中
- 太长会引入无关信息
- 太短会丢失上下文

`chunk_size` 是块大小，`chunk_overlap` 是相邻块的重叠内容。
递归切分器会优先按段落、换行、句子、字符等层级尝试。

常见调参问题：

- 块太大：召回内容不精确
- 块太小：一个完整语义被切碎
- overlap 太大：重复和成本增加

### 知识点三：Embedding

Embedding 把文本映射成向量，语义相近的文本通常距离更近。

示例中的 `HashEmbeddings` 只是本地确定性演示：

- 它按空格切词
- 用 MD5 得到哈希下标
- 生成稀疏计数向量并归一化

它不真正理解语义，中文没有空格时效果尤其有限。生产环境应替换为：

- OpenAI Embeddings
- HuggingFace Embeddings
- BGE 等中文 Embedding 模型

### 知识点四：Vector Store 与 Retriever

```python
vectorstore = InMemoryVectorStore.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
```

Vector Store 负责保存向量和原文，Retriever 负责按查询返回相关 Document。

`k=2` 表示最多召回两个 chunk。真实系统可能还要使用：

- 相似度阈值
- 关键词检索与向量检索混合
- rerank 重排序
- metadata 过滤

### 知识点五：上下文注入

```python
{
    "context": retriever | format_docs,
    "question": RunnablePassthrough(),
}
```

同一问题被用于两个分支：

- 一路负责检索并拼接上下文
- 一路原样作为用户问题

最终 Prompt 要求“只根据资料回答，资料中没有的信息不要编造”。

注意：Prompt 约束可以降低幻觉概率，但不能从机制上彻底消除幻觉。

### RAG 的完整链路

```text
原始文档
  -> 清洗
  -> 切分
  -> Embedding
  -> 写入向量库
  -> 查询 Embedding
  -> 相似度检索
  -> 可选重排序
  -> 上下文拼接
  -> LLM 生成
  -> 引用与评估
```

示例只实现最小闭环，不包含数据清洗、权限、缓存、评估和重排序。

### 学完后应该知道

- RAG 为什么需要切分和 Embedding
- Retriever 返回的是文档片段，不是最终答案
- 上下文如何进入 Prompt
- HashEmbeddings 为什么只适合演示

## 五、异步与工程可靠性

### 10_async_stream.py

文件链接：`10_async_stream.py`

### 文件定位

学习 Runnable 的异步接口，为 Web 服务和高并发场景打基础。

### 知识点一：同步和异步

同步：

```python
result = model.invoke("...")
```

异步：

```python
result = await model.ainvoke("...")
```

异步不是天然更快。它主要避免等待网络 I/O 时阻塞事件循环，提高服务并发效率。

### 知识点二：三种异步接口

- `ainvoke()`：异步单次调用
- `astream()`：异步流式调用
- `abatch()`：异步批量调用

`astream()` 必须使用：

```python
async for chunk in model.astream("..."):
    ...
```

### 知识点三：异步 Chain

同步链：

```python
chain.invoke({...})
```

异步链：

```python
await chain.ainvoke({...})
```

组合语法没有变化，调用方式变为异步。

### 知识点四：asyncio.run

```python
asyncio.run(main())
```

它创建事件循环并运行顶层异步函数。在 FastAPI 等已有事件循环的框架中，
不应在请求处理函数里再次调用 `asyncio.run()`。

### 学完后应该知道

- 异步主要用于 I/O 并发
- 同步接口和异步接口的对应关系
- 流式异步遍历如何写
- 什么时候不应该使用 `asyncio.run()`

### 11_retry_fallback.py

文件链接：`11_retry_fallback.py`

### 文件定位

学习链和模型层面的容错策略。

### 知识点一：RunnableLambda

```python
RunnableLambda(flaky)
```

它把普通 Python 函数包装成 Runnable，从而能够使用 `invoke`、`with_retry`、
`with_fallbacks` 等统一能力。

### 知识点二：with_retry

```python
runnable.with_retry(
    retry_if_exception_type=(TemporaryError,),
    stop_after_attempt=3,
    wait_exponential_jitter=False,
)
```

重试适合：

- 网络超时
- 供应商限流
- 临时 5xx
- 瞬时连接失败

重试不适合：

- API Key 错误
- 请求参数非法
- 余额不足
- 明确的权限错误

这些错误通常重试也没有意义，还可能放大问题。

### 知识点三：指数退避与抖动

指数退避会让等待时间逐渐增加，例如 1 秒、2 秒、4 秒。抖动会加入随机性，
避免大量客户端固定在同一时刻再次请求。

需要关注：

- 最大重试次数
- 单次超时
- 总超时
- 是否允许对非幂等操作重试

### 知识点四：with_fallbacks

```python
chain = primary.with_fallbacks([fallback])
```

当主链路抛出异常时，LangChain 会尝试备用 Runnable。

常见备用方案：

- 主模型到备用模型
- 远程模型到本地模型
- 复杂链到简化链
- 实时生成到缓存结果

要注意 fallback 的输入输出类型应兼容，否则链的下一步可能无法处理。

### 学完后应该知道

- 哪些异常适合重试
- 重试如何避免请求风暴
- fallback 和 retry 的区别
- 为什么生产系统还需要超时、熔断和限流

## 六、Prompt 与上下文控制

### 12_few_shot.py

文件链接：`12_few_shot.py`

### 文件定位

通过示例教模型输出模式。

### 知识点一：Zero-shot、Few-shot

Zero-shot：

```text
判断这条需求是低、中还是高复杂度。
```

Few-shot：

```text
输入：改一行文案
输出：复杂度：低

输入：增加支付方式和退款流程
输出：复杂度：高
```

Few-shot 的重点不是“喂更多知识”，而是通过示例定义边界、格式和判断标准。

### 知识点二：example_prompt

```python
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])
```

它定义每个示例如何变成消息。

### 知识点三：FewShotChatMessagePromptTemplate

它会把 examples 按照 example_prompt 展开，并插入最终 Prompt。

最终消息顺序通常是：

```text
SystemMessage
HumanMessage(example 1)
AIMessage(example 1)
HumanMessage(example 2)
AIMessage(example 2)
HumanMessage(真实输入)
```

### 知识点四：示例设计原则

- 示例要覆盖关键类别和边界
- 输入输出格式必须一致
- 不要用互相矛盾的示例
- 标签定义要清楚
- 示例过多会增加 Token 成本

更高级的写法是根据当前输入动态选择示例，例如语义相似度示例选择器。

### 学完后应该知道

- Few-shot 与 Prompt Engineering 的关系
- 示例在消息列表中的位置
- 为什么示例质量比数量更重要

### 13_message_trimming.py

文件链接：`13_message_trimming.py`

### 文件定位

控制每轮请求携带的历史上下文长度。

### 知识点一：为什么需要裁剪

上下文越长：

- Token 越多，成本越高
- 首字延迟通常越大
- 越容易触达模型上下文上限
- 旧消息可能干扰当前问题

### 知识点二：trim_messages

关键参数：

- `max_tokens=60`：目标 Token 上限
- `token_counter="approximate"`：使用近似计数
- `strategy="last"`：优先保留最后的消息
- `include_system=True`：保留系统提示
- `start_on=HumanMessage`：裁剪后从人类消息开始
- `end_on=HumanMessage`：裁剪后以人类消息结束

### 知识点三：为什么保持消息边界

聊天消息通常要求合理角色顺序。裁剪如果从一条 AIMessage 开始，或留下没有
对应 HumanMessage 的 ToolMessage，可能导致 Provider 拒绝请求，或者语义
不完整。

### 知识点四：裁剪不等于摘要

`trim_messages()` 主要做删除和保留。它不会把旧内容压缩成摘要。

更完整的记忆策略通常包括：

1. 保留最近若干轮消息
2. 保留系统规则和关键事实
3. 对远期历史做摘要
4. 对话记忆与 RAG 长期记忆分开
5. 只保留与当前任务相关的信息

### 知识点五：token_counter

近似计数速度快，但不等于供应商真实分词器结果。精确控制时应传入与模型匹配
的计数函数或分词器。

### 学完后应该知道

- 上下文窗口与 Token 成本的关系
- 消息裁剪和会话摘要的区别
- 为什么系统消息通常要保留
- 如何保证裁剪后的消息结构合法

## 七、Agent 控制与可观测性

### 14_agent_hitl.py

文件链接：`14_agent_hitl.py`

### 文件定位

展示 Human-in-the-loop，即 Agent 执行高风险工具前需要人工批准。

### 知识点一：敏感工具

`send_email` 会改变外部世界，因此属于有副作用的工具。类似的还有：

- 发起支付
- 删除数据库记录
- 修改生产配置
- 发布内容
- 调用内部管理系统

这类工具不应让 Agent 无条件自动执行。

### 知识点二：interrupt_before

```python
interrupt_before=["tools"]
```

Agent 图运行到工具节点前暂停，并保存当前状态。此时模型已经提出
`tool_calls`，但工具还没有真正执行。

### 知识点三：人工审批

示例通过：

- 环境变量 `CLAUDE_DEMO_APPROVAL` 自动测试
- TTY 环境下使用 `input()` 询问用户

真实系统还需要：

- 展示工具名和完整参数
- 展示风险和影响范围
- 身份认证和授权
- 审批记录
- 超时和取消机制
- 防止用户误批准

### 知识点四：恢复执行

批准后：

```python
final_state = agent.invoke(None, config=config)
```

传入 `None` 表示不添加新输入，从同一 `thread_id` 的 Checkpoint 恢复。

config 必须一致，否则会进入另一条不存在的会话状态。

### 知识点五：拒绝执行

示例拒绝后直接返回，并保持暂停。生产实现可以选择：

- 永久取消本次任务
- 回传一个“用户拒绝”的工具结果让模型继续
- 修改参数后重新审批
- 切换成人工处理

### 学完后应该知道

- HITL 解决什么风险
- `interrupt_before` 在哪个时刻暂停
- Checkpointer 为什么是恢复执行的前提
- 审批系统不能只做一个 Yes/No 按钮

### 15_stream_events.py

文件链接：`15_stream_events.py`

### 文件定位

从“只观察文本”升级到“观察整条链的运行事件”。

### 知识点一：普通 stream 的局限

`model.stream()` 主要关心模型生成文本。复杂链中我们还想知道：

- Prompt 何时完成
- 模型何时开始
- 哪个工具开始和结束
- Retriever 花了多久
- 哪个节点报错
- 链一共执行了哪些步骤

### 知识点二：astream_events

```python
async for event in chain.astream_events(input, version="v2"):
    ...
```

每个 event 通常包含：

- `event`：事件名称
- `name`：组件名称
- `run_id`：本次运行 ID
- `parent_ids`：父子调用关系
- `data`：事件数据
- `metadata`：附加信息

### 知识点三：常见事件

- `on_chat_model_start`：模型开始
- `on_chat_model_stream`：模型产生文本 chunk
- `on_chat_model_end`：模型结束
- `on_chain_start`：链或组件开始
- `on_chain_end`：链或组件结束
- `on_tool_start`：工具开始
- `on_tool_end`：工具结束
- `on_retriever_start` / `on_retriever_end`：检索开始和结束
- `on_chain_error`：执行失败

### 知识点四：可观测性

事件流可以用于：

- 自定义流式前端
- 展示“正在检索”“正在调用工具”等状态
- 统计每个节点的耗时
- 记录 Agent 决策轨迹
- 接入 LangSmith、OpenTelemetry 等监控系统

### 学完后应该知道

- 文本流和事件流的区别
- 为什么事件流适合 Agent 调试
- 父子事件关系如何帮助定位调用链
- 如何从事件中提取耗时和运行状态

## 八、推荐学习顺序

### 第一遍：先跑通

```text
quickstart.py
common.py
01_chat_models.py
02_prompt_templates.py
03_lcel_chains.py
```

目标是知道“输入怎样变成模型调用，再怎样变成输出”。

### 第二遍：掌握格式和工具

```text
04_output_parsers.py
05_structured_output.py
07_tool_calling.py
08_agent.py
```

目标是知道“模型输出怎样进入程序”和“程序怎样把工具结果交还给模型”。

### 第三遍：掌握记忆和 RAG

```text
06_memory.py
09_rag.py
13_message_trimming.py
```

目标是知道“模型应该看到哪些历史和哪些外部知识”。

### 第四遍：掌握工程能力

```text
10_async_stream.py
11_retry_fallback.py
12_few_shot.py
14_agent_hitl.py
15_stream_events.py
```

目标是知道“系统失败怎么办、上下文太长怎么办、高风险工具怎么办、
线上运行怎样观测”。

## 九、学完之后的自检问题

能够清楚回答下面的问题，说明基础知识已经形成框架：

1. 模型为什么是无状态的，多轮对话到底是谁在保存历史？
2. Prompt Template、Message、Model 和 Parser 的数据类型分别是什么？
3. LCEL 中每个 `|` 代表的转换关系是什么？
4. Parser 解析成功为什么不代表数据一定符合业务要求？
5. Tool Calling 中模型和程序各自负责什么？
6. Agent 为什么需要循环，什么时候应该停止？
7. `thread_id`、Checkpointer 和消息历史是什么关系？
8. RAG 中切块大小、overlap 和召回数量会怎样影响答案？
9. 同步、异步、流式分别解决什么工程问题？
10. retry 和 fallback 分别处理哪类失败？
11. Few-shot 为什么能提高格式稳定性？
12. trim_messages 和摘要记忆有什么区别？
13. Human-in-the-loop 为什么必须配合持久化状态？
14. `stream` 和 `astream_events` 观察的内容有什么不同？
15. 哪些 Demo 行为可以直接用于生产，哪些只是教学模拟？

## 十、必须注意的教学模拟

下面这些实现适合学习，不要不加改造直接用于生产：

1. `HashEmbeddings` 不理解语义，生产应使用正式 Embedding 模型。
2. `InMemorySaver` 会随进程退出丢失，生产应使用持久化 Checkpointer。
3. 天气工具返回 Mock 数据，不是真实天气。
4. 邮件工具没有真正发送邮件，也没有重试、幂等和退信处理。
5. 示例 Prompt 使用简单字符串，生产需要版本管理和注入防护。
6. 示例没有实现完整鉴权、限流、超时、审计和成本控制。
7. 工具参数由模型生成，进入真实业务前必须做 Pydantic 或业务层校验。

## 十一、Skill 加载专题

### 1. 文件定位

`16_skill_loading.py` 补上了本模块此前缺失的 Skill 示例。它复用
`tools/skill_loader.py` 和根目录 `skills/`，演示三种
加载方式：

- `agent`：`search_skills` 搜索 + `read_skill` 正文 + `read_skill_resource`
  资源三级渐进加载
- `prompt`：完整 SKILL.md 注入 SystemMessage
- `anthropic`：`ChatAnthropic.container.skills` 原生托管加载

Skill 和普通 Tool 的主要区别是：Skill 先提供“何时使用、按什么步骤做”的
指令，再由模型调用已有工具完成操作；它不一定本身就是一个可执行函数。

### 2. Agent 模式：目录注入和懒加载

数据流：

```text
discover_skills()
  -> RuleBasedSkillRouter 处理显式名称和 trigger
  -> 规则未命中时 LangChainSkillRouter 调用结构化模型
  -> SkillSessionManager 按 session_id 保存已加载状态
  -> render() 只注入 name/description 和已加载正文
  -> create_agent(tools=[search_skills, read_skill, read_skill_resource])
  -> Agent 先搜索候选
  -> Agent 判断任务相关后调用 read_skill(name)
  -> 工具返回完整 SKILL.md
  -> 只有需要时才调用 read_skill_resource
  -> Agent 按指令继续执行
```

`read_skill` 工具的定义是：

```python
@tool
def read_skill(name: str) -> str:
    """读取一个已加载 Skill 的完整 SKILL.md 指令。"""
    return find_skill(skills, name).markdown
```

这种设计接近 Agent Skill 的常见语义：

1. 先让模型看到低成本目录，判断是否相关。
2. 只有需要时才读取完整指令和引用资源。
3. 每个会话独立记录已加载 Skill，避免跨会话污染。

但这不是自动的“框架级 Skill 管理”。LangChain 本身并不要求 `SKILL.md`，
这里是通过 Agent system prompt 和工具组合出来的通用加载方式。

该模式默认仍使用 DeepSeek：

```bash
venv/bin/python langchain-demo/16_skill_loading.py --mode agent
venv/bin/python langchain-demo/16_skill_loading.py --mode agent --inspect
```

### 2.1 规则兜底与主 Agent 工具选择

`agent` 模式包含两个不同层次的模型参与方式：

1. `--router hybrid` 在进入 Agent 前执行一次预路由。`RuleBasedSkillRouter`
   先处理 `@skill:name` 和 trigger；只有规则未命中时，
   `LangChainSkillRouter` 才通过 `with_structured_output()` 让模型返回
   `skill_names` 和 `reason`。
2. 主 Agent 仍拥有 `search_skills`、`read_skill` 和
   `read_skill_resource` 工具，可以在执行任务时继续发现和加载 Skill。

预路由模型只看到候选 Skill 的 name 和 description，不读取完整
`SKILL.md`。模型返回的名称会再次与候选集合比对，随后交给
`SkillSession.load()` 执行权限、去重和上下文预算检查。

```bash
venv/bin/python langchain-demo/16_skill_loading.py \
  --mode agent --router hybrid --conversation
venv/bin/python langchain-demo/16_skill_loading.py \
  --mode agent --router rule --conversation
```

`--router rule` 延迟最低、行为可重复，但无法理解 trigger 之外的语义表达。
`--router hybrid` 在规则未命中时增加一次模型调用，换取更强的语义泛化能力。

### 3. Prompt 模式：完整 Skill 直接进 SystemMessage

```text
render_full_context([skill])
  -> SystemMessage
  -> [SystemMessage, HumanMessage]
  -> ChatOpenAI.invoke()
```

它没有工具调用循环，适合：

- 验证模型是否能按一份简单流程输出
- 只有一个 Skill 且上下文预算足够
- 对比“先目录后读取”和“一次性全量注入”的效果

代价是每个 Skill 的完整 Markdown 都会占用输入 Token。Skill 数量多时，
应该优先使用 Agent 模式的懒加载。

```bash
venv/bin/python langchain-demo/16_skill_loading.py --mode prompt
venv/bin/python langchain-demo/16_skill_loading.py --mode prompt --inspect
```

### 4. Anthropic 模式：`ChatAnthropic.container.skills`

LangChain 的 `langchain-anthropic` 封装了 Anthropic Messages API 的 Container
能力。核心写法是：

```python
model = ChatAnthropic(
    model="claude-opus-5",
    container={
        "skills": [
            {
                "type": "anthropic",
                "skill_id": "pptx",
                "version": "latest",
            }
        ]
    },
)

bound_model = model.bind_tools(
    [
        {
            "type": "code_execution_20260521",
            "name": "code_execution",
        }
    ]
)
```

要点：

- `container.skills` 指定本次容器要加载哪些托管 Skill。
- `type=anthropic` 表示 Anthropic 内置 Skill；`type=custom` 表示用户上传的
  自定义 Skill，此时 `skill_id` 是 Skills API 返回的 ID。
- Skill 加载本身不是普通本地工具，必须和代码执行工具一起发送。
- 模型回复的 `response_metadata["container"]` 可以查看容器 ID、过期时间和
  实际加载的 Skill。

离线检查：

```bash
venv/bin/python langchain-demo/16_skill_loading.py \
  --mode anthropic --skill-id pptx --inspect
```

真实调用需要支持 Skills/Container 的官方 Anthropic 端点：

```bash
venv/bin/python langchain-demo/16_skill_loading.py \
  --mode anthropic --skill-id pptx \
  --prompt "请创建一页标题为 AI Learning 的 PPT 并总结结果。"
```

自定义 Skill 可传：

```bash
venv/bin/python langchain-demo/16_skill_loading.py \
  --mode anthropic --skill-type custom --skill-id skill_xxx
```

### 5. `tools/skill_loader.py` 的职责

共享加载器负责本地 Skill 的基础设施：

- `discover_skills()`：递归发现 `SKILL.md`
- `find_skill()`：按 name 精确选择
- `render_catalog()`：生成低成本目录
- `render_full_context()`：生成完整 system prompt
- 解析最简 frontmatter，不依赖 LangChain 或 PyYAML

因此 Claude SDK 和 LangChain 两个模块可以复用同一份 `text-reversal`
Skill；AgentScope 目前使用它自己的临时 Skill 演示工作区加载。真正不同的
是“怎样把 Skill 交给各自的运行时”。

### 6. 三种模式对比

| 模式 | 模型侧 | 加载时机 | 是否需要 Anthropic 容器 | 适合 |
|---|---|---|---|---|
| agent | DeepSeek | 按需读取 | 否 | 多 Skill、上下文敏感 |
| prompt | DeepSeek | 每次全量 | 否 | 单 Skill、流程简单 |
| anthropic | Claude | 服务端加载 | 是 | 文档生成、代码执行等 |

### 7. 学完后应该知道

1. Skill 目录和完整 SKILL.md 分别承担什么职责？
2. `read_skill` 工具为什么能降低上下文成本？
3. Prompt 全量注入在什么情况下仍然更合适？
4. LangChain 原生是否理解 `SKILL.md`，本示例如何补齐？
5. `container.skills` 和 `bind_tools(code_execution)` 为什么通常成对出现？
6. `type=anthropic` 与 `type=custom` 的 `skill_id` 来源有什么不同？
7. `--inspect` 为什么适合先在 DeepSeek 配置下学习原生 Claude 模式？

### 8. 生产注意事项

1. Skill 正文来自文件，可能与用户 Prompt 混在一起，要注意指令注入。
2. `read_skill` 需要校验 name，并对未找到的 Skill 返回结构化错误。
3. 不要把所有 Skill 全量注入，应该按任务相关性选择。
4. 自定义 Skill 上传后会留在 Anthropic 工作区，必须有生命周期管理。
5. Container 代码执行可能产生文件、网络请求和费用，需要权限和审计。
6. `code_execution_*` 工具类型会演进，部署前应锁定已核对的 SDK/API 版本。
