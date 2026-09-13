# Claude SDK 逐文件知识扫盲

这份文档面向刚开始学习 Anthropic Claude SDK 的读者，按 `claude-sdk`
目录中的 Python 文件逐个解释：

- 文件在整条学习路线中的位置
- 代码调用的核心 API 和数据结构
- 一次请求从输入到输出的完整流程
- Anthropic 官方能力和兼容端点的差异
- 常见误区、生产风险和学后自检问题

本目录使用 Anthropic Python SDK 1.x。当前本地验证版本为 `anthropic 1.5.0`。
不同版本、不同中转服务的参数支持可能发生变化，运行前应以实际 SDK 签名和端点
文档为准。

## 一、先统一几个大概念

### 1. Messages API

Claude SDK 最核心的接口是：

```python
client.messages.create(...)
```

主要输入通常包括：

- `model`：模型名
- `max_tokens`：本次最多生成多少 Token
- `messages`：user / assistant 对话历史
- `system`：系统提示
- `tools`：可用工具定义
- `tool_choice`：工具选择策略
- `stream`：是否使用原始流式请求

返回值是 `Message` 对象，而不是单一的字符串。

### 2. Message 和 Content Block

一次响应大致包含：

```python
Message(
    id="msg_xxx",
    model="...",
    role="assistant",
    content=[
        TextBlock(...),
        ToolUseBlock(...),
        ThinkingBlock(...),
    ],
    stop_reason="end_turn",
    usage=Usage(...),
)
```

`response.content` 是一个列表，里面可以同时包含不同类型的内容块。最常见的是：

- `text`：普通文本回答
- `tool_use`：模型请求调用工具
- `thinking`：扩展思考内容

因此不能简单认为：

```python
response.content[0].text
```

永远都存在。正确做法通常是遍历 content，并根据 `block.type` 处理。

### 3. Claude 是无状态模型

Claude 不会自动记住上一次 HTTP 请求。所谓多轮对话，是应用每次把完整消息历史
重新发送给模型。

```python
messages = [
    {"role": "user", "content": "我叫小明"},
    {"role": "assistant", "content": "你好，小明"},
    {"role": "user", "content": "我叫什么？"},
]
```

历史保存在哪里、保存多久、怎么裁剪，都是应用层责任。

### 4. 同步、异步与流式

同步客户端：

```python
from anthropic import Anthropic

client = Anthropic()
response = client.messages.create(...)
```

异步客户端：

```python
from anthropic import AsyncAnthropic

client = AsyncAnthropic()
response = await client.messages.create(...)
```

流式输出：

```python
with client.messages.stream(...) as stream:
    for text in stream.text_stream:
        ...
```

三组概念不要混淆：

- 流式解决“何时把生成内容交给用户”
- 异步解决“等待网络 I/O 时是否阻塞事件循环”
- 批量或并发解决“多个独立请求如何一起执行”

### 5. Tool Use

工具调用不是 Claude 直接运行 Python 函数。流程是：

```text
应用声明 tools
  -> 模型返回 tool_use
  -> 应用校验并执行真实函数
  -> 应用回传 tool_result
  -> 模型基于结果继续回答
```

模型负责选择工具和生成参数，应用负责真正执行、鉴权、校验、审计和错误处理。

### 6. stop_reason

常见停止原因：

- `end_turn`：模型正常结束回答
- `tool_use`：模型请求调用工具
- `max_tokens`：达到最大输出 Token
- `stop_sequence`：命中自定义停止序列

只判断“有没有 text block”不足以完整控制流程，很多场景还要检查
`stop_reason` 和 `tool_use` block。

### 7. Usage

`response.usage` 常见字段：

- `input_tokens`
- `output_tokens`
- `cache_creation_input_tokens`
- `cache_read_input_tokens`

它用于成本统计、限流、缓存效果分析和容量规划。

### 8. 官方端点和兼容端点

官方 Anthropic SDK 可以读取：

```text
ANTHROPIC_API_KEY
ANTHROPIC_BASE_URL
```

当 `ANTHROPIC_BASE_URL` 指向第三方兼容服务时，SDK 请求格式相似，但能力不一定
完全一致。要特别注意：

- Thinking 是否支持
- tool_choice 是否支持具体工具名
- Prompt Caching 是否真正生效
- 模型名是否自动映射
- 额外参数是否被忽略或拒绝

## 二、基础设施文件

### __init__.py

### 文件定位

这是 Python 包标识文件，同时承担目录导航作用，列出了：

- 基础案例
- 进阶案例
- README 的位置

它没有业务逻辑，也不执行 API 调用。它的价值是让 `claude-sdk` 可以被当作包导入，
并给阅读者一个文件总览。

### 涉及的知识点

1. Python 包与模块

   当目录包含 `__init__.py` 时，它可以作为常规包导入：

   ```python
   from claude_sdk import common
   ```

   不过目录名包含连字符 `claude-sdk`，不能直接作为普通 Python 标识符导入。
   当前示例主要采用“直接运行脚本”的方式，因此脚本目录会进入 `sys.path`。

2. 模块文档字符串

   `__init__.py` 中的三引号内容属于模块 docstring，可以用于：

   - IDE 文档提示
   - 包级说明
   - 自动文档生成

3. 代码组织

   简单示例按主题分成多个独立脚本，而不是全部塞进一个文件。这种结构适合教学，
   但真实项目通常会进一步拆成 client、service、tools、schemas、workflow 等模块。

### 学完后应该知道

- Python 包标识文件的作用
- 教学内容按独立脚本组织的优缺点
- 为什么目录名最好不要使用连字符

### common.py

### 文件定位

这是所有示例的公共基础设施，主要提供：

- 项目根目录 `.env` 加载
- 默认模型名
- 同步 `Anthropic` 客户端工厂
- 异步 `AsyncAnthropic` 客户端工厂
- Message 文本提取
- 文本块打印

### 知识点一：项目根路径

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
```

脚本位置是：

```text
ai-learning/claude-sdk/common.py
```

因此 `parents[1]` 指向项目根目录 `ai-learning`。

将根目录加入 `sys.path` 后，可以稳定导入：

```python
from tools.load_env import load, require
```

### 知识点二：环境变量

配置来源优先级通常是：

```text
显式传参 > 系统环境变量 > 项目根 .env > SDK 默认值
```

`common.py` 中：

```python
result.setdefault("api_key", require("ANTHROPIC_API_KEY"))
```

意味着调用方显式传入 `api_key` 时优先使用调用方参数，否则读取环境变量。

Base URL 只有在配置后才显式传给客户端，否则 SDK 使用官方默认地址。

### 知识点三：客户端工厂

同步客户端：

```python
Anthropic(**kwargs)
```

异步客户端：

```python
AsyncAnthropic(**kwargs)
```

工厂模式让示例可以统一处理 Key、Base URL、超时和重试等配置，避免每个脚本重复。

### 知识点四：默认模型

```python
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
```

模型名集中在一个位置管理。切换官方模型或兼容端点时，不需要逐个修改脚本。

需要注意：

- 字符串默认值只代表当前示例的约定
- 官方账号不一定有权限访问该模型
- 第三方兼容端点可能把名称自动映射到自己的模型
- 模型名变更后应重新测试结构化输出、thinking 和 caching

### 知识点五：text_from_message

```python
return "".join(
    block.text
    for block in message.content
    if getattr(block, "type", None) == "text"
)
```

它会：

1. 遍历所有 content block
2. 只选择 `type == "text"`
3. 拼接所有文本

它不会处理：

- `tool_use`
- `thinking`
- 图片、引用或未来新增的内容块

如果响应只有工具调用而没有文本，返回值就是空字符串。

### 知识点六：客户端生命周期

同步和异步客户端内部维护 HTTP 连接池。生产服务通常应复用客户端，而不是每个
请求都创建一个。简单脚本用完即退出，影响较小。

### 学完后应该知道

- 配置优先级
- 同步和异步客户端如何创建
- content block 为什么不能直接假设为文本
- 生产环境为什么要复用客户端

### env_check.py

### 文件定位

这是运行其他示例前的环境自检，不会调用模型，因此不会消耗 API Token。

### 涉及的知识点

1. 依赖检查

   ```python
   import anthropic
   ```

   如果导入失败，说明依赖未安装或当前 Python 环境不对。

2. API Key 检查

   脚本只检查是否存在 `ANTHROPIC_API_KEY`，不会验证 Key 是否真实有效。真正的
   认证错误只能在请求阶段发现。

3. 密钥掩码

   ```python
   masked = api_key[:8] + "..." + api_key[-4:]
   ```

   即使做日志，也不应完整打印 API Key。生产日志还应避免把 Key 放进异常字符串、
   URL、请求头或进程参数中。

4. Base URL 检查

   未设置时使用官方 `https://api.anthropic.com`。使用中转服务时，端点地址和
   Key 必须匹配，不能把官方 Key 随意发给陌生代理。

5. fail fast

   环境错误应该在启动时尽快暴露，而不是在用户发起昂贵请求后才失败。

### 学完后应该知道

- 依赖、配置和认证是三件不同的事
- 自检脚本为什么有价值
- 为什么日志只能显示掩码
- 自定义 Base URL 的风险

## 三、基础对话

### basic_chat.py

### 文件定位

这是最小的一问一答，目标是认识 `messages.create()` 的输入和输出结构。

### 知识点一：消息角色

示例输入：

```python
messages=[
    {"role": "user", "content": "..."}
]
```

Messages API 中的对话历史通常使用：

- `user`
- `assistant`

系统提示在高阶用法中通过顶层 `system` 参数传入，而不是放进 `messages` 中的
`system` 角色。

### 知识点二：max_tokens

`max_tokens` 限制的是本次输出长度，不是输入加输出的总长度。

如果输出被截断，`stop_reason` 可能为 `max_tokens`。继续生成时，应用需要决定：

- 把已有内容作为 assistant 历史继续请求
- 提高输出上限
- 要求模型缩短回答

### 知识点三：content block

面向初学者的常见错误是：

```python
print(response.content)
```

这打印的是块的列表，不是最终文本。正确方式是遍历并筛选：

```python
for block in response.content:
    if block.type == "text":
        print(block.text)
```

### 知识点四：响应元信息

- `response.stop_reason`：停止原因
- `response.model`：实际使用的模型
- `response.usage.input_tokens`：输入 Token
- `response.usage.output_tokens`：输出 Token

这些字段对调试、成本控制和模型路由很重要。

### 学完后应该知道

- `messages.create()` 的最小参数
- `response.content` 为什么是列表
- `max_tokens` 限制什么
- 如何读取停止原因和 Token 用量

### multi_turn.py

### 文件定位

演示在没有服务端记忆的情况下，如何手动维护多轮上下文。

### 知识点一：消息历史必须成对演进

每次新问题前追加：

```python
messages.append({"role": "user", "content": user_input})
```

得到回复后还必须追加：

```python
messages.append({"role": "assistant", "content": reply})
```

只追加用户消息、不追加助手消息，模型虽然仍能回答，但下一轮看到的不是完整对话。

### 知识点二：历史增长

每轮都携带全部历史会导致：

- 输入 Token 持续增加
- 请求延迟增加
- 成本和上下文压力增加

生产应用通常需要：

- 保留最近 N 轮
- 对早期内容摘要
- 只保留与当前任务相关的内容
- 会话结束时归档，而不是无限增长

### 知识点三：回复内容格式

示例把文本回复拼成一个字符串作为 assistant 历史。这对纯文本对话没问题，但如果
模型返回 tool_use 或 thinking block，就不能只保存文本。

更通用的做法通常是保留完整的 content block，或使用专门的消息序列化策略。

### 当前工作区注意

当前未提交版本的 `multi_turn.py` 第 49 行写作：

```python
messages.apjei shipend({"role": "assistant", "content": reply})
```

这不是有效的 Python 语法。按文件原本逻辑，正确写法应为：

```python
messages.append({"role": "assistant", "content": reply})
```

本文按正确的 `append` 逻辑讲解。

### 学完后应该知道

- 多轮对话的状态由谁维护
- assistant 回复为什么要追加到历史
- 长历史会带来什么成本
- 工具调用和普通文本保存方式有何不同

### stream_chat.py

### 文件定位

演示最简单的流式文本输出，也就是常见的“打字机效果”。

### 知识点一：stream 上下文管理器

```python
with client.messages.stream(...) as stream:
    ...
```

进入上下文后，SDK 开始请求并处理流事件。离开上下文时，SDK 负责清理连接状态。

### 知识点二：text_stream

```python
for text in stream.text_stream:
    print(text, end="", flush=True)
```

`text_stream` 是 SDK 提供的高层文本流，只关注文本增量。它适合：

- 命令行聊天
- WebSocket 推送
- SSE/HTTP 流式响应
- 聊天产品的前端增量展示

它不展示完整的工具参数流、thinking delta 或所有原始事件。

### 知识点三：final message

```python
final = stream.get_final_message()
```

流结束后仍可取得完整 `Message`，用于读取：

- `stop_reason`
- `usage`
- 完整 content blocks

这就是“既有增量过程，又有最终完整对象”的模式。

### 知识点四：flush

`print(..., flush=True)` 让文本立即刷到终端。若不 flush，缓冲可能导致虽然已经
收到 chunk，但用户迟迟看不到。

### 学完后应该知道

- 流式输出和普通请求的区别
- `text_stream` 与完整 Message 的关系
- 为什么需要 flush
- 流式输出不等于总耗时一定更短

## 四、工具调用

### tool_use.py

### 文件定位

演示一次完整的单工具往返，是理解 Claude Tool Use 的基础。

### 知识点一：工具定义

```python
TOOLS = [
    {
        "name": "get_weather",
        "description": "查询某个中国城市的当前天气",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
            "required": ["city"],
        },
    }
]
```

`input_schema` 使用 JSON Schema 描述参数。工具描述要足够清楚，模型才能准确判断：

- 什么时候使用该工具
- 参数应该如何填写
- 每个字段代表什么

### 知识点二：模型返回 tool_use

```python
if block.type == "tool_use":
    tool_name = block.name
    tool_input = block.input
    tool_use_id = block.id
```

关键认知：

- 模型不会执行函数
- `block.input` 是模型生成的参数
- 参数必须视为不可信数据
- 真实系统要做类型、范围和权限校验

### 知识点三：assistant 消息

第一轮响应必须追加到消息历史：

```python
messages.append({"role": "assistant", "content": response.content})
```

不能只追加文本，因为模型这次返回的核心是 tool_use block。

### 知识点四：tool_result

工具执行结果按以下结构回传：

```python
{
    "type": "tool_result",
    "tool_use_id": tool_use_id,
    "content": result,
}
```

`tool_use_id` 用来把结果和模型发起的调用关联起来。ID 必须来自对应的 tool_use
block，不能在应用侧随意生成。

### 知识点五：为什么用 user 角色回传

tool_result 放在一条 user 消息中：

```python
messages.append({"role": "user", "content": tool_results})
```

这是 Messages API 的协议约定。它不代表“用户真的执行了工具”，而是表示应用正在
向模型提供工具执行结果。

### 知识点六：工具错误的正确表达

本示例只返回普通字符串。`multi_tool_loop.py` 和 `human_approval.py` 使用：

```python
"is_error": True
```

让模型明确知道工具失败，而不是把错误文本误当成有效业务结果。

### 学完后应该知道

- 工具声明和函数执行的区别
- tool_use_id 的作用
- assistant 消息为什么必须保留完整 content
- 工具错误为什么应显式标记

### multi_tool_loop.py

### 文件定位

把单次工具调用扩展成多工具、多轮循环，并加入异常和最大步骤限制。

### 知识点一：工具分发

```python
def execute_tool(name: str, args: dict):
    if name == "get_weather":
        return get_weather(**args)
    if name == "calculate":
        return calculate(**args)
    ...
```

这本质上是一个工具注册和分发表。真实项目通常使用字典注册：

```python
TOOL_REGISTRY = {
    "get_weather": get_weather,
    "calculate": calculate,
}
```

然后校验工具是否存在，再调用。应避免把模型提供的字符串直接用于动态导入、反射
或任意函数调用。

### 知识点二：工具参数校验

示例的 `calculate()` 手动检查：

- 运算符是否允许
- 除数是否为 0

这是业务层校验。模型生成的 JSON Schema 约束不能替代服务端验证。

生产环境建议使用 Pydantic 等工具对 `block.input` 做严格解析。

### 知识点三：多个 tool_use

同一个模型响应可能包含多个 `tool_use` block。循环收集所有工具结果后再回传，
比一次只处理一个更符合协议。

### 知识点四：异常回传

```python
except Exception as exc:
    result = f"工具执行失败: {exc}"
    is_error = True
```

这样模型可以：

- 向用户说明工具失败
- 换一个工具
- 修正参数后再次请求
- 停止并请求人工帮助

注意不要把完整堆栈、密钥或内部路径直接暴露给模型和终端用户。

### 知识点五：循环上限

```python
for step in range(1, 7):
    ...
```

Agent 可能陷入重复工具调用，因此必须有最大步骤、最大 Token、最大时间等终止条件。

### 知识点六：什么时候结束

当某一轮没有 tool_use：

```python
if not tool_results:
    print(text_from_message(response))
    return
```

表示模型已经给出最终文本回答。

### 学完后应该知道

- 如何安全分发多个工具
- 为什么要同时有 Schema 校验和业务校验
- 工具异常如何返回
- Agent 循环为什么不能没有上限

### human_approval.py

### 文件定位

演示高风险工具在真正执行前，如何加入人工审批。

### 知识点一：哪些工具需要审批

示例使用 `send_email`。类似的高风险操作还包括：

- 支付和退款
- 删除文件和数据库记录
- 部署到生产
- 修改权限
- 发送外部消息

判断标准不是“调用是否容易实现”，而是“执行错误的后果是否不可逆或高影响”。

### 知识点二：审批位置

安全边界应放在真实函数执行之前：

```text
模型提出 tool_use
  -> 展示工具名和参数
  -> 人工批准或拒绝
  -> 批准后 execute_email
```

不能在模型请求后立即执行，再把结果拿去让用户“审批”。

### 知识点三：审批信息

审批界面对应展示：

- 工具名
- 完整参数
- 将影响的对象和环境
- 风险等级
- 发起人和会话
- 是否可撤销

只显示“是否执行工具”会让审批人无法做有效判断。

### 知识点四：拒绝也是正常结果

拒绝时，代码仍然构造 tool_result：

```python
{
    "type": "tool_result",
    "tool_use_id": block.id,
    "content": "操作未执行：用户拒绝",
    "is_error": True,
}
```

这样模型知道工具没有执行，不会误以为邮件已经发出。

### 知识点五：环境变量自动决策

```python
override = os.environ.get("CLAUDE_DEMO_APPROVAL")
```

它用于自动化测试，不能替代真实审批系统。生产系统需要：

- 用户身份和权限校验
- 审批记录
- 超时和撤回
- 操作幂等性
- 审计日志
- 禁止通过普通环境变量绕过审批

### 学完后应该知道

- Human-in-the-loop 的核心边界在哪里
- 审批界面最少应该展示哪些信息
- 拒绝结果为什么要回传模型
- 自动审批只适合什么场景

### structured_output.py

### 文件定位

通过强制工具调用，让模型返回固定结构的 JSON，而不是自由文本。

### 知识点一：工具可以只用于提取

```python
EXTRACT_PROFILE_TOOL = {
    "name": "extract_profile",
    "description": "从自然语言中提取用户资料",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "city": {"type": "string"},
            "occupation": {"type": "string"},
            "skills": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["name", "city", "occupation", "skills"],
    },
}
```

它不需要真实执行函数。模型对工具的 `input` 就是希望得到的结构化对象。

### 知识点二：tool_choice

```python
tool_choice={"type": "any"}
```

表示必须从提供的工具中选一个调用。示例只有一个工具，因此效果等于“必须做资料
提取”。

其他常见策略还有：

- `auto`：模型自行决定
- `any`：必须调用某个工具
- 指定具体工具：必须调用指定工具

DeepSeek 兼容端点当前可能不支持指定具体工具名，因此文档示例使用 `any`。

### 知识点三：读取 tool_use.input

```python
if block.type == "tool_use":
    print(json.dumps(block.input, ensure_ascii=False, indent=2))
```

得到的是 Python 字典。进入业务系统前仍应做 Pydantic 或等价校验。

### 知识点四：结构化输出和工具调用

本质上，模型没有“直接返回 Python 对象”的魔法。常见实现是：

1. 把目标结构定义成工具 Schema
2. 强制模型调用工具
3. 从 tool_use.input 读取结构化参数
4. 在应用层再次校验

### 知识点五：限制

- 字段太多可能降低稳定性
- Enum 和 required 应尽量明确
- 不要用工具调用输出大段自由文本
- 兼容端点可能不完全支持 tool_choice
- 模型输出仍可能缺失语义或填写错误

### 学完后应该知道

- 如何用 tool_choice 约束输出
- 工具调用和结构化输出的关系
- 为什么仍要做应用层校验
- 兼容端点的 tool_choice 差异

## 五、并发、流事件和参数

### async_chat.py

### 文件定位

演示 `AsyncAnthropic` 的普通异步调用和异步流式输出。

### 知识点一：AsyncAnthropic

```python
client = get_async_client()
response = await client.messages.create(...)
```

异步客户端的方法返回 awaitable，必须在 `async` 函数中 `await`。

### 知识点二：异步流

```python
async with client.messages.stream(...) as stream:
    async for text in stream.text_stream:
        ...
```

异步流的三个关键词：

- `async with`：管理异步上下文资源
- `async for`：异步迭代
- `await`：等待异步调用完成

### 知识点三：异步不等于并发

单个 `await` 只是在等待当前 I/O，不会自动并行。要并发执行多个独立请求，还需要
`asyncio.gather()`、TaskGroup 等并发控制。

同时要设置并发上限，避免触发供应商限流。

### 知识点四：适用场景

- FastAPI 等异步 Web 框架
- WebSocket 服务
- 多个独立模型请求
- 高并发网络 I/O

CPU 密集任务不会因为改成 async 就变快。

### 学完后应该知道

- 同步客户端和异步客户端的对应方法
- 异步流上下文如何写
- 异步为什么主要解决 I/O 并发
- 并发还需要限流和保护

### stream_events.py

### 文件定位

从 `text_stream` 深入到底层原始流事件。

### 知识点一：为什么需要原始事件

文本流只给最终文本增量。原始事件还能观察：

- message 开始和结束
- content block 开始和结束
- text delta
- thinking delta
- tool input 的 JSON 分片
- usage 和 stop reason

### 知识点二：遍历 stream

```python
for event in stream:
    print(event.type)
```

每个事件都有自己的结构。示例重点处理：

- `content_block_start`
- `content_block_delta`

### 知识点三：content_block_start

```python
if event.type == "content_block_start":
    print(event.content_block.type)
```

这告诉应用正在开始什么类型的块，例如 text、tool_use 或 thinking。

### 知识点四：delta 类型

- `text_delta`：普通文本增量
- `thinking_delta`：思考内容增量
- `input_json_delta`：工具参数 JSON 的分片

工具参数可能不是一次完整到达，而是被拆成多个 JSON 片段，应用需要按协议累积。

### 知识点五：最终消息

流事件适合实时 UI，`get_final_message()` 适合最终状态落库。两者通常一起使用。

### 学完后应该知道

- 高层 text_stream 和底层事件的区别
- content block 如何增量到达
- tool input 为什么要分片累积
- 哪些信息只能从最终 Message 获取

### parameter_control.py

### 文件定位

集中演示 system、停止序列、metadata 和兼容端点的额外参数。

### 知识点一：system

```python
system="你是一名严谨的技术讲师。..."
```

system 是顶层参数，用来设置：

- 角色
- 行为准则
- 输出格式
- 安全边界

它也可以传文本块列表，以支持更复杂的块级控制和缓存控制。

### 知识点二：stop_sequences

```python
stop_sequences=["###"]
```

当模型生成 `###` 时提前停止。响应中可读取 `stop_sequence`。

需要注意：

- 停止序列是生成控制，不是格式化保证
- 模型可能本来就很少生成该字符串
- 命中后内容会截断，业务逻辑要能处理

### 知识点三：metadata

```python
metadata={"user_id": "demo-user-001"}
```

metadata 用于业务侧标记和追踪，不应放敏感信息。它不是要求模型理解的提示词。

### 知识点四：SDK 1.x 参数变化

当前本地 `anthropic 1.5.0` 的 `messages.create()` 签名中，顶层没有
`temperature` 和 `top_p`。示例通过：

```python
extra_body={"temperature": 0.3, "top_p": 0.9}
```

把参数透传给兼容端点。

这里要理解两个层面：

- SDK 声明的正式参数
- 兼容服务额外接受的请求体字段

`extra_body` 灵活性更高，但可移植性和类型安全更弱。切换端点时必须验证。

### 知识点五：temperature 和 top_p

- `temperature` 影响采样随机性
- `top_p` 影响候选概率范围

通常不应同时大幅调整两者，否则行为难以解释。结构化任务通常使用更低随机性，
创意任务可以使用更高随机性。

### 学完后应该知道

- system 为什么是顶层参数
- stop_sequences 的实际作用
- metadata 和 Prompt 的区别
- `extra_body` 适合什么情况、有什么风险

## 六、生产实践

### error_handling.py

### 文件定位

演示 SDK 客户端超时、自动重试和常见异常分类。

### 知识点一：客户端级配置

```python
client = get_client(timeout=30.0, max_retries=2)
```

- `timeout`：单次网络请求的超时时间
- `max_retries`：SDK 对部分临时错误自动重试

超时时间要考虑业务总超时、流式响应长度和重试次数，不能只看单次请求。

### 知识点二：AuthenticationError

通常表示：

- API Key 缺失
- Key 无效或已撤销
- Key 和 Base URL 不匹配
- 认证方案不正确

这类错误通常不应自动重试。

### 知识点三：RateLimitError

表示触发限流或额度限制。处理方式：

- 指数退避
- 降低并发
- 排队
- 切换备用模型或端点
- 检查账户配额

### 知识点四：APIStatusError

一般表示服务端返回了非成功 HTTP 状态。需要结合：

- `status_code`
- 响应正文
- 请求 ID
- 是否可重试

来处理。不要直接把完整响应正文和内部错误信息暴露给终端用户。

### 知识点五：APIConnectionError

通常涉及：

- 网络不可达
- DNS 失败
- 代理配置错误
- TLS 问题
- Base URL 错误

### 知识点六：异常顺序

更具体的异常应放在更通用的异常之前。Python 会匹配第一个命中的 `except`，
例如 `AuthenticationError` 应在 `APIError` 之前。

### 学完后应该知道

- 哪些错误适合重试
- timeout 和 max_retries 如何配合
- 如何区分认证、限流、状态和连接错误
- 为什么异常日志要脱敏

### thinking_chat.py

### 文件定位

演示扩展思考模式，以及如何区分 thinking block 和最终 text block。

### 知识点一：启用 thinking

```python
thinking={
    "type": "enabled",
    "budget_tokens": 1024,
}
```

`budget_tokens` 表示允许模型用于思考的预算。它和 `max_tokens` 不是同一个概念：

- `budget_tokens` 控制思考预算
- `max_tokens` 控制本次总输出上限

实际限制和兼容行为取决于官方服务或中转实现。

### 知识点二：thinking block

```python
if block.type == "thinking":
    print(block.thinking)
```

Thinking 内容不是最终答案。应用不应默认把它直接展示给终端用户，因为：

- 内容可能包含内部推理
- UI 体验可能不合适
- 产品策略和合规要求可能不允许

通常只展示最终 text block，thinking 用于调试、审计或受控展示。

### 知识点三：text block

最终答案仍来自：

```python
if block.type == "text":
    print(block.text)
```

不能把 thinking 当成最终回答，也不能假设 thinking 一定存在。

### 知识点四：兼容性

部分中转服务可能：

- 忽略 `budget_tokens`
- 不返回 thinking block
- 禁止 thinking 与某些 tool_choice 组合
- 使用不同模型映射

因此示例必须能同时处理“有 thinking”和“没有 thinking”的情况。

### 学完后应该知道

- Thinking 和普通回答的 block 类型差异
- budget_tokens 与 max_tokens 的区别
- 为什么不应无条件展示 thinking
- 兼容端点可能忽略哪些配置

### prompt_caching.py

### 文件定位

演示 Prompt Caching 的结构和缓存 Token 统计。

### 知识点一：什么是 Prompt Caching

当多个请求拥有相同的长前缀时，可以把该前缀标记为可缓存。后续请求如果命中缓存，
可减少重复处理成本并降低延迟。

稳定前缀通常包括：

- 长 system 提示
- 固定知识库内容
- 固定工具说明
- 不变的业务规则

### 知识点二：cache_control

```python
system=[
    {
        "type": "text",
        "text": LONG_SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]
```

`ephemeral` 表示临时缓存。缓存机制通常要求前缀稳定，并在有效期内重复使用。

### 知识点三：缓存 Token 统计

关注：

- `cache_creation_input_tokens`：创建缓存的输入 Token
- `cache_read_input_tokens`：从缓存读取的输入 Token
- `input_tokens`：本次普通输入 Token

第一次请求可能创建缓存，后续请求才可能读取缓存。

### 知识点四：DeepSeek 兼容性

当前 README 说明 DeepSeek 兼容端点会忽略 `cache_control`，相关缓存字段通常为
0。因此此文件在 DeepSeek 上主要验证请求结构，不保证观察真实缓存命中。

### 知识点五：收益判断

Prompt Caching 不是所有请求都值得使用。要考虑：

- 前缀是否足够长
- 是否会被重复使用
- 请求频率
- 缓存有效期
- 创建缓存与读取缓存的成本

### 学完后应该知道

- 哪些内容适合缓存
- cache_control 放在哪里
- 如何判断缓存是否命中
- 兼容端点为什么可能没有缓存收益

### interactive_chat.py

### 文件定位

把前面的消息历史、流式输出和系统提示组合成一个可交互会话。

### 知识点一：本地消息历史

```python
messages = []
```

用户输入后先追加 user 消息，模型回复后再追加 assistant 消息。它和
`multi_turn.py` 的原理相同，但加入了交互体验。

### 知识点二：命令设计

- `/exit`：退出程序
- `/clear`：清空上下文
- `/history`：查看历史

命令由应用自己解析，不会发送给模型。真实产品可能还会加入：

- `/help`
- `/retry`
- `/model`
- `/system`
- `/save`
- `/load`

### 知识点三：流式界面

```python
print("AI: ", end="", flush=True)
with client.messages.stream(...):
    ...
```

流式输出结束后，再通过 `get_final_message()` 获取完整回复，并把文本加入历史。

### 知识点四：系统提示

```python
SYSTEM_PROMPT = "你是一个简洁、友好的中文 AI 助手。"
```

system 没有放进 messages 数组，而是每次请求作为顶层参数传入。

### 知识点五：工程缺口

这个交互示例仍没有：

- 历史长度限制
- 持久化
- 多用户隔离
- 异常恢复
- Token 预算
- 输入注入防护
- 会话超时

生产聊天系统必须补齐这些能力。

### 学完后应该知道

- 交互命令如何和模型消息分离
- 流式输出后如何保存历史
- system 和 messages 的区别
- 教学聊天循环距离生产聊天服务还差什么

## 七、推荐学习顺序

### 第一遍：先跑通基础调用

```text
env_check.py
common.py
basic_chat.py
multi_turn.py
stream_chat.py
```

目标是理解：

```text
配置 -> 客户端 -> messages -> content blocks -> 多轮历史 -> 流式输出
```

### 第二遍：掌握工具

```text
tool_use.py
multi_tool_loop.py
human_approval.py
structured_output.py
```

目标是理解：

```text
工具声明 -> tool_use -> 执行 -> tool_result -> 最终回答
```

同时掌握工具异常、人工审批和结构化提取。

### 第三遍：掌握异步和底层流

```text
async_chat.py
stream_events.py
parameter_control.py
```

目标是知道：

- sync / async / stream 的边界
- 高层文本流和底层事件流差异
- 官方参数和兼容参数的差异

### 第四遍：掌握生产实践

```text
error_handling.py
thinking_chat.py
prompt_caching.py
interactive_chat.py
```

目标是知道：

- 失败怎么处理
- thinking 怎么解析
- 长前缀怎么缓存
- 如何组织完整聊天循环

## 八、学完之后的自检问题

能够清楚回答这些问题，说明已经建立了 Claude SDK 的基础框架：

1. Claude Messages API 为什么是无状态的？
2. `response.content` 为什么是 block 列表？
3. text、tool_use、thinking block 分别代表什么？
4. `max_tokens`、thinking budget 和上下文长度有什么区别？
5. 多轮对话为什么要同时保存 user 和 assistant 消息？
6. `messages.stream()` 和原始 stream event 有什么不同？
7. Tool Use 中模型和应用各自负责什么？
8. `tool_use_id` 为什么必须原样回传？
9. 工具执行失败时为什么要设置 `is_error=True`？
10. 高风险工具为什么必须在执行前审批？
11. 如何通过 tool_choice 实现结构化输出？
12. AsyncAnthropic 为什么不会自动让所有请求并发？
13. 哪些 Anthropic 异常不应该重试？
14. thinking block 为什么不能直接当作最终回答？
15. Prompt Caching 为什么要求前缀稳定？
16. DeepSeek 兼容端点与官方 Anthropic 有哪些差异？

## 九、当前示例中的教学模拟

下面这些实现适合学习，不应不加改造直接用于生产：

1. 天气返回 Mock 数据，不是真实天气服务。
2. 邮件函数的返回文案表示“已发送”，但实际没有发送任何邮件。
3. `multi_tool_loop.py` 的工具分发没有完整权限模型。
4. 工具参数来自模型，进入真实业务前必须做严格校验。
5. 人工审批只读取环境变量或终端输入，不是正式审批系统。
6. 消息历史只保存在内存中，进程退出即丢失。
7. `interactive_chat.py` 没有会话持久化和 Token 预算。
8. `prompt_caching.py` 在 DeepSeek 兼容端点通常没有实际缓存效果。
9. `thinking_chat.py` 的 thinking 能力取决于具体模型和兼容端点。
10. 示例没有完整实现超时、熔断、限流、审计、脱敏和监控。

## 十一、Skill 加载专题

### 1. 本模块中的 Skill 指什么

Skill 是一份可复用的操作说明，核心文件通常是：

```text
skills/<skill-name>/SKILL.md
```

本项目的统一约定是：

- `SKILL.md` 顶部 frontmatter 声明 `name` 和 `description`
- 正文描述触发条件、执行步骤、输入输出格式和示例
- Skill 本身不一定是可调用函数，更常见的是告诉 Agent 如何组合已有工具

Claude SDK 中有两个相关但不同的概念：

1. 本地 Agent Skill：应用自己发现并解析 `SKILL.md`，再把正文注入 Prompt。
2. 托管 Skill：通过 Anthropic Skills API 上传 Skill，再让模型在代码执行
   容器中使用它。

不要把两者混为一谈。前者只要模型支持 Messages API 就能做；后者依赖
Anthropic 1.5 SDK、Skills API、Container 和代码执行能力。

### 2. 共享加载器 `tools/skill_loader.py`

`skill_loading.py` 没有重新实现一套 Skill 解析逻辑，而是使用项目根目录下的
`tools/skill_loader.py`。它提供：

- `discover_skills(roots)`：递归查找所有 `SKILL.md`
- `find_skill(skills, name)`：按名称精确选择
- `render_catalog(skills)`：只渲染 name/description 的目录
- `render_full_context(skills)`：把完整 Markdown 包裹成 system prompt
- `upload_files(skill)`：生成 Skills API 上传参数

`Skill` 是 frozen dataclass，加载后不会在运行中意外修改：

```python
Skill(
    name="text-reversal",
    description="...",
    directory=Path(...),
    path=Path(...),
    markdown="完整 SKILL.md 内容",
    body="去掉 frontmatter 后的正文",
)
```

解析器只支持本示例需要的最简 frontmatter，没有引入 PyYAML。真实项目如果
需要复杂 YAML、frontmatter 类型校验、继承或引用关系，应使用正式解析库，
并且不要把所有本地文件都盲目塞给模型。

### 3. `skill_loading.py --mode session`

默认会话语义的数据流是：

```text
skills/text-reversal/SKILL.md
  -> discover_skills()
  -> 权限过滤与 Skill 目录
  -> RuleBasedSkillRouter
  -> 规则未命中时 AnthropicSkillRouter
  -> SkillSession 渐进加载完整正文
  -> render()
  -> messages.create(system=..., messages=[...])
  -> 模型按 SKILL.md 的步骤回答
```

关键点：

1. 会话状态保存在 `SkillSessionManager` 中，以 `session_id` 隔离。
2. 已加载 Skill 在后续轮次复用，只加载本轮新增命中的 Skill。
3. `--conversation` 运行两个会话，`--inspect` 不调用真实模型。
4. `--mode local` 仍保留显式单 Skill 注入，`--mode hosted` 会把选中 Skill
   懒上传并通过 `container.skills` 挂载。

执行：

```bash
venv/bin/python claude-sdk/skill_loading.py --mode local
venv/bin/python claude-sdk/skill_loading.py --mode local --inspect
```

### 4. `skill_loading.py --mode hosted`

托管模式演示的是 Anthropic SDK 1.5 的两步流程：

```text
1. client.skills.create(files=[("SKILL.md", bytes, "text/markdown")])
   -> 返回 Skill(id, display_name, latest_version_id, source)

2. client.messages.create(
       container={
           "skills": [
               {
                   "type": "custom",
                   "skill_id": created.id,
                   "version": "latest",
               }
           ]
       },
       tools=[code_execution],
       messages=[...],
   )
```

这里要理解三个层次：

- `skills.create` 只是把文件包上传为托管 Skill 的一个版本。
- `container.skills` 才是告诉模型在本次代码执行容器中加载哪些 Skill。
- `code_execution` 是容器真正运行脚本、读写文件所依赖的服务端工具。

本示例使用：

```python
CODE_EXECUTION_TOOL = {
    "type": "code_execution_20260521",
    "name": "code_execution",
}
```

工具类型会随 API 版本演进。运行前应核对当前 SDK 和官方文档，不要认为某个
日期后缀永远可用。

`hosted` 模式默认创建测试 Skill，调用结束后通过 `client.skills.delete`
清理。若调试时需要保留，可加 `--keep`。

### 5. `--inspect` 为什么重要

Skills 和 Container 依赖官方 Anthropic 能力。项目 `.env` 如果指向
DeepSeek 兼容端点，`client.skills.create` 和 `container.skills` 通常会被
拒绝或静默忽略。

因此学习流程建议先执行：

```bash
venv/bin/python claude-sdk/skill_loading.py --mode hosted --inspect
```

它不调用网络，只展示：

- 将要上传的文件名、MIME 类型和大小
- 将调用的 `POST /v1/skills`
- 将发送的 `container.skills`、工具和消息参数

看到参数后，再决定是否用官方端点真实调用。

### 6. 本地 Prompt 注入与托管 Skill 的区别

| 对比项 | local | hosted |
|---|---|---|
| 核心机制 | SKILL.md 放入 system | Skills API + Container |
| 模型是否自己执行代码 | 否 | 是，服务端代码执行工具 |
| 是否支持 DeepSeek 兼容端点 | 通常可以 | 通常不可以 |
| 是否需要上传 Skill | 不需要 | 需要 |
| 离线参数检查 | `--mode local --inspect` | `--mode hosted --inspect` |
| 适合学习的内容 | Skill 结构、Prompt 组织 | SDK 新 API、容器能力 |

### 7. 学完后应该知道

1. Agent Skill 的 `SKILL.md` 和 function tool 的区别是什么？
2. `name`、`description` 在 Skill 目录中分别起什么作用？
3. 本地加载时为什么要解析 frontmatter，而目录模式又为什么只注入
   name/description？
4. `client.skills.create` 上传的究竟是什么？
5. `container.skills` 中的 `type`、`skill_id`、`version` 分别表示什么？
6. 为什么加载 Skill 还要求绑定 code execution 工具？
7. 哪些场景应该使用本地 Prompt 注入，哪些应该使用托管 Skill？
8. 为什么 `--inspect` 比直接真实调用更适合第一步验证？

### 8. 生产注意事项

1. 不要无条件加载所有 Skill 的完整正文，应先做相关性筛选。
2. 第三方 Skill 可能包含提示注入或恶意步骤，执行前必须审计。
3. `client.skills.create` 会产生远程资源，测试后应删除，避免成本和数据残留。
4. 容器执行可能产生文件、网络请求和费用，需要预算、权限和审计。
5. 中转服务可能接受参数但不真正提供 Skills/Container 能力，不能只看请求成功。

## 十二、当前代码检查提醒

阅读或运行前建议先做一次语法检查：

```bash
venv/bin/python -m compileall claude-sdk
```

当前工作区中的 `multi_turn.py` 含有一个未提交的无效语法：

```python
messages.apjei shipend(...)
```

这会阻止该文件运行。根据文件上下文，正确调用应为：

```python
messages.append(...)
```

本知识文档仅做梳理，没有修改这项用户工作区变更。
