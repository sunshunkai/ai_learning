# AI Learning 学习项目

> 一个模块化的 Python AI 应用开发学习仓库。项目采用「单仓库、多模块」
> 的组织方式，将 Python 基础、LLM SDK、AgentScope、LangChain 等主题
> 拆分到独立目录中，便于按主题学习和按需扩展。
> 各模块保持独立演进，并在内容迭代时尽量兼容已有学习路径与代码示例。

## 目录结构

```
ai-learning/
├── README.md          # 本文件：总入口与学习路线
├── .env.example       # API Key 配置模板（复制为 .env 并填 Key）
├── .gitignore         # 忽略 venv / __pycache__ / .env 等
├── venv/              # Python 虚拟环境（首次 pip install 后生成）
├── skills/            # 跨模块复用的本地 Skill 目录（SKILL.md）
│   └── text-reversal/ # 反转文本的示例 Skill
├── basics/            # 【module 1】Python 基础语法（零基础从这里开始）
│   ├── __init__.py    # 模块学习顺序说明
│   ├── hello.py       # 第一个可运行脚本
│   ├── data_types.py  # 变量 / list / dict / 流程控制
│   ├── funcs.py       # 函数
│   ├── classes.py     # 类（入门够用）
│   ├── async_basics.py# async/await（学 AI SDK 前必看）
│   └── requirements.txt
├── claude-sdk/        # 【module 2】Anthropic Claude SDK
│   ├── README.md      # 本 module 使用说明
│   ├── __init__.py
│   ├── common.py      # 客户端与文本解析公共工具
│   ├── env_check.py   # 自检 API Key（先跑这个）
│   ├── basic_chat.py  # 一问一答
│   ├── multi_turn.py  # 多轮对话（带记忆）
│   ├── stream_chat.py # 流式输出
│   ├── tool_use.py    # 工具调用（function calling）
│   ├── human_approval.py  # 人工审批后执行高风险工具
│   ├── multi_tool_loop.py # 多工具循环和工具异常
│   ├── structured_output.py # 强制结构化输出
│   ├── async_chat.py    # 异步调用与异步流
│   ├── stream_events.py # 底层流事件
│   ├── parameter_control.py # system / stop / metadata 等参数
│   ├── error_handling.py # 超时、重试和异常
│   ├── thinking_chat.py # extended thinking
│   ├── prompt_caching.py # prompt caching
│   ├── interactive_chat.py # 交互式多轮命令行
│   ├── skill_loading.py # 本地 Skill 注入 / 托管 Skill 容器加载
│   └── requirements.txt
├── agentscope-demo/   # 【module 3】AgentScope 2.x（DeepSeek）
│   ├── README.md      # 安装、学习路线和运行命令
│   ├── KNOWLEDGE_GUIDE.md # 官方文档与 Demo 的知识整合
│   ├── common.py      # DeepSeek 模型公共工厂
│   ├── c01_02_basic_model.py ... c08_01_agent_service.py
│   └── requirements.txt
├── langchain-demo/    # 【module 4】LangChain（框架层，1.x 写法）
│   ├── README.md      # 本 module 使用说明
│   ├── common.py      # DeepSeek 模型工厂
│   ├── quickstart.py  # invoke / stream / messages
│   ├── 01_chat_models.py ... 16_skill_loading.py
│   └── requirements.txt
└── tools/             # 公用小工具（跨 module 共享）
    ├── __init__.py
    ├── load_env.py    # 从 .env 加载环境变量
    └── skill_loader.py # 发现/解析/渲染 SKILL.md
```

## 如何新增一个 module（想学新主题时）

想加一个新的学习主题（比如某 Agent 框架、某向量库），照抄下面的模板即可：

```bash
# 1. 新建文件夹并标记为包
mkdir ai-learning/xxx
touch ai-learning/xxx/__init__.py

# 2. 建一个能跑通的 hello 脚本
#    ai-learning/xxx/hello.py

# 3. 写这个 module 专属的依赖清单
#    ai-learning/xxx/requirements.txt
```

然后在本文档的「目录结构」和「学习路线」里补一行即可。

## 为什么这样组织（对应 Java 的概念）

| 本项目的概念 | Java 的对应物 |
|--------------|--------------|
| 顶层目录 `python-learning` | Maven 的 `parent` / 根工程 |
| 每个子文件夹 `basics/`、`claude-sdk/` | Maven 的一个 `module` |
| 文件夹里的 `__init__.py` | 把它变成一个 Python 包 |
| 每个子文件夹自己的 `requirements.txt` | 各 module 的 `pom.xml` 依赖清单 |
| `import basics.xxx` | `import com.xxx.yyy` |

> ⚠️ 注意：Java 用 Maven/Gradle 一个命令统一管理整个工程；
> **Python 没有等价的中央构建工具**。这里用的是「虚拟环境 + 每个子文件夹
> 独立管理依赖」的方式，简单直观，足够学习用。

## 学习路线（强烈建议按顺序）

1. **`basics/`**：先跑通 Python 基础（约 1 周）。没有这个基础，直接看 AI SDK 会很吃力。
2. **`claude-sdk/`**：用 Claude SDK 感受"在代码里调大模型"，并学习 Skill
   的本地注入和托管容器加载，最简单直接。
3. **`agentscope-demo/`**：学习 AgentScope 2.x 的 Agent、工具、权限、RAG、
   记忆、工作区、MCP 和服务化能力。
4. **`langchain-demo/`**：对比 LangChain 这类框架怎么做编排，包括 Skill
   目录懒加载。（LangChain 本身可以对接 Claude，很多概念是相通的。）

## 环境准备（首次使用）

### 1. 创建并激活虚拟环境（每个 module 或整个项目用一个 venv 都行）

```bash
# 从项目根目录进入
cd python-learning

# 用 Python 3 创建虚拟环境（venv 会把依赖隔离在本项目内，不污染全局）
python3 -m venv venv

# 激活（macOS / Linux）
source venv/bin/activate
# Windows 激活方式：
# venv\Scripts\activate
```

激活后，命令行前缀会出现 `(venv)`。

### 2. 安装某个 module 的依赖

```bash
# 先学哪个 module，就装哪个的依赖
pip install -r basics/requirements.txt
pip install -r claude-sdk/requirements.txt
pip install -r agentscope-demo/requirements.txt
pip install -r langchain-demo/requirements.txt
```

### 3. 配置 API Key（学 claude-sdk / agentscope-demo / langchain 时需要）

Claude SDK 需要 Anthropic 的 API Key。国内直连有限制，通常需要代理或
兼容 Anthropic 格式的中转服务。配置方式有两种，任选其一：

AgentScope 与 LangChain demo 使用 DeepSeek 的 OpenAI 兼容接口，需要在
`.env` 中配置 `DEEPSEEK_API_KEY`。

**方式 A：手动 export（临时，仅当前终端有效）**

```bash
# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-xxx"

# Windows（PowerShell）
# $env:ANTHROPIC_API_KEY="sk-ant-xxx"
```

**方式 B（推荐）：用项目根 `.env` 文件（一劳永逸，脚本自动读取）**

```bash
# 1. 复制模板并改名
cp .env.example .env      # Windows: copy .env.example .env

# 2. 用编辑器打开 .env，把 ANTHROPIC_API_KEY= 后面换成你的 Key
```

各脚本顶部会自动调用 `tools/load_env.py` 读取这个 `.env`，无需手动 export，
且 `.env` 已被 .gitignore 忽略，不会误提交密钥。

### 4. 运行脚本

```bash
python basics/hello.py
python claude-sdk/env_check.py    # 先自检环境
python claude-sdk/basic_chat.py
python agentscope-demo/c01_01_env_check.py
python agentscope-demo/c03_01_basic_agent.py
```

## 常见问题

- **为什么有的 module 代码 import 报错？** 检查是否激活了 venv、是否 `pip install` 过
  对应 module 的 `requirements.txt`。
- **想中断长任务 / 流式输出？** 终端里按 `Ctrl+C`。
