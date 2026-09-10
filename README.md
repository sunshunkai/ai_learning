# AI Learning 学习项目

> 面向有 Java 经验、想学 AI 应用开发的开发者（当前零 Python 基础），
> 用「单仓库多文件夹」的方式，像 Maven 多 module 一样把不同主题的
> 学习内容拆开、互不干扰。主题会随学习进度持续扩展（当前已含
> Python 基础、Claude SDK、LangChain，未来可加 Agent 框架等）。

## 目录结构

```
ai-learning/
├── README.md          # 本文件：总入口与学习路线
├── .env.example       # API Key 配置模板（复制为 .env 并填 Key）
├── .gitignore         # 忽略 venv / __pycache__ / .env 等
├── venv/              # Python 虚拟环境（首次 pip install 后生成）
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
│   ├── env_check.py   # 自检 API Key（先跑这个）
│   ├── basic_chat.py  # 一问一答
│   ├── multi_turn.py  # 多轮对话（带记忆）
│   ├── stream_chat.py # 流式输出
│   ├── tool_use.py    # 工具调用（function calling）
│   └── requirements.txt
├── langchain-demo/    # 【module 3】LangChain（框架层，1.x 写法）
│   ├── README.md      # 本 module 使用说明
│   ├── __init__.py
│   ├── quickstart.py  # 接入聊天模型 + 流式 + 多轮
│   └── requirements.txt
└── tools/             # 公用小工具（跨 module 共享）
    ├── __init__.py
    └── load_env.py    # 从 .env 加载环境变量
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
2. **`claude-sdk/`**：用 Claude SDK 感受"在代码里调大模型"，最简单直接。
3. **`langchain-demo/`**：看完 Claude SDK 后，再对比 LangChain 这类框架怎么做编排。
   （LangChain 本身可以对接 Claude，很多概念是相通的。）

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
```

### 3. 配置 API Key（学 claude-sdk / langchain 时需要）

Claude SDK 需要 Anthropic 的 API Key。国内直连有限制，通常需要代理或
兼容 Anthropic 格式的中转服务。配置方式有两种，任选其一：

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
```

## 常见问题

- **为什么有的 module 代码 import 报错？** 检查是否激活了 venv、是否 `pip install` 过
  对应 module 的 `requirements.txt`。
- **想中断长任务 / 流式输出？** 终端里按 `Ctrl+C`。
