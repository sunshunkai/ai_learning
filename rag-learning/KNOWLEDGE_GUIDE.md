# RAG 学习详细指南

这个模块的目标不是只跑一个 RAG，而是把 RAG 拆开，理解每个组件分别做什么。

## 文件地图

| 文件 | 知识点 | 是否需要额外依赖 |
|---|---|---|
| `01_document_loading.py` | `Document`、`page_content`、`metadata` | 否 |
| `02_chunking.py` | `chunk_size`、`chunk_overlap`、分块边界 | 否 |
| `03_embeddings_hash.py` | 向量、维度、cosine similarity | 否 |
| `04_embeddings_bge.py` | 本地 BGE 中文 embedding | 是 |
| `05_embeddings_dashscope.py` | 阿里云 DashScope embedding | 是 |
| `06_vectorstore_memory.py` | 内存向量库、写入、检索 | 否 |
| `07_vectorstore_chroma.py` | Chroma、持久化、metadata | 是 |
| `08_vectorstore_faiss.py` | FAISS、高性能本地索引 | 是 |
| `09_rag_naive.py` | 基础 RAG 完整链路 | 在线需 DeepSeek |
| `10_rag_citation.py` | 带来源引用的 RAG | 在线需 DeepSeek |
| `11_provider_switch.py` | 配置化切换 embedding / vectorstore | 部分可选 |
| `12_retrieval_scores.py` | 检索分数、结果排序 | 否 |
| `demo.py` | 快速入口，通过参数切换 provider | 可选 |
| `verify_examples.py` | 离线批量验证基础 demo | 否 |

## 1. 文档加载

RAG 的第一步是把资料变成 `Document` 对象。

```bash
python rag-learning/01_document_loading.py
```

需要理解：

- `page_content` 是实际会参与分块和 embedding 的正文。
- `metadata` 不参与检索，但可以保存来源、作者、页码等信息。
- 回答阶段应该尽量把 `metadata` 展示给用户，形成引用。

## 2. 文本分块

文档通常太长，不能整篇塞给模型，也不能直接整篇 embedding。

```bash
python rag-learning/02_chunking.py
```

关键参数：

- `chunk_size`：每个 chunk 大约多少字符。
- `chunk_overlap`：相邻 chunk 重叠多少内容。
- `separators`：优先在哪些边界切分。

经验：

- 中文知识库可以先尝试 `chunk_size=300` 到 `800`，但本 demo 的文档很短，所以用较小值。
- 如果某个答案经常被切断，可以增大 overlap。
- 如果检索结果太宽泛，可以减小 chunk_size。

## 3. Embedding

Embedding 把文本变成向量。

### 3.1 Hash Embedding

```bash
python rag-learning/03_embeddings_hash.py
```

它是本地确定性实现，不需要网络，也不理解同义词，只适合教学和离线跑通链路。

### 3.2 BGE Embedding

```bash
python rag-learning/04_embeddings_bge.py
```

BGE 是中文场景常用的本地 embedding。建议模型：

- `BAAI/bge-small-zh-v1.5`
- `BAAI/bge-m3`

需要安装：

```bash
pip install langchain-community sentence-transformers
```

### 3.3 DashScope Embedding

```bash
python rag-learning/05_embeddings_dashscope.py
```

阿里云 embedding 模型需要设置：

```bash
export DASHSCOPE_API_KEY=...
```

模型默认使用 `text-embedding-v4`，可以在 `.env` 中配置：

```text
DASHSCOPE_API_KEY=...
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4
```

## 4. VectorStore

VectorStore 负责保存向量，并执行相似度检索。

### 4.1 InMemoryVectorStore

```bash
python rag-learning/06_vectorstore_memory.py
```

特点：

- 不需要安装额外服务。
- 数据只存在于当前进程，进程结束就消失。
- 适合 demo、测试、理解流程。

### 4.2 Chroma

```bash
python rag-learning/07_vectorstore_chroma.py
```

特点：

- 本地可运行。
- 支持持久化到目录。
- 容易查看 collection 和 metadata。

安装：

```bash
pip install langchain-community chromadb
```

### 4.3 FAISS

```bash
python rag-learning/08_vectorstore_faiss.py
```

特点：

- 检索速度很快。
- 适合千万级以下向量、本地部署。
- 相比 Chroma，metadata 管理和使用门槛稍高。

安装：

```bash
pip install langchain-community faiss-cpu
```

## 5. RAG 主链路

基础 RAG 可以先用 `hash + memory` 跑通：

```bash
python rag-learning/09_rag_naive.py --offline
python rag-learning/09_rag_naive.py
```

带来源引用：

```bash
python rag-learning/10_rag_citation.py --offline
python rag-learning/10_rag_citation.py
```

查看检索分数：

```bash
python rag-learning/12_retrieval_scores.py
```

## 6. Provider 切换

核心封装在 `factory.py`：

- `RAGConfig`
- `build_embedding`
- `build_vectorstore`
- `build_retriever`
- `build_rag_chain`
- `RAGFactory`

切换 provider 时，业务代码只改配置：

```python
config = RAGConfig(
    embedding_provider="hash",
    vectorstore_provider="memory",
)
```

换成 BGE 和 Chroma：

```python
config = RAGConfig(
    embedding_provider="bge-local",
    vectorstore_provider="chroma",
)
```

换成阿里云 embedding：

```python
config = RAGConfig(
    embedding_provider="dashscope",
    vectorstore_provider="chroma",
)
```

运行：

```bash
python rag-learning/11_provider_switch.py
```

## 7. 离线验证

没有额外依赖时，可以批量验证基础 demo：

```bash
python rag-learning/verify_examples.py
```

## 8. 后续进阶方向

基础 RAG 跑通后，可以继续学习：

- Rerank：检索后重新排序，提高最相关 chunk 的排名。
- Hybrid Search：向量检索 + 关键词检索组合。
- Query Rewriting：把用户问题改写成更适合检索的查询。
- Agentic RAG：让模型决定是否需要检索、何时检索、检索后是否重写。
- RAG Evaluation：评测召回率、答案相关性、忠实度。
- 多路召回：不同 embedding、不同 vectorstore 同时召回再合并。

这些内容不建议一开始就做，先完成本模块的 `01` 到 `12`。
