# RAG Learning

这个模块用一组由浅入深的脚本演示 RAG，重点说明如何封装 Embedding 和 VectorStore，
让后续切换供应商时尽量不改业务逻辑。

## 知识点

1. 文档加载：把知识库读取成 `Document` 对象
2. 文本分块：使用 `RecursiveCharacterTextSplitter` 切成 chunk
3. Embedding：hash、BGE、DashScope 三种实现
4. VectorStore：memory、Chroma、FAISS 三种实现
5. Retriever：统一的检索接口
6. RAG Prompt：把检索结果作为上下文
7. LLM 生成：DeepSeek 根据上下文回答
8. 来源展示：通过 metadata 显示引用
9. Provider 封装：通过工厂函数隔离具体供应商
10. 进阶方向：rerank、问题重写、Agentic RAG、RAG 评测

详细逐文件说明见 `KNOWLEDGE_GUIDE.md`。

## 运行

先安装依赖：

```bash
source venv/bin/activate
pip install -r rag-learning/requirements.txt
```

默认使用本地 hash embedding 和内存向量库，可以离线验证检索：

```bash
python rag-learning/verify_examples.py
python rag-learning/demo.py --offline --show-docs
```

调用 DeepSeek 生成答案：

```bash
python rag-learning/demo.py
```

换成真实中文 embedding 和本地 Chroma：

```bash
python rag-learning/demo.py --embedding bge-local --vectorstore chroma --show-docs
```

换成阿里云 DashScope embedding：

```bash
DASHSCOPE_API_KEY=... python rag-learning/demo.py --embedding dashscope --vectorstore chroma
```

也可以逐个运行独立 demo：

```bash
python rag-learning/01_document_loading.py
python rag-learning/02_chunking.py
python rag-learning/03_embeddings_hash.py
python rag-learning/06_vectorstore_memory.py
python rag-learning/09_rag_naive.py --offline
```

## 如何切换 provider

业务代码只依赖 `retriever` 和 `rag_chain`：

```python
config = RAGConfig(
    embedding_provider="hash",
    vectorstore_provider="memory",
    top_k=3,
)
```

换供应商只改配置：

```python
config = RAGConfig(
    embedding_provider="bge-local",
    vectorstore_provider="chroma",
)
```

主流程不需要变化。
