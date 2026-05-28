# Psychology RAG — 心理学文档智能问答系统

基于 **RAG (Retrieval-Augmented Generation)** 的心理学文档问答系统，支持段落级切分、层级索引、混合检索 + LLM 精排，以及完整的检索+生成效果评估。

## 特性

- **段落级切分**: OCR 文本重建为语义完整段落 (~1200字)，携带章节/前后链式 metadata
- **v4 Embedding**: DashScope text-embedding-v4，启用 text_type + instruct 参数
- **层级索引**: 文档级 → 章节级 → 段落级三层索引，宏观/微观问题自动路由
- **混合检索 + Rerank**: Vector + BM25 → RRF 融合 → LLM 精选，支持 HyDE 和查询改写
- **严格生成**: 区分核心/补充文档，禁止编造引用
- **完整评估**: 6 项检索指标 + 2 项生成质量指标 (LLM-as-Judge)，15 条测试用例

## 架构

```
Raw Documents (PDF)
    ↓ [MuPDF + OCR text extraction]
    ↓ [Paragraph Chunker: OCR清洗 → 章节检测 → 段落重建]
Paragraph Chunks (~1200 chars, with chapter/prev/next metadata)
    ↓ [text-embedding-v4 (text_type=document, dim=1024)]
    ↓ + [Hierarchy: L1 全书摘要 → L2 章节摘要 → L3 段落]
ChromaDB (320 records) + BM25 Index
    ↓
User Query
    ↓ [HyDE / Query Rewrite]
    ↓ [is_macro_query? → L1+L2摘要优先 / 否则 ↓]
    ↓ [Vector(top-30) + BM25(top-30) → RRF Fusion]
    ↓ [LLM Rerank → Top-8]
    ↓ [Context Expansion (optional)]
Context (core docs + supplementary)
    ↓ [DeepSeek-V4-Flash: strict grounded generation]
Answer + Sources
```

## 当前最优指标 (2026-05-28)

| 指标 | hybrid_reranked | 历史最优 |
|------|-----------------|---------|
| Recall@5 | **0.709** | 0.669 |
| MRR | **0.733** | 0.612 |
| Hit@1 | **0.733** | 0.533 |
| Faithfulness | 0.873 | 0.933 |
| Answer Relevance | **0.953** | 0.933 |

## 目录结构

```
psychology/
├── main.py                     # 入口
├── rag_agent/                  # 核心模块
│   ├── cli.py                  # 命令行接口
│   ├── pipeline.py             # RAG 流水线与配置
│   ├── document.py             # 文档加载与切分 (sentence/recursive)
│   ├── paragraph_chunker.py    # 段落级切分 (章节检测+链式metadata)
│   ├── embeddings.py           # DashScope text-embedding-v4
│   ├── vectorstore.py          # Chroma 向量存储
│   ├── retriever.py            # 混合检索 + RRF + Rerank + 上下文扩展
│   ├── generator.py            # LLM 答案生成 (严格grounded)
│   ├── hierarchy.py            # 三层层级索引
│   ├── logger.py               # 结构化日志
│   └── evaluation/             # 评估子系统
│       ├── test_cases.py       # 15条测试用例 (6类)
│       ├── metrics.py          # 检索指标计算
│       ├── generation_eval.py  # 生成质量评估 (LLM-as-Judge)
│       ├── runner.py           # 评估执行器 (含reranker评估)
│       └── reporter.py         # 结果格式化与导出
├── tests/                      # 单元测试
├── data/
│   ├── documents/              # 源文档 (PDF)
│   ├── chunks/                 # 切分产物 + manifest
│   └── chroma_db/              # Chroma 持久化
├── docs/                       # 架构文档
└── benchmarks/                 # 评估基线归档
```

## 快速开始

### 环境要求

- Python 3.10+
- DashScope API Key

### 安装

```bash
pip install -r requirements.txt
export DASHSCOPE_API_KEY="your-api-key"
```

### 使用

```bash
# 1. 摄入文档 (默认: 段落切分 1200字)
python main.py ingest

# 2. 查询 (默认: hybrid + rerank + HyDE)
python main.py query "什么是灾难化思维？"

# 3. 交互对话
python main.py chat --show-sources

# 4. 评估
python main.py evaluate --eval-modes hybrid,hybrid_reranked
```

### CLI 默认参数 (最优配置)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `hybrid` | 混合检索 (Vector + BM25) |
| `--top-k` | `30` | 候选检索数量 |
| `--rerank-top-k` | `8` | LLM 精选保留数量 |
| `--chunk-size` | `1200` | 段落目标大小 |
| `--chunk-strategy` | `paragraph` | 段落级切分 |

## 语料库

| 文档 | 段落数 | 平均字数 |
|------|--------|---------|
| 焦虑心理学 | 68 | 1212 |
| 乌合之众：大众心理研究 | 102 | 1217 |
| 人人都该懂的心理学 | 126 | 1248 |
| **合计** | **296** | **1227** |

## 技术栈

| 层级 | 技术 |
|------|------|
| Embedding | DashScope text-embedding-v4 (1024-dim, text_type + instruct) |
| LLM | DeepSeek-V4-Flash (via DashScope) |
| 向量数据库 | ChromaDB (SQLite) |
| BM25 | scikit-learn TfidfVectorizer + jieba |
| 文档解析 | MuPDF + pytesseract OCR |

## License

MIT
