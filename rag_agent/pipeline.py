import os
import pickle
from dataclasses import dataclass, field

from .document import load_documents, chunk_documents, ChunkingConfig
from .embeddings import create_embeddings
from .vectorstore import create_vectorstore, add_documents, load_vectorstore, get_all_documents
from .retriever import (
    retrieve, format_context, RetrievalConfig, BM25Retriever,
)
from .generator import create_llm, generate
from . import logger


DB_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
DOCS_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "documents")


@dataclass
class RAGConfig:
    chunk_size: int = 500
    chunk_overlap: int = 80
    chunk_strategy: str = "sentence"
    top_k: int = 8
    collection_name: str = "rag_agent"

    retrieval_mode: str = "hybrid"
    enable_rerank: bool = True
    rerank_top_k: int = 4
    enable_query_rewrite: bool = True


class RAGPipeline:
    def __init__(
        self,
        db_dir: str | None = None,
        config: RAGConfig | None = None,
    ):
        self.db_dir = db_dir or DB_DIR_DEFAULT
        self.config = config or RAGConfig()
        self._embeddings = None
        self._llm = None
        self._bm25: BM25Retriever | None = None

    def _get_embeddings(self):
        if self._embeddings is None:
            self._embeddings = create_embeddings()
        return self._embeddings

    @property
    def llm(self):
        if self._llm is None:
            self._llm = create_llm()
        return self._llm

    def _get_bm25(self, vectorstore) -> BM25Retriever:
        if self._bm25 is not None:
            return self._bm25

        bm25_path = os.path.join(self.db_dir, "bm25_index.pkl")
        if os.path.exists(bm25_path):
            logger.info("[BM25] 加载已有索引")
            with open(bm25_path, "rb") as f:
                self._bm25 = pickle.load(f)
                if self._bm25._documents:
                    logger.info(f"[BM25] 索引已加载: {len(self._bm25._documents)} 文档")
            return self._bm25

        logger.info("[BM25] 未找到索引文件，正在从向量库构建 BM25 索引 ...")
        all_docs = get_all_documents(vectorstore)
        if not all_docs:
            logger.info("[BM25] 向量库为空，跳过 BM25 构建")
            self._bm25 = BM25Retriever()
            return self._bm25

        self._bm25 = BM25Retriever()
        self._bm25.index(all_docs)
        with open(bm25_path, "wb") as f:
            pickle.dump(self._bm25, f)
        logger.info(f"[BM25] 索引构建完成并保存: {len(all_docs)} 文档")
        return self._bm25

    def ingest(self, docs_dir: str | None = None) -> int:
        logger.section("文档导入 (Ingestion)")
        source = docs_dir or DOCS_DIR_DEFAULT
        logger.keyval("文档目录", source)
        logger.keyval("向量库目录", self.db_dir)
        logger.keyval("Chunk 大小", str(self.config.chunk_size))
        logger.keyval("Chunk 重叠", str(self.config.chunk_overlap))

        docs = load_documents(source)
        if not docs:
            logger.info("未找到文档，跳过导入")
            return 0

        logger.info(f"共加载 {len(docs)} 个文档")
        print(f"共加载 {len(docs)} 个文档")

        chunk_cfg = ChunkingConfig(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            strategy=self.config.chunk_strategy,
        )
        chunks = chunk_documents(docs, chunk_cfg)

        embeddings = create_embeddings()
        vs = create_vectorstore(
            self.db_dir,
            embeddings,
            self.config.collection_name,
        )
        add_documents(vs, chunks)

        # 删除旧的 BM25 索引，force rebuild on next query
        bm25_path = os.path.join(self.db_dir, "bm25_index.pkl")
        if os.path.exists(bm25_path):
            os.remove(bm25_path)
            logger.info("[BM25] 旧索引已删除，将在下次查询时重建")

        logger.info(f"导入完成: {len(chunks)} 个 chunks -> {self.db_dir}")
        return len(chunks)

    def query(self, question: str) -> dict:
        logger.section(f"RAG 问答: {question}")

        embeddings = self._get_embeddings()
        vs = load_vectorstore(
            self.db_dir,
            embeddings,
            self.config.collection_name,
        )

        bm25 = None
        if self.config.retrieval_mode in ("bm25", "hybrid"):
            bm25 = self._get_bm25(vs)

        retrieval_cfg = RetrievalConfig(
            top_k=self.config.top_k,
            retrieval_mode=self.config.retrieval_mode,
            enable_rerank=self.config.enable_rerank,
            rerank_top_k=self.config.rerank_top_k,
            enable_query_rewrite=self.config.enable_query_rewrite,
        )

        docs = retrieve(
            query=question,
            vectorstore=vs,
            llm=self.llm,
            bm25=bm25,
            config=retrieval_cfg,
        )

        context = format_context(docs)
        answer = generate(self.llm, question, context)

        logger.section("问答结果")
        logger.info(f"  {answer}")

        return {
            "question": question,
            "answer": answer,
            "sources": list(set(d.metadata.get("source", "unknown") for d in docs)),
            "chunks": docs,
        }
