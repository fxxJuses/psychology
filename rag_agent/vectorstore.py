from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from . import logger


def create_vectorstore(
    persist_dir: str,
    embedding: Embeddings,
    collection_name: str = "rag_agent",
) -> Chroma:
    logger.info(f"[VectorStore] 创建向量库: collection={collection_name}, "
                f"persist_dir={persist_dir}")
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding,
        collection_name=collection_name,
    )


EMBEDDING_BATCH_SIZE = 10


def add_documents(vectorstore: Chroma, documents: List[Document]) -> None:
    total = len(documents)
    logger.info(f"[VectorStore] 添加 {total} 个文档到向量库 (batch_size={EMBEDDING_BATCH_SIZE})")
    for i in range(0, total, EMBEDDING_BATCH_SIZE):
        batch = documents[i:i + EMBEDDING_BATCH_SIZE]
        vectorstore.add_documents(batch)
        logger.info(f"[VectorStore] 批次 {i // EMBEDDING_BATCH_SIZE + 1}: "
                    f"已添加 {min(i + EMBEDDING_BATCH_SIZE, total)}/{total}")
    logger.info(f"[VectorStore] 添加完成, 当前共 {vectorstore._collection.count()} 条记录")


def get_all_documents(vectorstore: Chroma) -> List[Document]:
    collection = vectorstore._collection
    if collection is None or collection.count() == 0:
        return []
    result = collection.get(include=["documents", "metadatas"])
    docs = []
    for i, content in enumerate(result.get("documents", [])):
        metadata = result.get("metadatas", [{}])[i] if i < len(result.get("metadatas", [])) else {}
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def load_vectorstore(
    persist_dir: str,
    embedding: Embeddings,
    collection_name: str = "rag_agent",
) -> Chroma:
    vs = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding,
        collection_name=collection_name,
    )
    count = vs._collection.count() if vs._collection else 0
    logger.info(f"[VectorStore] 加载向量库: collection={collection_name}, "
                f"现有 {count} 条记录")
    return vs
