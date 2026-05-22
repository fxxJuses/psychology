import os
from typing import List

from langchain_core.embeddings import Embeddings
from openai import OpenAI

from . import logger

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_EMBEDDING_MODEL = "text-embedding-v3"


class DashScopeEmbeddings(Embeddings):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY is not set")
        self._client = OpenAI(
            api_key=api_key,
            base_url=DASHSCOPE_BASE_URL,
        )
        self._model = model or DASHSCOPE_EMBEDDING_MODEL
        logger.info(f"[Embeddings] 模型: {self._model}, dim=1024, base_url={DASHSCOPE_BASE_URL}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [t.replace("\n", " ") for t in texts]
        if logger.is_verbose():
            previews = [t[:80] + "..." if len(t) > 80 else t for t in texts]
            logger.info(f"[Embeddings] 批量向量化 {len(texts)} 条文本")
            for i, p in enumerate(previews):
                logger.keyval(f"chunk[{i}]", p)
        resp = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        result = [d.embedding for d in resp.data]
        if logger.is_verbose():
            logger.info(f"[Embeddings] 返回 {len(result)} 个向量, dim={len(result[0])}, "
                        f"tokens={resp.usage.total_tokens}")
        return result

    def embed_query(self, text: str) -> List[float]:
        if logger.is_verbose():
            preview = text[:80] + "..." if len(text) > 80 else text
            logger.info(f"[Embeddings] 查询向量化: \"{preview}\"")
        return self.embed_documents([text])[0]


def create_embeddings(api_key: str | None = None) -> DashScopeEmbeddings:
    return DashScopeEmbeddings(api_key=api_key)
