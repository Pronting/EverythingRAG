"""嵌入器接口与实现：云端 OpenAI 兼容（/embeddings）。

隐私约束：云端嵌入（CloudEmbedder）：构造零出网，首次 embed 才建连（OpenAI 兼容端点）。
出网即用户显式配置云端嵌入的结果，属 opt-in。

本地嵌入（fastembed / bge-m3 下载 + CPU 推理）已移除：统一走云端 embedding 服务，
省去本地模型下载、2GB+ 磁盘占用与 CPU 推理耗时。向量库 collection 仍按 fingerprint
隔离，切换嵌入模型 = 新向量空间 = 需重新导入。
"""

from __future__ import annotations

import hashlib
from threading import BoundedSemaphore, Lock
from typing import Protocol

from openai import OpenAI

from app.core.config import settings
from app.core.settings_store import EmbedModelConfig


class Embedder(Protocol):
    """文本嵌入接口：文档与查询使用明确分开的编码路径。"""

    fingerprint: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


QWEN3_RETRIEVAL_INSTRUCTION = (
    "Given a user query, retrieve relevant passages that answer the query"
)


def _is_qwen3_embedding_model(model: str) -> bool:
    """Return whether the configured model is a Qwen3 embedding model."""

    normalized = model.lower().replace("_", "-")
    return "qwen3" in normalized and "embedding" in normalized


def format_query_for_embedding(text: str, model: str) -> str:
    """Apply the Qwen3 query-side instruction; leave every other model untouched.

    Qwen3-Embedding is trained with ``Instruct: ...\nQuery:...`` on queries only.  Documents
    must remain raw, so changing this query policy never requires rebuilding the vector index.
    """

    if _is_qwen3_embedding_model(model):
        return f"Instruct: {QWEN3_RETRIEVAL_INSTRUCTION}\nQuery:{text}"
    return text


def embedding_space_fingerprint(base_url: str, model: str) -> str:
    """Return a collection-safe vector-space identity without exposing the endpoint URL.

    A model name is not globally unique across OpenAI-compatible providers.  Reusing one
    collection after changing ``base_url`` can therefore compare vectors from incompatible
    spaces.  API credentials are deliberately excluded: rotating a key does not change a space.
    """

    # Preserve path casing: some compatible servers route ``/V1`` and ``/v1`` differently.
    # Treating uncertain aliases as different spaces only causes a safe reindex; conflating them
    # could silently query vectors written by another deployment.
    canonical_endpoint = base_url.strip().rstrip("/")
    endpoint_digest = hashlib.sha256(canonical_endpoint.encode("utf-8")).hexdigest()[:12]
    safe_model = model.replace("/", "-")
    return f"cloud-{safe_model}-endpoint-{endpoint_digest}"


class CloudEmbedder:
    """OpenAI 兼容云端嵌入（POST {base_url}/embeddings，同步 OpenAI client）。

    - 构造零出网：OpenAI client 惰性，首次 embed_texts 才建连。
    - dim 惰性：配置未显式给 dim 时，从首次响应推导（OpenAI SDK 返回 item.embedding）。
    - fingerprint 含模型名与端点摘要：collection 还会按入库结构版本隔离；
      切换模型或服务端点 = 新向量空间 = 需重新导入，轮换 API key 则不需要。
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        dim: int | None = None,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._api_key = api_key
        self._configured_dim = dim
        self._resolved_dim: int | None = dim
        self._client: OpenAI | None = None
        self._client_lock = Lock()
        self._request_slots = BoundedSemaphore(settings.ingest_embed_workers)

    @property
    def fingerprint(self) -> str:
        """嵌入空间指纹：模型 + 不可逆端点摘要，供 collection 名隔离。"""

        return embedding_space_fingerprint(self._base_url, self._model)

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        """嵌入维度：配置给定或已从首次响应解析；未解析时抛错（不应在 embed 前用到）。"""
        if self._resolved_dim is None:
            raise RuntimeError("嵌入维度尚未解析：请先执行一次嵌入，或配置显式 dim")
        return self._resolved_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks verbatim; no query instruction is ever added."""

        return self._embed_inputs(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed one query with model-specific query formatting when supported."""

        return self.embed_queries([text])[0]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Batch query embeddings, applying Qwen3 instructions to each query only."""

        formatted = [format_query_for_embedding(text, self._model) for text in texts]
        return self._embed_inputs(formatted)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Backward-compatible alias for document ingestion callers."""

        return self.embed_documents(texts)

    def _embed_inputs(self, texts: list[str]) -> list[list[float]]:
        """Call the compatible /embeddings endpoint and resolve vector dimension once."""

        if not texts:
            return []
        client = self._ensure_client()
        with self._request_slots:
            response = client.embeddings.create(model=self._model, input=texts)
        # Providers may return batch items out of order. Preserve text-to-vector identity.
        data = list(response.data)
        if all(hasattr(item, "index") for item in data):
            if sorted(item.index for item in data) != list(range(len(texts))):
                raise RuntimeError("嵌入响应索引错误")
            data.sort(key=lambda item: item.index)
        vectors = [item.embedding for item in data]
        if len(vectors) != len(texts):
            raise RuntimeError("嵌入响应数量错误")
        with self._client_lock:
            expected_dim = self._resolved_dim or (len(vectors[0]) if vectors else 0)
            if not expected_dim or any(len(vector) != expected_dim for vector in vectors):
                raise RuntimeError("嵌入响应维度不一致")
            self._resolved_dim = expected_dim
        return vectors

    def _ensure_client(self) -> OpenAI:
        """惰性构造同步 OpenAI client（构造零出网，首次 embed 才建连）。"""
        with self._client_lock:
            if self._client is None:
                self._client = OpenAI(base_url=self._base_url, api_key=self._api_key or "local")
            return self._client


def create_embedder_from_config(embed: EmbedModelConfig) -> Embedder:
    """按设置构造嵌入器：统一走 OpenAI 兼容云端 /embeddings。

    云端配置不完整时 raise ValueError（设置页 PUT 已校验，这里兜底）。
    """
    if not embed.base_url or not embed.model:
        raise ValueError("云端嵌入未配置完整：需同时提供 Base URL 与模型名")
    api_key = embed.api_key.get_secret_value() if embed.api_key is not None else None
    return CloudEmbedder(base_url=embed.base_url, model=embed.model, api_key=api_key)
