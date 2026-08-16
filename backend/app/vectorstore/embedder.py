"""嵌入器接口与实现：云端 OpenAI 兼容（/embeddings）。

隐私约束：云端嵌入（CloudEmbedder）：构造零出网，首次 embed 才建连（OpenAI 兼容端点）。
出网即用户显式配置云端嵌入的结果，属 opt-in。

本地嵌入（fastembed / bge-m3 下载 + CPU 推理）已移除：统一走云端 embedding 服务，
省去本地模型下载、2GB+ 磁盘占用与 CPU 推理耗时。向量库 collection 仍按 fingerprint
隔离，切换嵌入模型 = 新向量空间 = 需重新导入。
"""

from __future__ import annotations

from typing import Protocol

from openai import OpenAI

from app.core.settings_store import EmbedModelConfig


class Embedder(Protocol):
    """文本嵌入接口：指纹 + 维度 + 批量嵌入。"""

    fingerprint: str
    dim: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class CloudEmbedder:
    """OpenAI 兼容云端嵌入（POST {base_url}/embeddings，同步 OpenAI client）。

    - 构造零出网：OpenAI client 惰性，首次 embed_texts 才建连。
    - dim 惰性：配置未显式给 dim 时，从首次响应推导（OpenAI SDK 返回 item.embedding）。
    - fingerprint 含模型名：collection 按 `chunks__cloud-{model}__v1` 隔离，
      切换嵌入模型 = 新向量空间 = 需重新导入（天然由 collection 名隔离）。
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

    @property
    def fingerprint(self) -> str:
        """嵌入指纹：cloud-{model}（斜杠转连字符），供 collection 名隔离。"""
        return f"cloud-{self._model.replace('/', '-')}"

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        """嵌入维度：配置给定或已从首次响应解析；未解析时抛错（不应在 embed 前用到）。"""
        if self._resolved_dim is None:
            raise RuntimeError("嵌入维度尚未解析：请先执行一次嵌入，或配置显式 dim")
        return self._resolved_dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """调用云端 /embeddings；首次调用推导 dim 并缓存。"""
        client = self._ensure_client()
        response = client.embeddings.create(model=self._model, input=texts)
        vectors = [item.embedding for item in response.data]
        if self._resolved_dim is None and vectors:
            self._resolved_dim = len(vectors[0])
        return vectors

    def _ensure_client(self) -> OpenAI:
        """惰性构造同步 OpenAI client（构造零出网，首次 embed 才建连）。"""
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
