"""嵌入器接口与实现：本地 bge-m3（fastembed/ONNX）+ 云端 OpenAI 兼容（/embeddings）。

隐私约束：
- 本地 bge-m3 下载只在首次 embed_texts 惰性触发（用户已授权）；import 本模块、
  创建实例均零副作用。
- 云端嵌入（CloudEmbedder）：构造零出网，首次 embed 才建连（OpenAI 兼容端点）。
  出网即用户显式配置云端嵌入的结果，属 opt-in。
"""

from __future__ import annotations

from typing import Protocol

from fastembed import TextEmbedding
from openai import OpenAI

from app.core.settings_store import EmbedModelConfig

# bge-m3 输出维度（HF 配置 word_embedding_dimension = 1024，CLS 池化）
_BGE_M3_DIM = 1024


class Embedder(Protocol):
    """文本嵌入接口：指纹 + 维度 + 批量嵌入。"""

    fingerprint: str
    dim: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedEmbedder:
    """bge-m3 嵌入器（fastembed TextEmbedding，ONNX）。

    惰性加载：底层模型对象在首次 embed_texts 时创建并缓存，之后复用，
    因此创建实例不触发模型加载或下载。

    内存安全（优化① 提速配套）：fastembed 按 batch 内最长序列 padding，超大
    输入（巨型代码块/配置转储）会让单个 batch 的内存爆炸（曾出现 15.8GB 分配
    失败）。故：
    - max_chars：嵌入前把每条输入截断到上限（缺省 2000 字符，覆盖常规块，仅
      裁剪超长块；不影响块文本在向量库的完整存储）。
    - batch_size：缺省 8——小 batch 既省内存，又避免「大 batch + 长序列」的
      padding 计算浪费（实测 batch=128 反而比 batch=8 慢 3 倍）。
    """

    fingerprint = "bge-m3"
    dim = _BGE_M3_DIM
    max_chars: int = 2000
    batch_size: int = 8

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        cache_dir: str | None = None,
        max_chars: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model_name = model_name
        self._cache_dir = cache_dir
        self._max_chars = max_chars if max_chars is not None else type(self).max_chars
        self._batch_size = batch_size if batch_size is not None else type(self).batch_size
        self._model: TextEmbedding | None = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        bounded = [text[: self._max_chars] for text in texts]
        return [vector.tolist() for vector in model.embed(bounded, batch_size=self._batch_size)]

    def _ensure_model(self) -> TextEmbedding:
        """首次调用才加载模型（触发 bge-m3 下载，已授权）；之后缓存复用。"""
        if self._model is None:
            self._model = _load_text_embedding(self.model_name, self._cache_dir)
        return self._model


def _load_text_embedding(model_name: str, cache_dir: str | None = None) -> TextEmbedding:
    """构造 fastembed TextEmbedding；模型未内置时先注册再构造。

    首次调用触发 bge-m3 的 HF 下载；已缓存则本地直读，不重复下载。
    cache_dir 缺省落到持久目录（data_dir/models），避免模型被下载进
    Windows 临时目录——Temp 会被系统定期清理，2GB+ 模型会被反复重下。
    """
    resolved = cache_dir or _default_cache_dir()
    try:
        return TextEmbedding(model_name=model_name, cache_dir=resolved)
    except ValueError:
        _register_bge_m3_custom(model_name)
        return TextEmbedding(model_name=model_name, cache_dir=resolved)


def _default_cache_dir() -> str:
    """缺省嵌入缓存目录：settings.embed_cache_dir 或 data_dir/models（持久）。"""
    from app.core.config import get_settings

    settings = get_settings()
    embed_dir = settings.embed_cache_dir or settings.data_dir / "models"
    return str(embed_dir)


def _register_bge_m3_custom(model_name: str) -> None:
    """把 bge-m3 注册为 fastembed 自定义 dense 模型（幂等，零出网）。

    fastembed 部分版本（如 0.8）内置注册表不含 bge-m3，需 add_custom_model
    登记其 ONNX 来源（BAAI/bge-m3 的 onnx/model.onnx，CLS 池化 + 归一化）。
    """
    from fastembed.common.model_description import ModelSource, PoolingType

    registered = {model["model"].lower() for model in TextEmbedding.list_supported_models()}
    if model_name.lower() not in registered:
        TextEmbedding.add_custom_model(
            model=model_name,
            pooling=PoolingType.CLS,
            normalization=True,
            sources=ModelSource(hf=model_name),
            dim=_BGE_M3_DIM,
            model_file="onnx/model.onnx",
            description="BGE-M3 multilingual embedding (ONNX)",
            license="mit",
            size_in_gb=2.5,
            additional_files=["onnx/model.onnx_data", "onnx/sentencepiece.bpe.model"],
        )


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
    """按设置构造嵌入器：mode=local -> 本地 bge-m3；mode=cloud -> OpenAI 兼容云端。

    云端配置不完整时 raise ValueError（设置页 PUT 已校验，这里兜底）。
    """
    if embed.mode == "cloud":
        if not embed.base_url or not embed.model:
            raise ValueError("云端嵌入未配置完整：需同时提供 Base URL 与模型名")
        api_key = embed.api_key.get_secret_value() if embed.api_key is not None else None
        return CloudEmbedder(base_url=embed.base_url, model=embed.model, api_key=api_key)
    return FastEmbedEmbedder()
