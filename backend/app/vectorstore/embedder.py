"""嵌入器接口与 bge-m3 实现（fastembed / ONNX / 惰性加载）。

隐私约束：bge-m3 模型下载只在首次 embed_texts 时惰性触发（用户已授权）；
import 本模块、创建 FastEmbedEmbedder 实例均零副作用（不加载、不下载、不出网）。
"""

from __future__ import annotations

from typing import Protocol

from fastembed import TextEmbedding

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
    """

    fingerprint = "bge-m3"
    dim = _BGE_M3_DIM

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self.model_name = model_name
        self._model: TextEmbedding | None = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        return [vector.tolist() for vector in model.embed(texts)]

    def _ensure_model(self) -> TextEmbedding:
        """首次调用才加载模型（触发 bge-m3 下载，已授权）；之后缓存复用。"""
        if self._model is None:
            self._model = _load_text_embedding(self.model_name)
        return self._model


def _load_text_embedding(model_name: str) -> TextEmbedding:
    """构造 fastembed TextEmbedding；模型未内置时先注册再构造。

    首次调用触发 bge-m3 的 HF 下载；已缓存则本地直读，不重复下载。
    """
    try:
        return TextEmbedding(model_name=model_name)
    except ValueError:
        _register_bge_m3_custom(model_name)
        return TextEmbedding(model_name=model_name)


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
