"""FastEmbedEmbedder 惰性加载与注册回退测试（fake TextEmbedding，零网络）。

隐私红线：全部用 fake 替代 fastembed.TextEmbedding，不触发任何真实模型
加载/下载/出网；验证「创建实例零副作用、首次 embed 才加载、之后缓存」。
"""

from __future__ import annotations

import numpy as np
import pytest

import app.vectorstore.embedder as embedder_mod
from app.vectorstore.embedder import FastEmbedEmbedder


class FakeTextEmbedding:
    """记录实例化次数；embed 返回 dim=1024 的伪向量。"""

    instances = 0

    def __init__(self, model_name: str) -> None:
        type(self).instances += 1
        self.model_name = model_name

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        return [np.zeros(FastEmbedEmbedder.dim, dtype=np.float32) for _ in texts]


def test_constants() -> None:
    """指纹与维度常量：bge-m3 / 1024；collection 命名与维度校验依赖它们。"""
    assert FastEmbedEmbedder.fingerprint == "bge-m3"
    assert FastEmbedEmbedder.dim == 1024
    assert FastEmbedEmbedder().model_name == "BAAI/bge-m3"


def test_creation_does_not_instantiate_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """创建实例零副作用：不加载、不下载、不实例化底层模型。"""
    monkeypatch.setattr(embedder_mod, "TextEmbedding", FakeTextEmbedding)
    FastEmbedEmbedder()
    assert FakeTextEmbedding.instances == 0


def test_lazy_load_once_then_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """惰性加载：首次 embed_texts 实例化一次，之后缓存复用。"""
    monkeypatch.setattr(embedder_mod, "TextEmbedding", FakeTextEmbedding)
    embedder = FastEmbedEmbedder()
    vectors = embedder.embed_texts(["第一条", "第二条"])
    assert FakeTextEmbedding.instances == 1
    assert len(vectors) == 2
    assert all(len(vector) == FastEmbedEmbedder.dim for vector in vectors)

    embedder.embed_texts(["第三条"])
    assert FakeTextEmbedding.instances == 1  # 缓存，不重复实例化


class FakeUnregisteredTextEmbedding:
    """模拟「模型未内置」：首次构造抛 ValueError，注册后第二次构造成功。"""

    instances = 0
    registered = False

    def __init__(self, model_name: str) -> None:
        type(self).instances += 1
        if not type(self).registered:
            raise ValueError(f"Model {model_name} is not supported")
        self.model_name = model_name

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        return [np.zeros(FastEmbedEmbedder.dim, dtype=np.float32) for _ in texts]

    @classmethod
    def list_supported_models(cls) -> list[dict]:
        return []

    @classmethod
    def add_custom_model(cls, **kwargs: object) -> None:
        cls.registered = True


def test_register_fallback_when_model_not_bundled(monkeypatch: pytest.MonkeyPatch) -> None:
    """fastembed 未内置 bge-m3 时：构造抛错 -> add_custom_model 注册 -> 重试成功。"""
    monkeypatch.setattr(embedder_mod, "TextEmbedding", FakeUnregisteredTextEmbedding)
    embedder = FastEmbedEmbedder()
    vectors = embedder.embed_texts(["text"])
    assert FakeUnregisteredTextEmbedding.instances == 2
    assert FakeUnregisteredTextEmbedding.registered is True
    assert len(vectors) == 1
    assert len(vectors[0]) == FastEmbedEmbedder.dim
