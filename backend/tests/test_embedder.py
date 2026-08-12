"""FastEmbedEmbedder 惰性加载与注册回退测试（fake TextEmbedding，零网络）。

隐私红线：全部用 fake 替代 fastembed.TextEmbedding，不触发任何真实模型
加载/下载/出网；验证「创建实例零副作用、首次 embed 才加载、之后缓存」。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import app.vectorstore.embedder as embedder_mod
from app.vectorstore.embedder import FastEmbedEmbedder


class FakeTextEmbedding:
    """记录实例化次数与最近 cache_dir；embed 返回 dim=1024 的伪向量。"""

    instances = 0
    last_cache_dir: str | None = None

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        type(self).instances += 1
        self.model_name = model_name
        self.cache_dir = cache_dir
        type(self).last_cache_dir = cache_dir

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

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        type(self).instances += 1
        if not type(self).registered:
            raise ValueError(f"Model {model_name} is not supported")
        self.model_name = model_name
        self.cache_dir = cache_dir

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


def test_cache_dir_defaults_to_persistent_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """缺省缓存目录落到 data_dir/models（持久），不落在系统临时目录。

    修复：fastembed 默认把模型下载进 tempfile.gettempdir()/fastembed_cache，
    Windows 会定期清理 Temp 导致 2GB+ 模型反复重下。
    """
    import app.core.config as config_mod

    class _FakeSettings:
        embed_cache_dir = None
        data_dir = Path("C:/fake-data")

    monkeypatch.setattr(config_mod, "get_settings", lambda: _FakeSettings())
    assert embedder_mod._default_cache_dir() == str(Path("C:/fake-data") / "models")


def test_cache_dir_passed_through_to_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式 cache_dir 透传给底层 TextEmbedding。"""
    monkeypatch.setattr(embedder_mod, "TextEmbedding", FakeTextEmbedding)
    embedder = FastEmbedEmbedder(cache_dir="C:/persistent/models")
    embedder.embed_texts(["text"])
    assert FakeTextEmbedding.last_cache_dir == "C:/persistent/models"


# ---------------------------------------------------------------- 云端嵌入（OpenAI 兼容）


class _FakeEmbeddings:
    """OpenAI client.embeddings 对象：create 返回定长伪向量并记录调用。"""

    def __init__(self, client: _FakeCloudClient, dim: int) -> None:
        self._client = client
        self._dim = dim

    def create(self, model: str, input: list[str]) -> SimpleNamespace:
        self._client.calls.append((model, list(input)))
        vectors = [[float(i % 5) for i in range(self._dim)] for _ in input]
        return SimpleNamespace(data=[SimpleNamespace(embedding=v) for v in vectors])


class _FakeCloudClient:
    """模拟 openai sync client：embeddings 为属性对象（含 .create）。"""

    def __init__(self, dim: int = 4) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.embeddings = _FakeEmbeddings(self, dim)


def test_cloud_embedder_resolves_dim_from_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """首次 embed 从响应推导 dim；向量批量返回；指纹含模型名。"""
    fake = _FakeCloudClient(dim=4)
    monkeypatch.setattr(embedder_mod, "OpenAI", lambda **kwargs: fake)
    emb = embedder_mod.CloudEmbedder(base_url="http://e.test/v1", model="bge-m3")

    vectors = emb.embed_texts(["a", "b"])
    assert emb.dim == 4  # 从响应推导
    assert len(vectors) == 2
    assert all(len(v) == 4 for v in vectors)
    assert fake.calls == [("bge-m3", ["a", "b"])]
    assert emb.fingerprint == "cloud-bge-m3"


def test_cloud_embedder_configured_dim_no_embed_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式给 dim -> 未 embed 也可读；构造不实例化 OpenAI。"""

    def _boom(**kwargs: object) -> None:
        raise AssertionError("OpenAI 不应在构造时实例化")

    monkeypatch.setattr(embedder_mod, "OpenAI", _boom)
    emb = embedder_mod.CloudEmbedder(base_url="http://e.test/v1", model="m", dim=1024)
    assert emb.dim == 1024


def test_cloud_embedder_construction_zero_outbound(monkeypatch: pytest.MonkeyPatch) -> None:
    """构造零副作用：不实例化 client、不出网；dim 未解析时读它抛错。"""

    def _boom(**kwargs: object) -> None:
        raise AssertionError("OpenAI 不应在构造时实例化")

    monkeypatch.setattr(embedder_mod, "OpenAI", _boom)
    emb = embedder_mod.CloudEmbedder(base_url="http://e.test/v1", model="m")
    assert emb.fingerprint == "cloud-m"
    with pytest.raises(RuntimeError):
        emb.dim  # noqa: B018 -- 未 embed，dim 未解析（故意触发）


def test_cloud_embedder_model_name_normalized_in_fingerprint() -> None:
    """指纹把 / 转 -：模型名含斜杠也不破坏 collection 名。"""
    emb = embedder_mod.CloudEmbedder(base_url="http://e.test/v1", model="BAAI/bge-m3")
    assert emb.fingerprint == "cloud-BAAI-bge-m3"


# ---------------------------------------------------------------- 嵌入器工厂（按设置）


def test_create_embedder_local_returns_fastembed() -> None:
    """mode=local -> FastEmbedEmbedder（本地 bge-m3）。"""
    from app.core.settings_store import EmbedModelConfig

    emb = embedder_mod.create_embedder_from_config(EmbedModelConfig(mode="local"))
    assert isinstance(emb, FastEmbedEmbedder)


def test_create_embedder_cloud_returns_cloud() -> None:
    """mode=cloud 且配置完整 -> CloudEmbedder。"""
    from pydantic import SecretStr

    from app.core.settings_store import EmbedModelConfig

    emb = embedder_mod.create_embedder_from_config(
        EmbedModelConfig(mode="cloud", base_url="http://e.test/v1", model="bge-m3", api_key=SecretStr("sk-x"))
    )
    assert isinstance(emb, embedder_mod.CloudEmbedder)
    assert emb.model == "bge-m3"
    assert emb.fingerprint == "cloud-bge-m3"


def test_create_embedder_cloud_incomplete_raises() -> None:
    """mode=cloud 但缺 base_url/model -> ValueError。"""
    from app.core.settings_store import EmbedModelConfig

    with pytest.raises(ValueError):
        embedder_mod.create_embedder_from_config(EmbedModelConfig(mode="cloud"))
