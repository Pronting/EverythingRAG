"""CloudEmbedder 云端嵌入（OpenAI 兼容 /embeddings）测试（fake OpenAI client，零网络）。

本地 fastembed/bge-m3 已移除；嵌入统一走云端。验证：
- 构造零出网（不实例化 client）
- 首次 embed 才建连并推导 dim，之后缓存
- fingerprint 含模型名（斜杠转连字符，隔离 collection）
- 工厂 create_embedder_from_config 只在配置完整时返回 CloudEmbedder
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.vectorstore.embedder as embedder_mod


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


def test_create_embedder_cloud_returns_cloud() -> None:
    """配置完整 -> CloudEmbedder；model 与 fingerprint 正确透传。"""
    from pydantic import SecretStr

    from app.core.settings_store import EmbedModelConfig

    emb = embedder_mod.create_embedder_from_config(
        EmbedModelConfig(base_url="http://e.test/v1", model="bge-m3", api_key=SecretStr("sk-x"))
    )
    assert isinstance(emb, embedder_mod.CloudEmbedder)
    assert emb.model == "bge-m3"
    assert emb.fingerprint == "cloud-bge-m3"


def test_create_embedder_cloud_incomplete_raises() -> None:
    """缺 base_url/model -> ValueError（统一云端后不允许缺省构造）。"""
    from app.core.settings_store import EmbedModelConfig

    with pytest.raises(ValueError):
        embedder_mod.create_embedder_from_config(EmbedModelConfig())
    with pytest.raises(ValueError):
        embedder_mod.create_embedder_from_config(EmbedModelConfig(base_url="http://e.test/v1"))
    with pytest.raises(ValueError):
        embedder_mod.create_embedder_from_config(EmbedModelConfig(model="bge-m3"))
