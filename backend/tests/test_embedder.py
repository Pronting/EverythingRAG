"""CloudEmbedder 云端嵌入（OpenAI 兼容 /embeddings）测试（fake OpenAI client，零网络）。

本地 fastembed/bge-m3 已移除；嵌入统一走云端。验证：
- 构造零出网（不实例化 client）
- 首次 embed 才建连并推导 dim，之后缓存
- fingerprint 含模型名和端点摘要（隔离 collection，不泄露 URL）
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
    assert emb.fingerprint == embedder_mod.embedding_space_fingerprint(
        "http://e.test/v1", "bge-m3"
    )


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
    assert emb.fingerprint == embedder_mod.embedding_space_fingerprint("http://e.test/v1", "m")
    with pytest.raises(RuntimeError):
        emb.dim  # noqa: B018 -- 未 embed，dim 未解析（故意触发）


def test_cloud_embedder_model_name_normalized_in_fingerprint() -> None:
    """指纹把 / 转 -：模型名含斜杠也不破坏 collection 名。"""
    emb = embedder_mod.CloudEmbedder(base_url="http://e.test/v1", model="BAAI/bge-m3")
    assert emb.fingerprint == embedder_mod.embedding_space_fingerprint(
        "http://e.test/v1", "BAAI/bge-m3"
    )


def test_cloud_embedder_fingerprint_separates_endpoints_but_not_credentials() -> None:
    left = embedder_mod.CloudEmbedder(
        base_url="https://one.example/v1/", model="bge-m3", api_key="first"
    )
    same_space = embedder_mod.CloudEmbedder(
        base_url="https://one.example/v1", model="bge-m3", api_key="second"
    )
    other_endpoint = embedder_mod.CloudEmbedder(
        base_url="https://two.example/v1", model="bge-m3", api_key="first"
    )

    assert left.fingerprint == same_space.fingerprint
    assert left.fingerprint != other_endpoint.fingerprint
    assert "example" not in left.fingerprint
    assert "first" not in left.fingerprint


def test_qwen3_query_gets_official_instruction_documents_stay_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qwen3 instruction is query-only, so enabling it never changes indexed documents."""

    fake = _FakeCloudClient(dim=4)
    monkeypatch.setattr(embedder_mod, "OpenAI", lambda **kwargs: fake)
    emb = embedder_mod.CloudEmbedder(
        base_url="http://e.test/v1", model="Qwen/Qwen3-Embedding-8B"
    )

    emb.embed_documents(["购物车分表方案"])
    emb.embed_query("京东和淘宝的分表对比")

    expected_query = (
        "Instruct: Given a user query, retrieve relevant passages that answer the query\n"
        "Query:京东和淘宝的分表对比"
    )
    assert fake.calls[0] == ("Qwen/Qwen3-Embedding-8B", ["购物车分表方案"])
    assert fake.calls[1] == (
        "Qwen/Qwen3-Embedding-8B",
        [expected_query],
    )


def test_non_qwen_query_is_not_reformatted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown embedding models use the safe OpenAI-compatible raw-query default."""

    fake = _FakeCloudClient(dim=4)
    monkeypatch.setattr(embedder_mod, "OpenAI", lambda **kwargs: fake)
    emb = embedder_mod.CloudEmbedder(base_url="http://e.test/v1", model="text-embedding-3-large")

    emb.embed_queries(["first", "second"])
    assert fake.calls == [("text-embedding-3-large", ["first", "second"])]


def test_empty_embedding_batch_does_not_create_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty ingestion/query batches are zero-I/O and return immediately."""

    def _boom(**kwargs: object) -> None:
        raise AssertionError("empty batch must not instantiate OpenAI")

    monkeypatch.setattr(embedder_mod, "OpenAI", _boom)
    emb = embedder_mod.CloudEmbedder(base_url="http://e.test/v1", model="qwen3-embedding-8b")
    assert emb.embed_documents([]) == []
    assert emb.embed_queries([]) == []


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
    assert emb.fingerprint == embedder_mod.embedding_space_fingerprint(
        "http://e.test/v1", "bge-m3"
    )


def test_create_embedder_cloud_incomplete_raises() -> None:
    """缺 base_url/model -> ValueError（统一云端后不允许缺省构造）。"""
    from app.core.settings_store import EmbedModelConfig

    with pytest.raises(ValueError):
        embedder_mod.create_embedder_from_config(EmbedModelConfig())
    with pytest.raises(ValueError):
        embedder_mod.create_embedder_from_config(EmbedModelConfig(base_url="http://e.test/v1"))
    with pytest.raises(ValueError):
        embedder_mod.create_embedder_from_config(EmbedModelConfig(model="bge-m3"))
