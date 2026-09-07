"""依赖注册表回归：Provider 配置轮换不得继续复用旧运行时客户端。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from app.api import deps
from app.core.settings_store import AppSettings, EmbedModelConfig


class MutableSettingsStore:
    """仅供测试动态轮换嵌入配置。"""

    def __init__(self, embed: EmbedModelConfig) -> None:
        self.embed = embed

    def load(self) -> AppSettings:
        return AppSettings(embed=self.embed)


@pytest.fixture(autouse=True)
def _clear_runtime_registries() -> None:
    deps._embedder_registry.clear()
    deps._vector_store_registry.clear()
    yield
    deps._embedder_registry.clear()
    deps._vector_store_registry.clear()


def _config(
    *,
    base_url: str = "https://embed-a.test/v1",
    model: str = "same-model",
    api_key: str = "sk-first-secret",
) -> EmbedModelConfig:
    return EmbedModelConfig(
        base_url=base_url,
        model=model,
        api_key=SecretStr(api_key),
    )


@pytest.mark.parametrize(
    "changed",
    [
        _config(base_url="https://embed-b.test/v1"),
        _config(api_key="sk-second-secret"),
        _config(model="another-model"),
    ],
    ids=["base-url", "api-key", "model"],
)
def test_embedder_cache_rotates_on_material_provider_config_change(
    monkeypatch: pytest.MonkeyPatch,
    changed: EmbedModelConfig,
) -> None:
    store = MutableSettingsStore(_config())
    monkeypatch.setattr(deps, "get_settings_store", lambda: store)

    first = deps.get_embedder()
    assert deps.get_embedder() is first

    store.embed = changed
    second = deps.get_embedder()

    assert second is not first
    assert len(deps._embedder_registry) == 1
    registry_keys = repr(tuple(deps._embedder_registry))
    assert "sk-first-secret" not in registry_keys
    assert "sk-second-secret" not in registry_keys


def test_vector_store_wrapper_rotates_with_provider_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同名模型轮换 endpoint 时，向量库包装器也不能继续持有旧 embedder。"""

    settings_store = MutableSettingsStore(_config())
    created: list[SimpleNamespace] = []

    def fake_create_vector_store(*, persist_dir: object, embedder: object) -> SimpleNamespace:
        del persist_dir
        vector_store = SimpleNamespace(embedder=embedder)
        created.append(vector_store)
        return vector_store

    monkeypatch.setattr(deps, "get_settings_store", lambda: settings_store)
    monkeypatch.setattr(deps, "create_vector_store", fake_create_vector_store)

    first = deps.get_vector_store()
    assert deps.get_vector_store() is first

    settings_store.embed = _config(base_url="https://embed-b.test/v1")
    second = deps.get_vector_store()

    assert second is not first
    assert second.embedder is not first.embedder
    assert created == [first, second]
