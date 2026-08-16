"""SettingsStore 单测：env 默认 / config.json 覆盖 / 保存往返 / SecretStr。

monkeypatch settings_store.get_settings 控制 env 值，config.json 落在 tmp_path。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from app.core import settings_store as mod
from app.core.config import Settings
from app.core.settings_store import (
    AppSettings,
    AvatarConfig,
    ChatModelConfig,
    EmbedModelConfig,
    SettingsStore,
)


def _env_settings(**overrides) -> Settings:
    base = {
        "chat_base_url": "http://env.test/v1",
        "chat_model": "env-model",
        "chat_api_key": SecretStr("sk-env"),
        "embed_base_url": "http://embed.env/v1",
        "embed_model": "embed-env-model",
        "embed_api_key": SecretStr("sk-embed-env"),
    }
    base.update(overrides)
    return Settings(**base)


def _store_with_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SettingsStore:
    monkeypatch.setattr(mod, "get_settings", lambda: _env_settings())
    return SettingsStore(tmp_path)


def test_load_empty_uses_env_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """无 config.json 时：chat/embed 均回退 env；system_prompt 用内置。"""
    store = _store_with_env(monkeypatch, tmp_path)
    s = store.load()
    assert s.chat.base_url == "http://env.test/v1"
    assert s.chat.model == "env-model"
    assert s.chat.api_key is not None
    assert s.chat.api_key.get_secret_value() == "sk-env"
    assert s.embed.mode == "cloud"
    assert s.embed.base_url == "http://embed.env/v1"
    assert s.embed.model == "embed-env-model"
    assert s.embed.api_key is not None
    assert s.embed.api_key.get_secret_value() == "sk-embed-env"
    assert s.system_prompt == mod.DEFAULT_SYSTEM_PROMPT


def test_save_and_load_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """保存 -> 落盘 -> 重载一致；config.json 明文含 key（本地 gitignored）。"""
    store = _store_with_env(monkeypatch, tmp_path)
    saved = AppSettings(
        chat=ChatModelConfig(base_url="http://file.test/v1", model="file-model", api_key=SecretStr("sk-custom")),
        embed=EmbedModelConfig(mode="cloud", base_url="http://embed.test/v1", model="BAAI/bge-m3", api_key=SecretStr("sk-embed")),
        system_prompt="你是测试助手",
    )
    store.save(saved)

    # 落盘内容
    raw = json.loads(store.path.read_text(encoding="utf-8"))
    assert raw["chat"]["api_key"] == "sk-custom"
    assert raw["embed"]["mode"] == "cloud"
    assert raw["system_prompt"] == "你是测试助手"

    # 重载一致
    reloaded = store.load()
    assert reloaded.chat.base_url == "http://file.test/v1"
    assert reloaded.chat.api_key.get_secret_value() == "sk-custom"
    assert reloaded.embed.mode == "cloud"
    assert reloaded.embed.model == "BAAI/bge-m3"
    assert reloaded.system_prompt == "你是测试助手"


def test_config_authoritative_when_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """config.json 存在即权威：save 全量写入，load 原样读回（不回退 env）。"""
    store = _store_with_env(monkeypatch, tmp_path)
    store.save(AppSettings(chat=ChatModelConfig(base_url="http://file.test/v1")))
    s = store.load()
    assert s.chat.base_url == "http://file.test/v1"
    # 权威语义：未写入的字段按模型默认（None），不再回退 env
    assert s.chat.model is None
    assert s.chat.api_key is None


def test_api_key_explicit_null_clears(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """config.json 里 api_key=null（清除操作落盘）-> load 得到 None，不回退 env key。"""
    store = _store_with_env(monkeypatch, tmp_path)
    store.save(AppSettings(chat=ChatModelConfig(base_url="http://file.test/v1")))  # api_key None 落盘
    raw = json.loads(store.path.read_text(encoding="utf-8"))
    assert raw["chat"]["api_key"] is None
    s = store.load()
    assert s.chat.api_key is None


def test_embed_mode_local_migrates_to_cloud(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """旧 config.json 的 embed.mode=local（本地嵌入已移除）读入时自动迁移为 cloud。"""
    store = _store_with_env(monkeypatch, tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        json.dumps(
            {
                "chat": {"base_url": "http://x.test/v1", "model": "m", "api_key": None},
                "embed": {"mode": "local", "base_url": None, "model": None, "api_key": None},
                "system_prompt": "x",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    s = store.load()
    assert s.embed.mode == "cloud"


def test_missing_or_corrupt_file_returns_env_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """文件缺失或损坏 -> 不崩，回退 env 默认。"""
    store = _store_with_env(monkeypatch, tmp_path)
    assert store.load().chat.base_url == "http://env.test/v1"
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{ not valid json", encoding="utf-8")
    assert store.load().chat.base_url == "http://env.test/v1"


def test_avatar_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """头像相对文件名保存 -> 落盘 -> 重载一致。"""
    store = _store_with_env(monkeypatch, tmp_path)
    store.save(
        AppSettings(avatars=AvatarConfig(user="user-abc.png", agent="agent-xyz.png"))
    )
    raw = json.loads(store.path.read_text(encoding="utf-8"))
    assert raw["avatars"] == {"user": "user-abc.png", "agent": "agent-xyz.png"}
    reloaded = store.load()
    assert reloaded.avatars.user == "user-abc.png"
    assert reloaded.avatars.agent == "agent-xyz.png"


def test_legacy_config_without_avatars_defaults_to_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """旧 config.json 无 avatars 字段 -> 读入为默认（None），不崩。"""
    store = _store_with_env(monkeypatch, tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        json.dumps(
            {
                "chat": {"base_url": "http://x.test/v1", "model": "m", "api_key": None},
                "embed": {"mode": "cloud", "base_url": None, "model": None, "api_key": None},
                "system_prompt": "x",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    s = store.load()
    assert s.avatars.user is None
    assert s.avatars.agent is None
    assert s.theme == "light"  # 旧配置无 theme -> 默认浅色


def test_theme_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """主题保存 -> 落盘 -> 重载一致；默认浅色。"""
    store = _store_with_env(monkeypatch, tmp_path)
    assert store.load().theme == "light"
    store.save(AppSettings(theme="dark"))
    raw = json.loads(store.path.read_text(encoding="utf-8"))
    assert raw["theme"] == "dark"
    assert store.load().theme == "dark"
