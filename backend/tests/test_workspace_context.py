"""``WorkspaceContext`` 运行时上下文与缓存键（PRD 5.2 / 5.3.3 / 5.3.4 / 11.5，T01）。

覆盖：
- 由 ``workspace_id`` 验证归属后解析目录（客户端只提交 id，路径由服务端解析）；
- 不可变上下文 + ``WorkspaceServiceHub`` 契约（``view`` + ``app_store``）；
- 向量句柄键（workspace + Provider 摘要 + collection 身份）与索引键
  （workspace + knowledge_revision + 索引指纹 + 生成器版本）；
- 每工作区独立 jieba ``Tokenizer`` 与 ``tmp_dir`` / ``cache_file``（PRD 11.5）；
- ``knowledge_revision`` 递增（PRD 5.3.6）。
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from app.core.workspace_context import (
    BM25_GENERATOR_VERSION,
    CATALOG_GENERATOR_VERSION,
    JIEBA_DICT_VERSION,
    JIEBA_GENERATOR_VERSION,
    JIEBA_TOKENIZER_MODE,
    WorkspaceContextFactory,
    WorkspaceNotFoundError,
    index_cache_key,
    is_service_hub,
    jieba_cache_file,
    jieba_tokenizer,
    vector_handle_cache_key,
)
from app.core.workspace_store import DEFAULT_WORKSPACE_ID, WorkspaceStore


def _factory(tmp_path: Path) -> WorkspaceContextFactory:
    return WorkspaceContextFactory(tmp_path, WorkspaceStore(tmp_path))


def test_resolve_default_and_validate_ownership(tmp_path: Path) -> None:
    factory = _factory(tmp_path)
    context = factory.resolve()
    assert context.workspace_id == DEFAULT_WORKSPACE_ID
    assert context.storage_key

    created = factory.workspace_store.create("研究")
    assert factory.resolve(created.id).workspace_id == created.id

    # PRD 5.1：客户端提交的 id 必须真的存在，且不能静默回落到默认区
    with pytest.raises(WorkspaceNotFoundError):
        factory.resolve("nope")
    with pytest.raises(WorkspaceNotFoundError):
        factory.resolve_storage_key("nope")


def test_context_is_immutable_and_satisfies_service_hub(tmp_path: Path) -> None:
    context = _factory(tmp_path).resolve()
    # frozen + slots 的 dataclass 在 CPython 上抛 TypeError 而非 FrozenInstanceError
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        context.knowledge_revision = 99  # type: ignore[misc]

    # t2 冻结的 WorkspaceServiceHub 成员名：view + app_store
    assert is_service_hub(context)
    assert context.view.workspace_id == context.workspace_id
    assert context.app_store is None  # 未注册时允许为 None（lifespan 负责注入）


def test_with_services_returns_a_new_bound_context(tmp_path: Path) -> None:
    context = _factory(tmp_path).resolve()
    state_store = object()
    bound = context.with_services(state_store=state_store)
    assert bound.state_store is state_store
    assert context.state_store is None  # 原上下文不变（PRD 5.2 末段）
    with pytest.raises(TypeError):
        context.with_services(nope=1)


def test_view_defaults_come_from_the_workspace_record(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path)
    workspace = store.create("研究")
    factory = WorkspaceContextFactory(tmp_path, store)
    view = factory.view_for(workspace.id)
    # PRD 18.1 默认值：ask / high / 记忆开启
    assert view.default_permission_mode.value == "ask"
    assert view.default_reasoning_level.value == "high"
    assert view.memory_enabled is True
    assert view.root_dir == tmp_path / "workspaces" / workspace.storage_key


def test_vector_handle_cache_key_covers_all_three_components() -> None:
    base = vector_handle_cache_key(
        workspace_id="a", provider_digest="p1", collection_identity="chunks__m__v7"
    )
    # PRD 5.3.3：三项任一变化都必须换键，否则会跨区/跨凭证/跨入库版本串读
    assert base != vector_handle_cache_key(
        workspace_id="b", provider_digest="p1", collection_identity="chunks__m__v7"
    )
    assert base != vector_handle_cache_key(
        workspace_id="a", provider_digest="p2", collection_identity="chunks__m__v7"
    )
    assert base != vector_handle_cache_key(
        workspace_id="a", provider_digest="p1", collection_identity="chunks__m__v6"
    )


def test_index_cache_key_covers_workspace_revision_fingerprint_and_generator() -> None:
    base = index_cache_key(
        workspace_id="a",
        knowledge_revision=1,
        index_fingerprint="chunks__m__v7",
        generator_version=BM25_GENERATOR_VERSION,
    )
    assert base != index_cache_key(
        workspace_id="a",
        knowledge_revision=2,
        index_fingerprint="chunks__m__v7",
        generator_version=BM25_GENERATOR_VERSION,
    )
    assert base != index_cache_key(
        workspace_id="a",
        knowledge_revision=1,
        index_fingerprint="chunks__m__v6",
        generator_version=BM25_GENERATOR_VERSION,
    )
    assert base != index_cache_key(
        workspace_id="a",
        knowledge_revision=1,
        index_fingerprint="chunks__m__v7",
        generator_version="bm25-v2",
    )
    with pytest.raises(ValueError):
        index_cache_key(
            workspace_id="a",
            knowledge_revision=-1,
            index_fingerprint="x",
            generator_version="y",
        )


def test_context_cache_keys_use_its_own_revision(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path)
    workspace = store.create("研究")
    factory = WorkspaceContextFactory(tmp_path, store)

    before = factory.resolve(workspace.id)
    first = before.index_cache_key(
        index_fingerprint="chunks__m__v7", generator_version=BM25_GENERATOR_VERSION
    )
    store.bump_knowledge_revision(workspace.id)
    after = factory.resolve(workspace.id)
    second = after.index_cache_key(
        index_fingerprint="chunks__m__v7", generator_version=BM25_GENERATOR_VERSION
    )
    # PRD 5.3.4 / 5.3.6：导入提交后版本递增，索引缓存键必须随之变化
    assert before.knowledge_revision == 0
    assert after.knowledge_revision == 1
    assert first != second
    # PRD 5.3.3 的向量句柄键**不含**知识版本：句柄指向同一个 collection，
    # 版本变化只让内容缓存（索引键）失效。
    assert before.vector_cache_key(provider_digest="p", collection_identity="c") == (
        after.vector_cache_key(provider_digest="p", collection_identity="c")
    )


def test_bump_knowledge_revision_is_bounded_to_known_workspaces(tmp_path: Path) -> None:
    store = WorkspaceStore(tmp_path)
    workspace = store.create("研究")
    assert store.bump_knowledge_revision(workspace.id) == 1
    assert store.bump_knowledge_revision(workspace.id, delta=2) == 3
    assert store.bump_knowledge_revision("nope") is None
    with pytest.raises(ValueError):
        store.bump_knowledge_revision(workspace.id, delta=-1)


def test_jieba_tokenizer_is_per_workspace(tmp_path: Path) -> None:
    """PRD 11.5：每工作区独立 ``Tokenizer``，缓存文件落在本区目录内。"""

    store = WorkspaceStore(tmp_path)
    first = store.create("研究")
    second = store.create("工作")
    factory = WorkspaceContextFactory(tmp_path, store)
    paths_a = factory.paths_for(first.id)
    paths_b = factory.paths_for(second.id)

    cache_a = jieba_cache_file(paths_a)
    cache_b = jieba_cache_file(paths_b)
    assert cache_a != cache_b
    assert cache_a.parent == paths_a.jieba_cache_dir
    assert paths_a.jieba_cache_dir.is_relative_to(paths_a.root_dir)
    # PRD 11.5：cache_file 必须**包含词典版本**（否则会复用按旧词典切好的缓存）
    assert JIEBA_DICT_VERSION in cache_a.name
    assert JIEBA_TOKENIZER_MODE in cache_a.name
    # 生成器版本含「分词模式 + 词典版本」：任一变化都会换掉文本索引缓存键
    assert JIEBA_TOKENIZER_MODE in JIEBA_GENERATOR_VERSION
    assert JIEBA_DICT_VERSION in JIEBA_GENERATOR_VERSION

    tokenizer_a = jieba_tokenizer(paths_a)
    tokenizer_b = jieba_tokenizer(paths_b)
    assert tokenizer_a is not tokenizer_b
    assert tokenizer_a is jieba_tokenizer(paths_a)  # 同区复用
    assert Path(tokenizer_a.cache_file) == cache_a
    assert Path(tokenizer_b.cache_file) == cache_b

    tokens = [token for token in tokenizer_a.cut("知识库隔离测试") if token.strip()]
    assert tokens
    assert JIEBA_GENERATOR_VERSION
    assert CATALOG_GENERATOR_VERSION
