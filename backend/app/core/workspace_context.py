"""工作区不可变上下文、目录解析与缓存键（Owner: T01；知识库布局见 D-KB）。

职责
----
1. **目录解析**：由 ``workspace_id`` 验证归属 → ``storage_key`` → 目录树。客户端只
   提交 ``workspace_id``，**路径永远由服务端解析**（PRD 5.1）。
2. **不可变上下文**：:class:`WorkspaceContext` 持有数据视图（身份 / 目录 / 知识版本 /
   工作区默认设置）与**服务实例**，并实现 ``contracts.WorkspaceServiceHub``
   （``view`` + ``app_store`` 两个成员名由 t2 冻结）。
3. **布局（D-KB，2026-09-20 用户决策，取代 PRD 5.1 的「每工作区知识库」）**：
   **所有工作区共享同一个知识库** —— 知识类路径一律落在 ``data_dir`` 根：文档
   （``documents/``）、Chroma（``data_dir`` 本身）、增量同步指纹库（``state.db``）、
   图片任务库（``image_state.db``）。工作区只保留**私有**件：``memory.json``、
   ``checkpoints.db``、``artifacts/``、jieba 缓存（含历史检索的术语词典与索引标记）。
   判定**不再**依赖 ``data_dir/storage.json``：该标记现在只记录工作区记录
   （JSON ↔ app.db 权威，D25/D29）的迁移状态，**改不动知识库布局**。
4. **缓存键**：向量句柄与 BM25 / 目录 / 术语缓存键的唯一构造函数 —— 只此一处。
   知识库共享后，向量句柄键**不再**含 ``workspace_id``：同一个 ``data_dir`` 只应存在
   **一个**句柄实例。
5. **jieba 工厂**：每工作区独立 ``Tokenizer``，``tmp_dir`` / ``cache_file`` 落在本区
    ``cache/jieba/``（PRD 11.5）。历史检索的内容实现由 T08 负责，这里只提供工厂。
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from app.agent.contracts import PermissionMode, ReasoningLevel, WorkspaceServiceHub, WorkspaceStatus
from app.agent.contracts import WorkspaceContext as WorkspaceContextView
from app.core.workspace_store import (
    DEFAULT_WORKSPACE_ID,
    WORKSPACE_LAYOUT_VERSION,
    WORKSPACE_SUBDIRS,
    Workspace,
    WorkspaceStore,
    workspace_from_record,
)

__all__ = [
    "BM25_GENERATOR_VERSION",
    "CATALOG_GENERATOR_VERSION",
    "JIEBA_DICT_VERSION",
    "JIEBA_GENERATOR_VERSION",
    "JIEBA_TOKENIZER_MODE",
    "SHARED_KNOWLEDGE_DIRS",
    "STORAGE_MARKER_FILENAME",
    "AppDbWorkspaceReader",
    "BackgroundTaskScope",
    "LayoutMode",
    "WorkspaceContext",
    "WorkspaceContextFactory",
    "WorkspaceNotFoundError",
    "WorkspacePaths",
    "commit_storage_version",
    "committed_storage_version",
    "ensure_layout_dirs",
    "index_cache_key",
    "is_service_hub",
    "jieba_cache_file",
    "jieba_tokenizer",
    "read_storage_marker",
    "vector_handle_cache_key",
]

#: 应用级布局标记（PRD 15.2 第 10 步提交 ``storage_schema_version``）。
STORAGE_MARKER_FILENAME = "storage.json"

#: 缓存键里的「生成器版本」（PRD 5.3.4）：算法或产物结构变化时必须递增，
#: 否则旧缓存会被新实现继续命中。
BM25_GENERATOR_VERSION = "bm25-v1"
CATALOG_GENERATOR_VERSION = "catalog-v1"
#: 历史检索的 jieba 分词模式（PRD 11.5）：导入与查询**必须**用同一个
#: ``cut_for_search()`` 配置。``cut("固定快照")`` 只给整词，而
#: ``cut_for_search`` 还会给 ``固定`` / ``快照`` —— 两者不一致时，按查询侧建立的
#: 子词项在导入侧建立的 FTS5 索引里不存在，历史搜索会整体查不中（A22 / A26）。
#: 所以这里是**唯一定义**，T08 的查询侧同样取它，不要各写一份。
JIEBA_TOKENIZER_MODE = "cut_for_search"
#: 术语词典版本（PRD 11.5）：词典变化后只重建相应文本索引，且 ``cache_file``
#: 必须包含该版本，避免复用过期的分词缓存。
JIEBA_DICT_VERSION = "v1"
#: 分词器 + 模式 + 词典 的生成器版本：任一变化都会换掉文本索引缓存键
#: （PRD 11.5「分词器版本或术语变化后只重建相应文本索引」）。
JIEBA_GENERATOR_VERSION = (
    f"jieba-0.42.1-{JIEBA_TOKENIZER_MODE}-{JIEBA_DICT_VERSION}"
)


class WorkspaceNotFoundError(LookupError):
    """``workspace_id`` 不存在或其归属无法验证（PRD 5.1：客户端只提交 id）。"""


#: 同步**只读**读取 ``app.db`` 权威工作区行的接缝（由 ``app/api/deps.py`` 提供：它知道
#: ``data_dir/app.db`` 的位置，core 层不重复拼路径）。返回同名字段的 mapping；
#: 任何失败都返回空序列 —— 水合是尽力而为，失败时存档保持原样。
AppDbWorkspaceReader = Callable[[], Sequence[Mapping[str, Any]]]


class LayoutMode(StrEnum):
    """本区实际使用的目录布局（D-KB 后只剩一种）。"""

    #: 知识库**全局共享**：知识数据直接放在 ``data_dir`` 根，所有工作区读写同一份；
    #: 工作区私有件（记忆 / 检查点 / 工件 / jieba 缓存）在
    #: ``data_dir/workspaces/<storage_key>/``。
    SHARED = "shared"


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# 布局标记（storage.json）
# ---------------------------------------------------------------------------


def read_storage_marker(data_dir: Path) -> dict[str, Any]:
    """读取 ``data_dir/storage.json``；缺失或损坏返回空 dict（按未迁移处理）。"""

    path = Path(data_dir) / STORAGE_MARKER_FILENAME
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def committed_storage_version(data_dir: Path) -> int:
    """返回已提交的 ``storage_schema_version``；未迁移为 0。"""

    value = read_storage_marker(data_dir).get("storage_schema_version", 0)
    return value if isinstance(value, int) and value >= 0 else 0


def commit_storage_version(
    data_dir: Path,
    *,
    version: int = WORKSPACE_LAYOUT_VERSION,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """原子提交 ``storage_schema_version``（PRD 15.2 第 10 步，校验通过后调用）。

    D-KB 之后本标记**只**表达「工作区记录已迁移到 app.db 权威（D25/D29）」，
    **不再**影响知识库布局（知识永远在 ``data_dir`` 根、所有工作区共享）。
    """

    base = Path(data_dir)
    base.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "storage_schema_version": version,
        "layout": LayoutMode.SHARED.value,
        "committed_at": _now_iso(),
    }
    payload.update(dict(extra or {}))
    path = base / STORAGE_MARKER_FILENAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _legacy_layout_present(data_dir: Path) -> bool:
    """``data_dir`` 根下是否已有知识数据（D-KB 后仅用于**诊断/测试**，不再判定布局）。

    知识永远在 ``data_dir`` 根，所以本函数不再参与任何解析分支；保留它是因为
    「这个目录里到底有没有既有知识」仍是导入/迁移报告与排障要回答的问题。
    """

    base = Path(data_dir)
    for name in ("chroma.sqlite3", "state.db", "image_state.db"):
        if (base / name).is_file():
            return True
    documents = base / "documents"
    try:
        return documents.is_dir() and any(documents.iterdir())
    except OSError:
        return False


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------


#: 知识类目录（**全局共享**）：全部在 ``data_dir`` 根，任何工作区读写同一份。
SHARED_KNOWLEDGE_DIRS: tuple[str, ...] = ("documents", "cache/jieba")


@dataclass(frozen=True, slots=True)
class WorkspacePaths:
    """本区**实际读写**使用的路径（D-KB：知识库全局共享）。

    分两类：

    - **共享知识路径**（``documents_dir`` / ``state_db`` / ``image_state_db`` /
      ``chroma_dir``）：永远在 ``data_dir`` 根，所有工作区解析结果**逐字段相同**；
    - **工作区私有路径**（``memory_json`` / ``checkpoints_db`` / ``artifacts_dir`` /
      ``jieba_cache_dir``）：在 ``root_dir`` = ``data_dir/workspaces/<storage_key>/``。

    所有服务句柄一律以本对象为准（``path`` 与 ``view`` 的派生属性必须一致）。
    """

    layout: LayoutMode
    data_dir: Path
    #: 本区私有根：``data_dir/workspaces/<storage_key>/``（记忆 / 检查点 / 工件 / 分词缓存）。
    root_dir: Path
    #: 共享知识：上传/导入的原始 Markdown 规范根。
    documents_dir: Path
    #: 共享知识：增量同步指纹库。
    state_db: Path
    #: 共享知识：图片任务 + 内容去重库。
    image_state_db: Path
    #: 本区私有：工作区记忆文件。
    memory_json: Path
    #: 本区私有：LangGraph 运行检查点库。
    checkpoints_db: Path
    #: 本区私有：运行产物。
    artifacts_dir: Path
    #: 共享知识：Chroma ``PersistentClient`` 的 persist 根 = ``data_dir`` 本身
    #: （``chroma.sqlite3`` 与全部分段目录都在根上）。
    chroma_dir: Path
    #: 本区私有：jieba 分词缓存 + 历史检索术语词典与索引标记。
    jieba_cache_dir: Path


def _shared_paths(data_dir: Path, storage_key: str) -> WorkspacePaths:
    """构造某个 ``storage_key`` 的路径对象：知识共享 + 私有件按区。"""

    base = Path(data_dir)
    root = base / "workspaces" / storage_key
    return WorkspacePaths(
        layout=LayoutMode.SHARED,
        data_dir=base,
        root_dir=root,
        documents_dir=base / "documents",
        state_db=base / "state.db",
        image_state_db=base / "image_state.db",
        memory_json=root / "memory.json",
        checkpoints_db=root / "checkpoints.db",
        artifacts_dir=root / "artifacts",
        chroma_dir=base,
        jieba_cache_dir=root / "cache" / "jieba",
    )


def ensure_layout_dirs(paths: WorkspacePaths) -> WorkspacePaths:
    """创建共享知识目录与本区私有目录（幂等，已有文件一律不动）。"""

    for relative in SHARED_KNOWLEDGE_DIRS:
        (paths.data_dir / relative).mkdir(parents=True, exist_ok=True)
    for relative in WORKSPACE_SUBDIRS:
        (paths.root_dir / relative).mkdir(parents=True, exist_ok=True)
    return paths


# ---------------------------------------------------------------------------
# 缓存键
# ---------------------------------------------------------------------------


def vector_handle_cache_key(
    *,
    data_dir: Path | str,
    provider_digest: str,
    collection_identity: str,
) -> str:
    """向量句柄缓存键：共享数据根 + Provider 配置摘要 + collection 身份。

    D-KB（知识库全局共享）后**刻意不含** ``workspace_id``：语料只有一份，
    同一个 ``data_dir`` 里应当只存在**一个** Chroma 句柄实例；再按区建句柄会在同一
    persist 根上开出多个 ``PersistentClient``（并发写句柄互踩的已知崩溃源）。
    含 ``data_dir`` 是因为句柄绑定 persist 根：换数据目录（测试隔离、多实例）后
    旧句柄绝不能被复用；缺摘要会在换凭证/端点后继续用旧客户端，缺 collection 身份
    会在入库版本变化后继续读旧 ANN 段 —— 这两项同样是必需成分。
    """

    return f"vec-v3:{Path(data_dir)}:{provider_digest}:{collection_identity}"


def index_cache_key(
    *,
    workspace_id: str,
    knowledge_revision: int,
    index_fingerprint: str,
    generator_version: str,
) -> str:
    """BM25 / 目录 / 术语缓存键（PRD 5.3.4）：工作区 + 知识版本 + 索引指纹 + 生成器版本。

    内容缓存因此**不可能**只以问题文本、URL、文件名或图片 hash 作为全局键
    （PRD 5.3.7）：这四项里任何一项变化都会换键。
    """

    if knowledge_revision < 0:
        raise ValueError("knowledge_revision 不能为负")
    return (
        f"idx-v2:{workspace_id}:{knowledge_revision}:{index_fingerprint}:{generator_version}"
    )


# ---------------------------------------------------------------------------
# jieba（PRD 11.5）
# ---------------------------------------------------------------------------


_tokenizers: dict[str, Any] = {}
_tokenizers_lock = threading.RLock()


def jieba_cache_file(paths: WorkspacePaths) -> Path:
    """本区 jieba 缓存文件路径。

    PRD 11.5 要求 ``cache_file`` **包含词典版本**：词典或分词模式变化后必须换
    缓存文件，否则会复用按旧配置切好的缓存。
    """

    return paths.jieba_cache_dir / (
        f"jieba.{JIEBA_TOKENIZER_MODE}.{JIEBA_DICT_VERSION}.cache"
    )


def jieba_tokenizer(paths: WorkspacePaths) -> Any:
    """返回本区独立的 ``jieba.Tokenizer``（同一目录复用同一实例）。

    ``tmp_dir`` 与 ``cache_file`` 都指向本区 ``cache/jieba/``：不同工作区的词频缓存
    互不可见，也不会污染进程级默认缓存目录（PRD 5.3.5「缓存限定在本区数据目录」）。
    """

    cache_dir = paths.jieba_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = str(cache_dir.resolve())
    with _tokenizers_lock:
        cached = _tokenizers.get(key)
        if cached is not None:
            return cached
        import jieba  # 局部导入：jieba 首次导入约 1 秒，只在真正需要分词时付这个代价

        tokenizer = jieba.Tokenizer()
        tokenizer.tmp_dir = str(cache_dir)
        tokenizer.cache_file = str(jieba_cache_file(paths))
        tokenizer.initialize()
        _tokenizers[key] = tokenizer
        return tokenizer


# ---------------------------------------------------------------------------
# 运行时上下文
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorkspaceContext:
    """PRD 5.2 的每工作区**不可变**上下文：数据视图 + 目录 + 服务实例。

    不可变的意义：后台任务在创建时绑定本对象，用户之后切换界面工作区不会让正在
    运行的任务改用别的知识库（PRD 5.2 末段）。工具执行器从这里取 ``workspace_id``，
    模型可见的工具参数里不出现该字段。
    """

    view: WorkspaceContextView
    paths: WorkspacePaths
    app_store: Any = None
    state_store: Any = None
    image_store: Any = None
    vector_store: Any = None
    bm25: Any = None
    catalog: Any = None
    memory: Any = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    # ---- 身份 ----

    @property
    def workspace_id(self) -> str:
        return self.view.workspace_id

    @property
    def storage_key(self) -> str:
        return self.view.storage_key

    @property
    def name(self) -> str:
        return self.view.name

    @property
    def status(self) -> WorkspaceStatus:
        return self.view.status

    @property
    def layout(self) -> LayoutMode:
        return self.paths.layout

    # ---- 知识版本与默认设置（PRD 5.3.6 / 18.1）----

    @property
    def knowledge_revision(self) -> int:
        return self.view.knowledge_revision

    @property
    def default_permission_mode(self) -> PermissionMode:
        return self.view.default_permission_mode

    @property
    def default_reasoning_level(self) -> ReasoningLevel:
        return self.view.default_reasoning_level

    @property
    def memory_enabled(self) -> bool:
        return self.view.memory_enabled

    # ---- 目录（PRD 5.1，一律取实际布局）----

    @property
    def documents_dir(self) -> Path:
        return self.paths.documents_dir

    @property
    def state_db(self) -> Path:
        return self.paths.state_db

    @property
    def image_state_db(self) -> Path:
        return self.paths.image_state_db

    @property
    def chroma_dir(self) -> Path:
        return self.paths.chroma_dir

    @property
    def memory_json(self) -> Path:
        return self.paths.memory_json

    @property
    def checkpoints_db(self) -> Path:
        return self.paths.checkpoints_db

    @property
    def artifacts_dir(self) -> Path:
        return self.paths.artifacts_dir

    # ---- 缓存键 ----

    def vector_cache_key(self, *, provider_digest: str, collection_identity: str) -> str:
        """共享知识库的向量句柄键（D-KB：共享数据根 + Provider 摘要 + collection 身份）。

        因此所有工作区拿到**同一个**键（同一个 ``data_dir`` 只应有一个句柄实例）——
        这正是「共享一个知识库」的落点。
        """

        return vector_handle_cache_key(
            data_dir=self.paths.data_dir,
            provider_digest=provider_digest,
            collection_identity=collection_identity,
        )

    def index_cache_key(self, *, index_fingerprint: str, generator_version: str) -> str:
        """本区当前知识版本下的索引缓存键（PRD 5.3.4）。"""

        return index_cache_key(
            workspace_id=self.workspace_id,
            knowledge_revision=self.knowledge_revision,
            index_fingerprint=index_fingerprint,
            generator_version=generator_version,
        )

    def jieba(self) -> Any:
        """返回本区独立的 jieba 分词器（PRD 11.5）。"""

        return jieba_tokenizer(self.paths)

    def with_services(self, **services: Any) -> WorkspaceContext:
        """返回替换了部分服务句柄的新上下文（保持不可变，原对象不变）。"""

        unknown = set(services) - {
            "app_store",
            "state_store",
            "image_store",
            "vector_store",
            "bm25",
            "catalog",
            "memory",
        }
        if unknown:
            raise TypeError(f"未知的服务句柄：{sorted(unknown)}")
        fields: dict[str, Any] = {
            "app_store": self.app_store,
            "state_store": self.state_store,
            "image_store": self.image_store,
            "vector_store": self.vector_store,
            "bm25": self.bm25,
            "catalog": self.catalog,
            "memory": self.memory,
        }
        fields.update(services)
        return WorkspaceContext(
            view=self.view,
            paths=self.paths,
            extras=self.extras,
            **fields,
        )


@dataclass(frozen=True, slots=True)
class BackgroundTaskScope:
    """后台任务在**创建时**绑定的工作区（PRD 5.2 末段 / A08）。

    「用户切换界面工作区后，已经运行的任务继续使用原来的上下文」这条只有在任务持有
    创建时的上下文（而不是每次去重新 ``resolve``）时才成立。任务只持有本对象，
    因此导入 / 识图 / 研究任务不可能中途改读别的知识库。
    """

    workspace_id: str
    storage_key: str
    context: WorkspaceContext
    created_at: str = ""

    @classmethod
    def bind(cls, context: WorkspaceContext) -> BackgroundTaskScope:
        return cls(
            workspace_id=context.workspace_id,
            storage_key=context.storage_key,
            context=context,
            created_at=_now_iso(),
        )

    @property
    def state_db(self) -> Path:
        return self.context.state_db

    @property
    def image_state_db(self) -> Path:
        return self.context.image_state_db

    @property
    def documents_dir(self) -> Path:
        return self.context.documents_dir

    @property
    def knowledge_revision(self) -> int:
        return self.context.knowledge_revision


class WorkspaceServiceFactory(Protocol):
    """由 ``app/api/deps.py`` 提供的服务句柄工厂（按上下文构造句柄）。"""

    def __call__(self, context: WorkspaceContext) -> Any: ...


def is_service_hub(context: object) -> bool:
    """对象是否满足 t2 冻结的 ``WorkspaceServiceHub`` 契约（``view`` + ``app_store``）。

    T02（RunManager）与工具执行器据此复用同一个上下文对象，不需要 import 具体实现。
    """

    return isinstance(context, WorkspaceServiceHub)


class WorkspaceContextFactory:
    """由 ``workspace_id`` 解析路径并构造 :class:`WorkspaceContext`。

    - 归属校验：非 ServiceHub 的调用方（HTTP 依赖、后台任务）一律走本工厂，因此
      「客户端提交的 id 是否真的存在」只有一处判定。
    - 布局：见 :class:`LayoutMode` —— D-KB 之后**只有共享布局**：知识路径与
      ``storage.json`` 无关，所有工作区逐字段相同；工作区只影响私有件路径。
    """

    def __init__(
        self,
        data_dir: Path,
        workspace_store: WorkspaceStore,
        *,
        app_store: Any = None,
        app_db_reader: AppDbWorkspaceReader | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._workspaces = workspace_store
        self._app_store = app_store
        self._app_db_reader = app_db_reader
        self._archive_checked = False

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def workspace_store(self) -> WorkspaceStore:
        return self._workspaces

    def default_workspace_id(self) -> str:
        """未指定工作区时的归属：``default``（PRD 5.1 / 15.1）。"""

        return DEFAULT_WORKSPACE_ID

    # ---- 归属校验 ----

    def _ensure_archive_fresh(self) -> None:
        """每个工厂实例最多做一次「存档 = app.db」核对（见 :meth:`refresh_archive_from_app_db`）。"""

        if self._archive_checked:
            return
        self._archive_checked = True
        self.refresh_archive_from_app_db()

    def refresh_archive_from_app_db(self) -> bool:
        """迁移后（``storage_schema_version>=2``）用 app.db 权威行按需刷新 JSON 存档。

        存档只是 app.db 的**可再生**物化视图：被删、损坏或过期时，下一次解析前自动重建，
        因此读入口（本工厂）**不依赖它成为单点**。只在权威字段有差异时写盘，稳态零 I/O；
        迁移前（<2）不做任何事（那时 JSON 本来就是活数据源，D25）。
        """

        if self._app_db_reader is None:
            return False
        if committed_storage_version(self._data_dir) < WORKSPACE_LAYOUT_VERSION:
            return False
        try:
            rows = list(self._app_db_reader() or ())
        except Exception:  # noqa: BLE001 -- 水合尽力而为，读入口绝不能因此失败
            return False
        if not rows:
            return False
        incoming = [workspace_from_record(row) for row in rows]
        if self._archive_matches(incoming):
            return False
        try:
            self._workspaces.replace_all(incoming)
        except OSError:
            # 存档不可写（只读盘）不应让读入口失败：内存副本已经是权威值。
            return False
        return True

    def _archive_matches(self, incoming: Sequence[Workspace]) -> bool:
        """存档是否与 app.db 权威行**逐字段**一致（一致则不写盘）。"""

        current = {item.id: item for item in self._workspaces.list()}
        if set(current) != {item.id for item in incoming}:
            return False
        for item in incoming:
            existing = current[item.id]
            existing_fields = (
                existing.name,
                existing.storage_key,
                existing.status,
                existing.knowledge_revision,
                existing.pinned,
                existing.order_index,
                existing.created_at,
                existing.updated_at,
                existing.default_permission_mode,
                existing.default_reasoning_level,
            )
            incoming_fields = (
                item.name,
                item.storage_key,
                item.status,
                item.knowledge_revision,
                item.pinned,
                item.order_index,
                item.created_at,
                item.updated_at,
                item.default_permission_mode,
                item.default_reasoning_level,
            )
            if existing_fields != incoming_fields:
                return False
        return True

    def _require_workspace(self, workspace_id: str) -> Workspace:
        self._ensure_archive_fresh()
        workspace = self._workspaces.get(workspace_id)
        if workspace is None:
            raise WorkspaceNotFoundError(f"工作区不存在：{workspace_id}")
        return workspace

    def workspace_for_storage_key(self, storage_key: str) -> Workspace:
        """目录 → 工作区反查；未知 ``storage_key`` 抛 :class:`WorkspaceNotFoundError`。"""

        self._ensure_archive_fresh()
        workspace = self._workspaces.get_by_storage_key(storage_key)
        if workspace is None:
            raise WorkspaceNotFoundError(f"storage_key 无归属：{storage_key}")
        return workspace

    # ---- 解析 ----

    def layout_mode(self, workspace_id: str) -> LayoutMode:
        """本区应使用的布局：**永远**是共享布局（D-KB）。

        仍然校验归属（未知 id 抛 ``WorkspaceNotFoundError``），但结果与
        ``storage.json`` / 工作区身份**无关** —— 知识库全局共享，不再按区划分。
        """

        self._require_workspace(workspace_id)
        return LayoutMode.SHARED

    def paths_for(self, workspace_id: str) -> WorkspacePaths:
        """本区路径：共享知识（所有区相同）+ 私有件（按 ``storage_key``）。"""

        workspace = self._require_workspace(workspace_id)
        return _shared_paths(self._data_dir, workspace.storage_key)

    def view_for(self, workspace_id: str) -> WorkspaceContextView:
        """构造 PRD 5.2 的数据视图（身份 / 目录 / 知识版本 / 默认设置）。

        ``root_dir`` 是**本区私有根**；共享知识路径由 ``data_dir`` 派生
        （与 :class:`WorkspacePaths` 的定义逐字段一致）。
        """

        workspace = self._require_workspace(workspace_id)
        return WorkspaceContextView(
            workspace_id=workspace.id,
            storage_key=workspace.storage_key,
            name=workspace.name,
            status=workspace.status,
            knowledge_revision=workspace.knowledge_revision,
            default_permission_mode=workspace.default_permission_mode,
            default_reasoning_level=workspace.default_reasoning_level,
            data_dir=self._data_dir,
            root_dir=self._data_dir / "workspaces" / workspace.storage_key,
        )

    def resolve(self, workspace_id: str | None = None) -> WorkspaceContext:
        """解析并返回不可变上下文；``workspace_id=None`` 落到默认工作区。"""

        target = workspace_id or self.default_workspace_id()
        return WorkspaceContext(
            view=self.view_for(target),
            paths=self.paths_for(target),
            app_store=self._app_store,
        )

    def resolve_storage_key(self, storage_key: str) -> WorkspaceContext:
        """按目录键解析上下文（原图 / 资料路径校验用）。"""

        workspace = self.workspace_for_storage_key(storage_key)
        return self.resolve(workspace.id)

    def ensure_dirs(self, workspace_id: str) -> WorkspacePaths:
        """创建本区目录树（PRD 5.1）；迁移与首次写入时调用。"""

        return ensure_layout_dirs(self.paths_for(workspace_id))

    def jieba(self, workspace_id: str) -> Any:
        """返回该工作区的 jieba 分词器（PRD 11.5）。"""

        return jieba_tokenizer(self.paths_for(workspace_id))
