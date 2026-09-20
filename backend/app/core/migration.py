"""V1.2.0 数据迁移：V1.1.0 根布局 → 每工作区布局（PRD 15.1–15.3，Owner: T01）。

PRD 15.2 的十步在这里一一对应，且每一步都可单独指认：

| 步骤 | 实现 |
|---|---|
| 1 排他锁 + 迁移记录 + 一致备份 | :meth:`V120Migration._stage_backup` |
| 2 ``workspaces.json``/``conversations.json`` → ``app.db`` | :meth:`_stage_import_json` |
| 3 ``storage_key`` + 目录树 | :meth:`_stage_workspace_dirs` |
| 4 SQLite 官方备份接口复制状态库 | :meth:`_stage_copy_state_databases` |
| 5 复制上传文件（保留字节与 mtime） | :meth:`_stage_copy_uploads` |
| 6 Chroma 公共 API 显式 embeddings 分批搬运 | :meth:`_stage_move_vectors` |
| 7 更新来源路径 / 同步根 / 图片引用 + 归属 | :meth:`_stage_update_paths` |
| 8 旧记忆按类别转换表导入 | :meth:`_stage_import_memory` |
| 9 目录术语基础索引 + 历史全文索引 | :meth:`_stage_build_indexes` |
| 10 校验（15.3）后提交 ``storage_schema_version`` | :meth:`_stage_verify` + :meth:`_stage_commit` |

三条硬约束（违反即迁移失败，不降级）：

1. **零出网**：全程不调用嵌入 / 识图 / 对话模型（PRD 15.1）。搬运向量只走 Chroma
   公共 API 的**显式 embeddings** 接口；**绝不**调用 ``ChromaVectorStore.upsert()``
   —— 它会读取惰性初始化的 ``embedder.dim``，在真实库上可能触发远程嵌入调用
   （PRD 15.2 末段）。
2. **枚举口径**：collection 与记录数一律经公共 API / ``chroma.sqlite3`` 枚举，
   **不按目录个数或文件名推断**（架构 §6.2c D16 第 4 条）。真实目录的分段目录数与
   segment 数并不相等，目录计数按构造就是错的。
3. **备份不足即中止**：余量不足以容纳「备份 + 目标 chroma」时抛
   :class:`MigrationAbortedError`，**不降级**为「不备份」（D16 第 6 条）。

备份范围遵循 D16：排除 ``models/``（已移除组件的 ONNX 残留）并在迁移记录写明排除项
与理由；表外路径（``models/`` / ``vision_batch_report*.json`` /
``state.pre-image-v7.db`` / ``.chroma-write.lock``）**原地保留，不删除**。

幂等与断点续跑：消息 ID 由 ``(conversation_id, 旧下标)`` 确定性派生（``uuid5``），
向量写入用 ``upsert``，已完成阶段记在迁移记录的 ``completed_stages`` 里；重跑跳过
已完成阶段且不产生重复行。**续跑入口**：把上一次的 ``job_id`` 传回 :meth:`run`。

> 本模块只做迁移，不改 ``app/core/workspace_store.py`` 的读写语义，也不让
> ``app/agent/app_store.py`` 反向依赖 ``app/core``（架构 §6.4）。
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import chromadb
from chromadb.api.shared_system_client import SharedSystemClient
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import InternalError as ChromaInternalError

from app.agent.app_store import AppStore
from app.agent.contracts import (
    ConversationStatus,
    MemoryEntryStatus,
    MemoryFile,
    MemoryFileEntry,
    MemoryKind,
    WorkspaceStatus,
)
from app.core.workspace_context import (
    WorkspaceContextFactory,
    commit_storage_version,
    committed_storage_version,
    jieba_tokenizer,
)
from app.core.workspace_store import (
    DEFAULT_STORAGE_KEY,
    DEFAULT_WORKSPACE_ID,
    WORKSPACE_LAYOUT_VERSION,
    Workspace,
    WorkspaceStore,
    derive_storage_key,
)
from app.vectorstore.write_lock import exclusive_write_lock

__all__ = [
    "BACKUP_EXCLUDED",
    "BACKUP_INCLUDED",
    "BACKUP_PATTERNS",
    "CHROMA_BATCH_SIZE",
    "MIGRATION_LOCK_FILENAME",
    "MIGRATION_REVIEW_FILENAME",
    "MIGRATION_VERSION",
    "MemoryConversion",
    "MigrationAbortedError",
    "MigrationError",
    "MigrationReport",
    "MigrationStage",
    "MigrationVerificationError",
    "V120Migration",
    "convert_legacy_memory",
    "stable_message_id",
]

MIGRATION_VERSION = "v1.2.0"
#: 迁移期间的排他锁（PRD 15.2 第 1 步：禁止新旧数据同时写入）。
MIGRATION_LOCK_FILENAME = "migration.lock"
#: 迁移审阅记录：来源无法确认 / 类别暂不能转换 / 旧 ``pending`` 的记忆落这里，
#: 用户确认归属后才启用（PRD 15.2 第 8 步）。
MIGRATION_REVIEW_FILENAME = "migration_review.json"
BACKUP_SUFFIX = ".migration-backup"
#: D16 第 1 条：必须纳入一致快照的条目。
BACKUP_INCLUDED: tuple[str, ...] = (
    "chroma.sqlite3",
    "state.db",
    "state.pre-image-v7.db",
    "image_state.db",
    "conversations.json",
    "workspaces.json",
    "config.json",
    "documents",
    "avatars",
    "staging",
    ".chroma-write.lock",
)
#: D16 第 1 条：通配纳入。
BACKUP_PATTERNS: tuple[str, ...] = ("vision_batch_report*.json",)
#: D16 第 1 条：排除项（理由写进迁移记录）。
BACKUP_EXCLUDED: tuple[str, ...] = ("models",)
#: D16 第 3 条：表外路径原地保留、不删除。
TABLE_OUTSIDE_PATHS: tuple[str, ...] = (
    "models",
    "state.pre-image-v7.db",
    ".chroma-write.lock",
)
#: Chroma 分段目录名（HNSW 段以 collection 的 UUID 命名，位于 persist 根下）。
SEGMENT_DIR_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
#: 向量搬运批次（PRD 15.2 第 6 步「分批」）：真实库约 16.8k 条 × 4096 维。
CHROMA_BATCH_SIZE = 256
#: 全量逐值比对向量的上限；超过则抽样比对并在校验结果里记录。
FULL_VECTOR_VERIFY_LIMIT = 20_000
#: 每工作区目录树（PRD 5.1）。
WORKSPACE_SUBDIRS: tuple[str, ...] = ("documents", "chroma", "artifacts", "cache/jieba")


class MigrationError(Exception):
    """迁移失败的基类。"""


class MigrationAbortedError(MigrationError):
    """前置条件不满足（余量不足、源缺失）→ 中止且**不降级**。"""


class MigrationVerificationError(MigrationError):
    """PRD 15.3 校验未通过 → 不提交 ``storage_schema_version``。"""


class MigrationStage(StrEnum):
    """PRD 15.2 的十步（加收尾提交），用于迁移记录与断点续跑。"""

    BACKUP = "backup"
    IMPORT_JSON = "import_json"
    WORKSPACE_DIRS = "workspace_dirs"
    COPY_STATE_DATABASES = "copy_state_databases"
    COPY_UPLOADS = "copy_uploads"
    MOVE_VECTORS = "move_vectors"
    UPDATE_PATHS = "update_paths"
    IMPORT_MEMORY = "import_memory"
    BUILD_INDEXES = "build_indexes"
    VERIFY = "verify"
    COMMIT = "commit"


STAGE_ORDER: tuple[MigrationStage, ...] = tuple(MigrationStage)


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def stable_message_id(conversation_id: str, index: int) -> str:
    """由 (会话, 旧下标) 确定性派生消息 ID —— 重跑迁移不会产生第二条同义消息。"""

    return uuid5(NAMESPACE_URL, f"everything-rag:msg:{conversation_id}:{index}").hex


@dataclass(frozen=True)
class MemoryConversion:
    """一条旧记忆的转换结论（PRD 15.2 类别转换表）。"""

    kind: MemoryKind | None
    status: MemoryEntryStatus | None
    review_reason: str = ""

    @property
    def needs_review(self) -> bool:
        return bool(self.review_reason)


def convert_legacy_memory(
    *,
    category: str,
    status: str,
    statement: str,
    quote: str = "",
    source_resolved: bool = True,
    user_confirmed: bool = False,
    workspace_linked: bool = True,
) -> MemoryConversion:
    """按 PRD 15.2 的类别 / 状态规则转换一条旧记忆。

    - ``preference`` / ``constraint``：明确的回答要求 → ``response_preference``；
      具体方案条件 → ``progress``。
    - ``decision`` / ``conclusion``：有用户确认依据 → ``confirmed_conclusion``；
      只有讨论依据 → ``progress``（保留讨论状态）。
    - ``identity``：与工作区主题或项目有关 → ``topic``；其余身份内容 → 审阅记录。
    - 只有旧 ``active`` **且**来源可核验（含引用原话）才成为新 ``active``；旧
      ``forgotten`` / ``superseded`` / ``orphaned`` 保持状态；旧 ``pending`` 与
      引用无法核验的一律进审阅记录，不能自动取得有效状态。
    """

    if status == "pending":
        return MemoryConversion(None, None, "旧 pending：须用户确认后才启用")
    if status in {"forgotten", "superseded", "orphaned"}:
        return MemoryConversion(None, MemoryEntryStatus(status), "")
    if status != "active":
        return MemoryConversion(None, None, f"未知旧状态：{status}")
    if not source_resolved:
        return MemoryConversion(None, None, "来源消息无法通过消息映射核验")
    if not workspace_linked:
        return MemoryConversion(None, None, "来源会话无法定位工作区")
    if not quote.strip():
        return MemoryConversion(None, None, "缺少可核验的引用原话")

    if category in {"preference", "constraint"}:
        kind = (
            MemoryKind.RESPONSE_PREFERENCE
            if _is_answer_requirement(statement)
            else MemoryKind.PROGRESS
        )
    elif category in {"decision", "conclusion"}:
        kind = MemoryKind.CONFIRMED_CONCLUSION if user_confirmed else MemoryKind.PROGRESS
    elif category == "identity":
        if _is_workspace_linked(statement):
            kind = MemoryKind.TOPIC
        else:
            return MemoryConversion(None, None, "身份内容与工作区主题无关")
    else:
        return MemoryConversion(None, None, f"类别暂不能转换：{category}")
    return MemoryConversion(kind, MemoryEntryStatus.ACTIVE, "")


#: 明确指向「回答方式」的表述 → response_preference（PRD 15.2 第 8 步）。
_ANSWER_REQUIREMENT_WORDS = ("回答", "请用", "语气", "格式", "中文", "英文", "简洁", "详细")
#: 与工作区主题 / 项目有关的身份内容 → topic。
_WORKSPACE_TOPIC_WORDS = ("项目", "工作区", "产品", "本仓库", "知识库")


def _is_answer_requirement(statement: str) -> bool:
    return any(word in statement for word in _ANSWER_REQUIREMENT_WORDS)


def _is_workspace_linked(statement: str) -> bool:
    return any(word in statement for word in _WORKSPACE_TOPIC_WORDS)


@dataclass
class MigrationReport:
    """迁移结果（同时作为迁移记录的 ``validation`` 内容）。"""

    data_dir: Path
    source_dir: Path
    job_id: str = ""
    backup_dir: Path | None = None
    excluded: tuple[str, ...] = ()
    included: tuple[str, ...] = ()
    collections: dict[str, int] = field(default_factory=dict)
    conversations: int = 0
    messages: int = 0
    uploads: int = 0
    images: int = 0
    memory_imported: int = 0
    memory_reviewed: int = 0
    path_map: dict[str, str] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    completed_stages: list[str] = field(default_factory=list)
    already_migrated: bool = False

    @property
    def vector_count(self) -> int:
        return sum(self.collections.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "migration_version": MIGRATION_VERSION,
            "job_id": self.job_id,
            "source_dir": str(self.source_dir),
            "data_dir": str(self.data_dir),
            "backup_dir": None if self.backup_dir is None else str(self.backup_dir),
            "excluded": list(self.excluded),
            "excluded_reason": "已移除 fastembed/bge-m3 组件的 ONNX 残留，非 PRD 5.1 目标路径",
            "included": list(self.included),
            "collections": dict(self.collections),
            "vector_count": self.vector_count,
            "conversations": self.conversations,
            "messages": self.messages,
            "uploads": self.uploads,
            "images": self.images,
            "memory_imported": self.memory_imported,
            "memory_reviewed": self.memory_reviewed,
            "path_map_size": len(self.path_map),
            "completed_stages": list(self.completed_stages),
            "already_migrated": self.already_migrated,
            "validation": self.validation,
        }


class V120Migration:
    """PRD 15.2 十步迁移的执行器（async：``app.db`` 访问层是 aiosqlite）。"""

    def __init__(
        self,
        data_dir: Path,
        *,
        app_store: AppStore,
        workspace_store: WorkspaceStore | None = None,
        batch_size: int = CHROMA_BATCH_SIZE,
        lock_timeout_seconds: float = 30.0,
        verify_vectors_fully: bool | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._app_store = app_store
        self._workspaces = workspace_store or WorkspaceStore(self._data_dir)
        self._batch_size = max(1, batch_size)
        self._lock_timeout = lock_timeout_seconds
        self._verify_vectors_fully = verify_vectors_fully
        #: 按 persist 根复用的 Chroma 客户端（见 :meth:`_chroma_client`）。
        self._clients: dict[str, Any] = {}
        #: 会话 → 工作区（第 2 步建立，供第 8 步记忆归属与第 9 步历史索引使用）。
        self._conversation_workspace: dict[str, str] = {}
        #: 会话 → {旧消息下标: 新消息 ID}（PRD 15.2 第 2 步）。
        self._message_map: dict[str, dict[str, str]] = {}
        #: 工作区 → storage_key（第 3 步建立）。
        self._storage_keys: dict[str, str] = {}

    # ==================================================================
    # 入口
    # ==================================================================

    async def run(
        self,
        *,
        source_dir: Path | None = None,
        job_id: str | None = None,
    ) -> MigrationReport:
        """执行迁移；已提交 ``storage_schema_version=2`` 时幂等返回。

        ``job_id`` 传上一次的迁移任务 ID 即从断点续跑（跳过 ``completed_stages``）。
        """

        source = Path(source_dir or self._data_dir)
        report = MigrationReport(data_dir=self._data_dir, source_dir=source)
        if committed_storage_version(self._data_dir) >= WORKSPACE_LAYOUT_VERSION:
            report.already_migrated = True
            report.validation = {"skipped": "已提交 storage_schema_version=2"}
            return report

        job_id = await self._begin_job(job_id, source)
        report.job_id = job_id
        completed = await self._completed_stages(job_id)
        if completed:
            self._hydrate_from_record(report, await self._app_store.get_migration_job(job_id))
        report.completed_stages = sorted(
            completed, key=lambda name: STAGE_ORDER.index(MigrationStage(name))
        )

        try:
            # PRD 15.2 第 1 步：排他锁 —— 迁移期间禁止新旧数据同时写入。
            with exclusive_write_lock(
                self._data_dir / MIGRATION_LOCK_FILENAME, timeout_seconds=self._lock_timeout
            ):
                await self._run_stages(report, source, job_id, completed)
        except MigrationError as exc:
            await self._fail(job_id, report, status="failed", error=str(exc))
            raise
        except Exception as exc:
            await self._fail(
                job_id, report, status="interrupted", error=f"{type(exc).__name__}: {exc}"
            )
            raise
        return report

    async def _fail(
        self, job_id: str, report: MigrationReport, *, status: str, error: str
    ) -> None:
        """失败/中断时也保留**顶层** ``completed_stages``，断点续跑才能接得上。

        ``validation`` 的布局与 :meth:`_checkpoint` 完全一致（报告字段在顶层），
        否则重跑时读不到已完成阶段，会把整段迁移重做一遍。
        """

        validation = report.to_dict()
        validation["error"] = error
        await self._app_store.update_migration_job(
            job_id, stage="failed", status=status, validation=validation
        )

    async def _run_stages(
        self,
        report: MigrationReport,
        source: Path,
        job_id: str,
        completed: set[str],
    ) -> None:
        """按 PRD 15.2 顺序执行；已完成的阶段直接跳过（断点续跑）。"""

        plan: tuple[tuple[MigrationStage, Any], ...] = (
            (MigrationStage.BACKUP, lambda: self._stage_backup(report, source)),
            (MigrationStage.IMPORT_JSON, lambda: self._stage_import_json(report, source)),
            (MigrationStage.WORKSPACE_DIRS, lambda: self._stage_workspace_dirs(report)),
            (
                MigrationStage.COPY_STATE_DATABASES,
                lambda: self._stage_copy_state_databases(report, source),
            ),
            (MigrationStage.COPY_UPLOADS, lambda: self._stage_copy_uploads(report, source)),
            (MigrationStage.MOVE_VECTORS, lambda: self._stage_move_vectors(report, source)),
            (MigrationStage.UPDATE_PATHS, lambda: self._stage_update_paths(report, source)),
            (MigrationStage.IMPORT_MEMORY, lambda: self._stage_import_memory(report, source)),
            (MigrationStage.BUILD_INDEXES, lambda: self._stage_build_indexes(report, source)),
            (MigrationStage.VERIFY, lambda: self._stage_verify(report, source)),
        )
        for stage, action in plan:
            if stage.value in completed:
                continue
            await self._app_store.update_migration_job(
                job_id, stage=stage.value, status="running"
            )
            result = action()
            if hasattr(result, "__await__"):
                await result
            report.completed_stages.append(stage.value)
            await self._checkpoint(job_id, report, stage)

        if MigrationStage.COMMIT.value not in completed:
            self._stage_commit(report)
            report.completed_stages.append(MigrationStage.COMMIT.value)
            await self._checkpoint(job_id, report, MigrationStage.COMMIT)

        await self._app_store.update_migration_job(
            job_id,
            stage=MigrationStage.COMMIT.value,
            status="completed",
            validation=report.to_dict(),
        )

    async def _begin_job(self, job_id: str | None, source: Path) -> str:
        if job_id is not None:
            existing = await self._app_store.get_migration_job(job_id)
            if existing is not None:
                return job_id
        record = await self._app_store.record_migration_job(
            migration_version=MIGRATION_VERSION,
            stage=MigrationStage.BACKUP.value,
            source_fingerprint=self._source_fingerprint(source),
            target=str(self._data_dir / "workspaces"),
            status="running",
            job_id=job_id,
        )
        return record.id

    async def _completed_stages(self, job_id: str) -> set[str]:
        record = await self._app_store.get_migration_job(job_id)
        if record is None:
            return set()
        stages = record.validation.get("completed_stages")
        return {str(name) for name in stages} if isinstance(stages, list) else set()

    def _hydrate_from_record(self, report: MigrationReport, record: Any) -> None:
        """续跑时把上一轮已完成的计数/映射接回来，避免报告与校验数字变空。"""

        if record is None or not isinstance(record.validation, dict):
            return
        # ``validation`` 的字段布局与 report.to_dict() 一致（见 _checkpoint / _fail）。
        previous: Any = record.validation
        if not isinstance(previous.get("conversations"), int) and isinstance(
            previous.get("report"), dict
        ):
            previous = previous["report"]  # 兼容早期把报告嵌在 "report" 下的记录
        if not isinstance(previous, dict):
            return
        for key in ("conversations", "messages", "uploads", "images", "memory_imported",
                    "memory_reviewed"):
            value = previous.get(key)
            if isinstance(value, int):
                setattr(report, key, value)
        for key in ("collections", "path_map"):
            value = previous.get(key)
            if isinstance(value, dict):
                getattr(report, key).update(value)
        excluded = previous.get("excluded")
        if isinstance(excluded, list):
            report.excluded = tuple(str(item) for item in excluded)
        included = previous.get("included")
        if isinstance(included, list):
            report.included = tuple(str(item) for item in included)
        backup_dir = previous.get("backup_dir")
        if isinstance(backup_dir, str) and report.backup_dir is None:
            report.backup_dir = Path(backup_dir)

    async def _checkpoint(
        self, job_id: str, report: MigrationReport, stage: MigrationStage
    ) -> None:
        await self._app_store.update_migration_job(
            job_id, stage=stage.value, status="running", validation=report.to_dict()
        )

    # ==================================================================
    # 第 1 步：排他锁 + 迁移记录 + 一致备份
    # ==================================================================

    def _backup_plan(self, source: Path) -> tuple[list[str], int]:
        """返回（纳入快照的相对路径, 所需字节数）；``models/`` 按 D16 排除。

        D16 第 1 条要求 ``chroma.sqlite3`` **与全部分段目录**一起进快照：分段目录是
        HNSW 图本体（``data_level0.bin`` / ``link_lists.bin`` …），只备份 sqlite 会得到
        一个「有逻辑记录但没有近邻图」的残缺库。分段目录名是 collection 的 UUID，
        因此按名字形态整体纳入，而不是只复制写死的清单。
        """

        names: list[str] = []
        total = 0
        for name in BACKUP_INCLUDED:
            path = source / name
            if not path.exists():
                continue
            names.append(name)
            total += self._tree_size(path)
        for pattern in BACKUP_PATTERNS:
            for path in sorted(source.glob(pattern)):
                if path.name not in names:
                    names.append(path.name)
                    total += self._tree_size(path)
        for name in self._segment_directory_names(source):
            if name not in names:
                names.append(name)
                total += self._tree_size(source / name)
        return names, total

    @staticmethod
    def _segment_directory_names(source: Path) -> list[str]:
        """列出 persist 根下的 Chroma 分段目录名（按名字形态识别，不数记录数）。"""

        try:
            entries = sorted(source.iterdir())
        except OSError:
            return []
        return [
            entry.name
            for entry in entries
            if entry.is_dir()
            and entry.name not in BACKUP_EXCLUDED
            and SEGMENT_DIR_PATTERN.match(entry.name)
        ]

    def _stage_backup(self, report: MigrationReport, source: Path) -> None:
        names, snapshot_bytes = self._backup_plan(source)
        # 目标 chroma 也要落盘，容量校验必须把它算进去（D16 第 6 条）。
        required = snapshot_bytes + self._tree_size(source / "chroma.sqlite3")
        backup_dir = source.with_name(f"{source.name}{BACKUP_SUFFIX}-{_timestamp()}")
        free = shutil.disk_usage(backup_dir.parent).free
        if free < required:
            # 中止，而不是降级为「不备份」。
            raise MigrationAbortedError(
                f"磁盘余量不足：需要 {required} 字节（备份 + 目标 chroma），可用 {free} 字节"
            )

        backup_dir.mkdir(parents=True, exist_ok=False)
        for name in names:
            src = source / name
            dst = backup_dir / name
            if src.is_dir():
                shutil.copytree(src, dst, symlinks=True)
            elif self._is_sqlite(src):
                self._sqlite_backup(src, dst)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        report.backup_dir = backup_dir
        report.excluded = BACKUP_EXCLUDED
        report.included = tuple(names)
        report.validation["backup_required_bytes"] = required
        report.validation["backup_free_bytes"] = free
        report.validation["backup_excluded"] = list(BACKUP_EXCLUDED)
        report.validation["table_outside_kept"] = list(TABLE_OUTSIDE_PATHS)

    @staticmethod
    def _is_sqlite(path: Path) -> bool:
        return path.suffix in {".db", ".sqlite3"}

    @staticmethod
    def _sqlite_backup(source: Path, target: Path) -> None:
        """SQLite 官方备份接口（PRD 15.2 第 4 步），不复制 WAL 中间态。

        非 SQLite 内容的 ``*.db``（历史遗留的损坏/占位文件）退化为字节复制：
        备份阶段的职责是**先保住原始字节**，不能因为这个文件不是数据库就让整次
        迁移中止（真正的损坏判定由校验阶段负责）。
        """

        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with (
                sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src_conn,
                sqlite3.connect(str(target)) as dst_conn,
            ):
                src_conn.backup(dst_conn)
        except sqlite3.DatabaseError:
            shutil.copy2(source, target)

    def _tree_size(self, path: Path) -> int:
        if not path.exists():
            return 0
        if path.is_file():
            try:
                return path.stat().st_size
            except OSError:
                return 0
        total = 0
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except OSError:
                    continue
        return total

    def _source_fingerprint(self, source: Path) -> str:
        digest = hashlib.sha256()
        for name in ("workspaces.json", "conversations.json", "state.db", "image_state.db"):
            path = source / name
            if not path.is_file():
                continue
            digest.update(name.encode("utf-8"))
            digest.update(str(path.stat().st_size).encode("utf-8"))
        return digest.hexdigest()[:32]

    # ==================================================================
    # 第 2 步：JSON → app.db
    # ==================================================================

    async def _stage_import_json(self, report: MigrationReport, source: Path) -> None:
        workspaces = self._read_json_list(source / "workspaces.json")
        conversations = self._read_json_list(source / "conversations.json")

        for raw in workspaces:
            workspace_id = str(raw.get("id") or "").strip()
            if not workspace_id:
                continue
            if await self._app_store.get_workspace(workspace_id) is None:
                await self._app_store.create_workspace(
                    workspace_id,
                    str(raw.get("storage_key") or derive_storage_key(workspace_id)),
                    name=str(raw.get("name") or ""),
                    status=(
                        WorkspaceStatus.ARCHIVED
                        if str(raw.get("status")) == "archived"
                        else WorkspaceStatus.ACTIVE
                    ),
                )

        if await self._app_store.get_workspace(DEFAULT_WORKSPACE_ID) is None:
            await self._app_store.create_workspace(
                DEFAULT_WORKSPACE_ID, DEFAULT_STORAGE_KEY, name="默认"
            )

        for raw in conversations:
            conversation_id = str(raw.get("id") or "").strip()
            if not conversation_id:
                continue
            workspace_id = str(raw.get("workspace_id") or DEFAULT_WORKSPACE_ID)
            if await self._app_store.get_workspace(workspace_id) is None:
                # PRD 15.2 第 2 步：会话引用的工作区记录缺失 → 建 archived 记录，
                # 保留原 id 与归属，名称标明已归档。
                await self._app_store.create_workspace(
                    workspace_id,
                    derive_storage_key(workspace_id),
                    name=f"{workspace_id}（已归档）",
                    status=WorkspaceStatus.ARCHIVED,
                )
            self._conversation_workspace[conversation_id] = workspace_id

            if await self._app_store.get_conversation(conversation_id) is None:
                await self._app_store.create_conversation(
                    conversation_id,
                    workspace_id,
                    title=str(raw.get("title") or ""),
                    parent_conversation_id=raw.get("parent_id") or None,
                )
            if raw.get("archived"):
                await self._app_store.update_conversation(
                    conversation_id, status=ConversationStatus.ARCHIVED
                )

            mapping = self._message_map.setdefault(conversation_id, {})
            for index, message in enumerate(raw.get("messages") or []):
                message_id = stable_message_id(conversation_id, index)
                mapping[str(index)] = message_id
                if await self._app_store.get_message(message_id) is None:
                    await self._app_store.append_message(
                        conversation_id=conversation_id,
                        role=str(message.get("role") or "user"),
                        content=str(message.get("content") or ""),
                        source_snapshot=self._message_snapshot(message),
                        message_id=message_id,
                        created_at=str(message.get("created_at") or "") or None,
                    )
            report.conversations += 1
            report.messages += len(raw.get("messages") or [])

        report.validation["message_map_conversations"] = len(self._message_map)
        report.validation["conversations_without_workspace_record"] = sum(
            1
            for raw in conversations
            if str(raw.get("workspace_id") or DEFAULT_WORKSPACE_ID)
            not in {str(item.get("id") or "") for item in workspaces}
        )

    @staticmethod
    def _message_snapshot(message: Mapping[str, Any]) -> dict[str, Any]:
        """消息附加字段（依据 / 来源 / 思维链）整存进 ``source_snapshot``。"""

        snapshot: dict[str, Any] = {}
        for key in ("sources", "answer_basis", "policy_version", "reasoning"):
            value = message.get(key)
            if value:
                snapshot[key] = value
        return snapshot

    @staticmethod
    def _read_json_list(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MigrationError(f"无法读取 {path.name}：{exc}") from exc
        if not isinstance(raw, list):
            raise MigrationError(f"{path.name} 不是数组")
        return [item for item in raw if isinstance(item, dict)]

    # ==================================================================
    # 第 3 步：storage_key + 目录树
    # ==================================================================

    async def _stage_workspace_dirs(self, report: MigrationReport) -> None:
        mirrored: list[Workspace] = []
        for record in await self._app_store.list_workspaces():
            storage_key = record.storage_key or derive_storage_key(record.id)
            self._storage_keys[record.id] = storage_key
            root = self._data_dir / "workspaces" / storage_key
            for relative in WORKSPACE_SUBDIRS:
                (root / relative).mkdir(parents=True, exist_ok=True)
            mirrored.append(
                Workspace(
                    id=record.id,
                    name=record.name,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    is_default=record.id == DEFAULT_WORKSPACE_ID,
                    storage_key=storage_key,
                    status=record.status,
                    default_permission_mode=record.default_permission_mode,
                    default_reasoning_level=record.default_reasoning_level,
                    knowledge_revision=record.knowledge_revision,
                )
            )
        # 架构 §6.4：app.db 是唯一权威，workspaces.json 迁移后降级为只读存档；
        # 方向永远是 app.db → 存档（单向适配器）。
        self._workspaces.replace_all(mirrored)
        report.validation["workspace_count"] = len(mirrored)

    # ==================================================================
    # 第 4 步：SQLite 官方备份接口复制状态库
    # ==================================================================

    def _stage_copy_state_databases(self, report: MigrationReport, source: Path) -> None:
        target_root = self._target_workspace_root()
        copied: list[str] = []
        for name in ("state.db", "image_state.db"):
            src = source / name
            if not src.is_file():
                continue
            self._sqlite_backup(src, target_root / name)
            copied.append(name)
        report.validation["copied_state_databases"] = copied

    def _target_workspace_root(self) -> Path:
        """全局知识归入 ``default`` 工作区（PRD 15.1）。"""

        return self._data_dir / "workspaces" / DEFAULT_STORAGE_KEY

    # ==================================================================
    # 第 5 步：复制上传文件（保留字节与 mtime）
    # ==================================================================

    def _stage_copy_uploads(self, report: MigrationReport, source: Path) -> None:
        src_root = source / "documents"
        if not src_root.is_dir():
            return
        dst_root = self._target_workspace_root() / "documents"
        for item in sorted(src_root.rglob("*")):
            if not item.is_file():
                continue
            relative = item.relative_to(src_root)
            target = dst_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(item, target)
            report.path_map[str(item)] = str(target)
            report.uploads += 1

    # ==================================================================
    # 第 6 步：Chroma 公共 API 分批搬运
    # ==================================================================

    def _stage_move_vectors(self, report: MigrationReport, source: Path) -> None:
        # 旧布局：Chroma persist 根就是 data_dir 本身（chroma.sqlite3 与全部分段
        # 目录同层，见架构 §6.2c）。
        if not (source / "chroma.sqlite3").is_file():
            return
        target_chroma = self._target_workspace_root() / "chroma"
        target_chroma.mkdir(parents=True, exist_ok=True)

        src_client = self._chroma_client(source)
        dst_client = self._chroma_client(target_chroma)
        for name in self._collection_names(src_client):
            src_collection = src_client.get_collection(name)
            count = int(src_collection.count())
            dst_collection = dst_client.get_or_create_collection(
                name=name,
                metadata=dict(src_collection.metadata or {"hnsw:space": "cosine"}),
            )
            for offset in range(0, count, self._batch_size):
                raw = src_collection.get(
                    include=["documents", "embeddings", "metadatas"],
                    limit=self._batch_size,
                    offset=offset,
                )
                ids = [str(item) for item in (raw.get("ids") or [])]
                if not ids:
                    continue
                embeddings = self._as_float_lists(raw.get("embeddings"))
                documents = [str(text) for text in (raw.get("documents") or [])]
                # 显式 embeddings 写入：不经过 ChromaVectorStore.upsert()，不触碰
                # 惰性 embedder.dim，因此不可能触发远程嵌入调用（PRD 15.2 末段）。
                dst_collection.upsert(
                    ids=ids,
                    documents=documents or None,
                    embeddings=embeddings or None,
                    metadatas=(
                        [dict(meta or {}) for meta in (raw.get("metadatas") or [])] or None
                    ),
                )
            # 计数一律取公共 API 的枚举结果（D16 第 4 条），不看目录个数。
            report.collections[name] = int(dst_collection.count())
            # 物化 HNSW 分段（见 _materialize_segments）：避免后续阶段在另一个
            # PersistentClient 上读到「只有 sqlite 行、没有分段目录」的中间态。
            self._materialize_segments(dst_collection)

    def _chroma_client(self, path: Path) -> Any:
        """按 persist 根**复用** Chroma 客户端（同一次迁移内不重复建 System）。

        为什么不每次新建：写入后立即用**另一个** ``PersistentClient`` 打开同一目录，
        会读到「sqlite 里已有 collection 行、但 ``<uuid>/`` HNSW 分段目录尚未物化」的
        中间态，Chromadb 抛 ``Error creating hnsw segment reader: Nothing found on disk``。
        复用同一个 System 即可消除该竞争（并少开一套 sqlite 连接）。
        """

        key = str(Path(path).resolve())
        client = self._clients.get(key)
        if client is None:
            client = chromadb.PersistentClient(
                path=key, settings=ChromaSettings(anonymized_telemetry=False)
            )
            self._clients[key] = client
        return client

    @staticmethod
    def _materialize_segments(collection: Any, *, sample: int = 1) -> None:
        """强制物化 HNSW 分段并做一次可达性自检。

        ``get`` 只走 sqlite；``query`` 才会实例化 HNSW 分段读取器并把分段目录落盘。
        生产侧 ``ChromaVectorStore.upsert`` 也是靠自查询做可达性校验，这里保持一致。
        """

        raw = collection.get(include=["embeddings"], limit=sample)
        embeddings = V120Migration._as_float_lists(raw.get("embeddings"))
        if not embeddings:
            return
        collection.query(query_embeddings=[embeddings[0]], n_results=1)

    @classmethod
    def _collection_names(cls, client: Any) -> list[str]:
        names: list[str] = []
        for item in client.list_collections():
            name = item if isinstance(item, str) else str(getattr(item, "name", ""))
            if name:
                names.append(name)
        return sorted(names)

    @staticmethod
    def _as_float_lists(embeddings: Any) -> list[list[float]]:
        if embeddings is None:
            return []
        rows: list[list[float]] = []
        for row in embeddings:
            if hasattr(row, "tolist"):
                rows.append([float(value) for value in row.tolist()])
            else:
                rows.append([float(value) for value in row])
        return rows

    # ==================================================================
    # 第 7 步：来源路径 / 同步根 / 图片引用 + 归属
    # ==================================================================

    def _stage_update_paths(self, report: MigrationReport, source: Path) -> None:
        if not report.path_map:
            return
        target_root = self._target_workspace_root()
        updated = 0
        for name in ("state.db", "image_state.db"):
            db_path = target_root / name
            if not db_path.is_file():
                continue
            with sqlite3.connect(str(db_path)) as conn:
                updated += self._rewrite_column(conn, "documents", "path", report.path_map)
                updated += self._rewrite_column(
                    conn, "image_tasks", "source_file", report.path_map
                )
                conn.commit()
        report.validation["rewritten_paths"] = updated

    @staticmethod
    def _rewrite_column(
        conn: sqlite3.Connection,
        table: str,
        column: str,
        mapping: Mapping[str, str],
    ) -> int:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if exists is None:
            return 0
        rows = conn.execute(f"SELECT rowid, {column} FROM {table}").fetchall()
        changed = 0
        for rowid, value in rows:
            replacement = mapping.get(str(value))
            if replacement is None:
                continue
            conn.execute(
                f"UPDATE {table} SET {column} = ? WHERE rowid = ?",
                (replacement, rowid),
            )
            changed += 1
        return changed

    # ==================================================================
    # 第 8 步：旧记忆导入
    # ==================================================================

    async def _stage_import_memory(self, report: MigrationReport, source: Path) -> None:
        rows = self._read_legacy_memory(source / "memory.db")
        if not rows:
            return
        workspace_entries: dict[str, list[MemoryFileEntry]] = {}
        review: list[dict[str, Any]] = []
        for row in rows:
            conversation_id = str(row.get("source_conversation_id") or "")
            workspace_id = self._conversation_workspace.get(conversation_id)
            old_index = row.get("source_message_index")
            message_ids: list[str] = []
            if isinstance(old_index, int) and old_index >= 0:
                resolved = self._message_map.get(conversation_id, {}).get(str(old_index))
                if resolved:
                    message_ids.append(resolved)
            conversion = convert_legacy_memory(
                category=str(row.get("category") or ""),
                status=str(row.get("status") or "active"),
                statement=str(row.get("statement") or ""),
                quote=str(row.get("quote") or ""),
                source_resolved=bool(message_ids),
                user_confirmed=bool(row.get("user_pinned")),
                workspace_linked=workspace_id is not None,
            )
            if conversion.needs_review:
                review.append(self._review_record(row, conversion.review_reason))
                report.memory_reviewed += 1
                continue
            workspace_entries.setdefault(str(workspace_id), []).append(
                MemoryFileEntry(
                    id=str(row.get("id")),
                    kind=conversion.kind or MemoryKind.PROGRESS,
                    text=str(row.get("statement") or ""),
                    source_conversation_id=conversation_id,
                    source_message_ids=message_ids,
                    created_at=str(row.get("created_at") or ""),
                    updated_at=str(row.get("updated_at") or ""),
                    user_pinned=bool(row.get("user_pinned")),
                    status=conversion.status or MemoryEntryStatus.ACTIVE,
                    superseded_by=row.get("superseded_by") or None,
                )
            )
            report.memory_imported += 1

        for workspace_id, entries in workspace_entries.items():
            path = (
                self._data_dir
                / "workspaces"
                / self._storage_key_of(workspace_id)
                / "memory.json"
            )
            existing = self._read_memory_file(path)
            by_id = {entry.id: entry for entry in existing.entries}
            for entry in entries:
                by_id[entry.id] = entry
            merged = MemoryFile(
                revision=existing.revision + 1,
                updated_at=_now_iso(),
                entries=list(by_id.values()),
                invalidations=existing.invalidations,
                processed_cursors=existing.processed_cursors,
                applied_operations=existing.applied_operations,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(merged.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if review:
            self._append_review_records(review)

    @staticmethod
    def _review_record(row: Mapping[str, Any], reason: str) -> dict[str, Any]:
        return {
            "memory_id": row.get("id"),
            "reason": reason,
            "category": row.get("category"),
            "status": row.get("status"),
            "statement": row.get("statement"),
            "quote": row.get("quote"),
            "source_conversation_id": row.get("source_conversation_id"),
            "source_message_index": row.get("source_message_index"),
        }

    def _append_review_records(self, records: list[dict[str, Any]]) -> None:
        """迁移审阅记录：追加且按 ``memory_id`` 去重（重跑迁移不重复登记）。"""

        path = self._data_dir / MIGRATION_REVIEW_FILENAME
        payload = self._read_raw_json(path)
        existing = payload.get("entries")
        entries: list[dict[str, Any]] = existing if isinstance(existing, list) else []
        known = {str(entry.get("memory_id")) for entry in entries if isinstance(entry, dict)}
        for record in records:
            if str(record.get("memory_id")) in known:
                continue
            entries.append(record)
        path.write_text(
            json.dumps(
                {
                    "migration_version": MIGRATION_VERSION,
                    "created_at": payload.get("created_at") or _now_iso(),
                    "entries": entries,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _read_legacy_memory(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    "SELECT id, category, statement, confidence, source_conversation_id,"
                    " source_message_index, quote, status, superseded_by, user_pinned,"
                    " created_at, updated_at FROM memories"
                ).fetchall()
            except sqlite3.DatabaseError:
                return []
        return [dict(row) for row in rows]

    @staticmethod
    def _read_memory_file(path: Path) -> MemoryFile:
        raw = V120Migration._read_raw_json(path)
        if not raw:
            return MemoryFile()
        try:
            return MemoryFile.model_validate(raw)
        except ValueError:
            return MemoryFile()

    @staticmethod
    def _read_raw_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def _storage_key_of(self, workspace_id: str) -> str:
        return self._storage_keys.get(workspace_id) or derive_storage_key(workspace_id)

    # ==================================================================
    # 第 9 步：目录术语基础索引 + 历史全文索引
    # ==================================================================

    async def _stage_build_indexes(self, report: MigrationReport, source: Path) -> None:
        target_root = self._target_workspace_root()
        catalog = self._build_catalog(target_root / "chroma")
        if catalog:
            (target_root / "catalog.json").write_text(
                json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            report.validation["catalog_documents"] = len(catalog.get("documents", []))
        indexed = 0
        tokenizers: dict[str, Any] = {}
        factory = WorkspaceContextFactory(self._data_dir, self._workspaces)
        for workspace_id, conversation_id, message_id, content in await self._all_messages():
            tokenizer = tokenizers.get(workspace_id)
            if tokenizer is None:
                # PRD 11.5：历史索引分词用**本区** tokenizer（tmp_dir / cache_file 在本区
                # cache/jieba 下），不写进程级默认缓存目录。
                tokenizer = jieba_tokenizer(factory.paths_for(workspace_id))
                tokenizers[workspace_id] = tokenizer
            await self._app_store.index_history_message(
                message_id=message_id,
                conversation_id=conversation_id,
                workspace_id=workspace_id,
                tokens=self._tokenize(content, tokenizer),
            )
            indexed += 1
        report.validation["history_indexed"] = indexed

    def _build_catalog(self, chroma_dir: Path) -> dict[str, Any]:
        """本地目录 / 术语**基础**索引：只做可从元数据确定性聚合的部分。

        需要模型才能生成的扩展目录信息在迁移完成后作为独立任务运行（PRD 15.2 第 9 步）。
        """

        if not (chroma_dir / "chroma.sqlite3").is_file():
            return {}
        client = self._chroma_client(chroma_dir)
        documents: dict[str, dict[str, Any]] = {}
        for name in self._collection_names(client):
            collection = client.get_collection(name)
            count = int(collection.count())
            for offset in range(0, count, self._batch_size):
                raw = collection.get(
                    include=["metadatas"], limit=self._batch_size, offset=offset
                )
                for meta in raw.get("metadatas") or []:
                    metadata = dict(meta or {})
                    source_file = str(metadata.get("source_file") or "")
                    if not source_file:
                        continue
                    entry = documents.setdefault(
                        source_file, {"source_file": source_file, "headings": [], "chunks": 0}
                    )
                    entry["chunks"] += 1
                    heading = metadata.get("heading_path")
                    if heading and heading not in entry["headings"]:
                        entry["headings"].append(heading)
        return {
            "migration_version": MIGRATION_VERSION,
            "generated_at": _now_iso(),
            "note": "基础目录索引（标题/层级/切片计数）；扩展目录由迁移后独立任务生成",
            "documents": sorted(documents.values(), key=lambda item: item["source_file"]),
        }

    async def _all_messages(self) -> list[tuple[str, str, str, str]]:
        """返回 ``(workspace_id, conversation_id, message_id, content)``。"""

        rows: list[tuple[str, str, str, str]] = []
        for workspace in await self._app_store.list_workspaces():
            for conversation in await self._app_store.list_conversations(workspace.id):
                for message in await self._app_store.list_messages(conversation.id):
                    rows.append((workspace.id, conversation.id, message.id, message.content))
        return rows

    @staticmethod
    def _tokenize(text: str, tokenizer: Any) -> str:
        """按 PRD 11.5 的**统一分词模式**切词（导入侧）。

        PRD 11.5：「通过实例加载 UTF-8 术语词典，导入与查询使用相同的
        ``cut_for_search()`` 配置」。两者确实会分叉，但**只在词典里的合成词上**，
        实测（jieba 0.42.1 默认词典）：

        - ``知识库管理系统``：``cut`` → 2 个词，``cut_for_search`` → 6 个词
          （多出 ``知识`` / ``管理`` / ``系统`` / ``管理系``）；
        - ``中华人民共和国``：``cut`` → 1 个词，``cut_for_search`` → 6 个词；
        - ``固定快照``：两者**相同**（各 2 个词）—— 不是分叉样例。

        因此导入侧若用 ``cut``、查询侧用 ``cut_for_search``，查询会对子词做 AND，而
        索引里只有整词 —— 合成词查询会查不中（影响 A22 / A26）。模式常量
        :data:`JIEBA_TOKENIZER_MODE` 是唯一定义，T08 查询侧同样取它。
        """

        if not text:
            return ""
        cut = getattr(tokenizer, "cut_for_search", None) or tokenizer.cut
        try:
            return " ".join(token for token in cut(text) if token.strip())
        except Exception:  # noqa: BLE001 -- 分词失败不能阻断迁移
            return " ".join(text.split())

    # ==================================================================
    # 第 10 步：校验 + 提交
    # ==================================================================

    async def _stage_verify(self, report: MigrationReport, source: Path) -> None:
        problems: list[str] = []
        target_root = self._target_workspace_root()

        file_mismatches = self._verify_files(report, source)
        report.validation["file_mismatches"] = file_mismatches
        if file_mismatches:
            problems.append(f"文件校验不一致：{file_mismatches}")

        problems.extend(self._verify_vectors(source, target_root / "chroma", report))

        image_mismatches = self._verify_images(source, target_root, report)
        if image_mismatches:
            problems.append(f"图片校验不一致：{image_mismatches}")

        problems.extend(await self._verify_conversations(source, report))

        # 无修改同步 / 迁移零出网（PRD 15.1 / 15.3）。
        from app.core.outbound import outbound_client

        report.validation["outbound_entries"] = len(outbound_client.entries())
        if not outbound_client.is_empty():
            problems.append("迁移期间发生出网调用（应为 0）")

        report.validation["ok"] = not problems
        report.validation["problems"] = problems
        if problems:
            raise MigrationVerificationError("；".join(problems))

    def _verify_files(self, report: MigrationReport, source: Path) -> int:
        """字节 SHA-256 与 mtime 一致（PRD 15.3 文件行）。"""

        del source  # 只比对 path_map 记录的新旧路径
        mismatches = 0
        for old, new in report.path_map.items():
            old_path, new_path = Path(old), Path(new)
            if not new_path.is_file():
                mismatches += 1
                continue
            if self._sha256(old_path) != self._sha256(new_path) or abs(old_path.stat().st_mtime - new_path.stat().st_mtime) > 1.0:
                mismatches += 1
        return mismatches

    def _verify_vectors(
        self, source: Path, target_chroma: Path, report: MigrationReport
    ) -> list[str]:
        """collection 数量与记录数量一致；记录维度与数值一致（PRD 15.3 向量行）。"""

        problems: list[str] = []
        if not (source / "chroma.sqlite3").is_file():
            return problems
        src_client, dst_client, src_names, dst_names = self._verify_clients(source, target_chroma)
        report.validation["collections_source"] = len(src_names)
        report.validation["collections_target"] = len(dst_names)
        if src_names != dst_names:
            problems.append(f"collection 数量或名称不一致：{src_names} → {dst_names}")
            return problems

        total = sum(report.collections.values())
        verify_all = (
            self._verify_vectors_fully
            if self._verify_vectors_fully is not None
            else total <= FULL_VECTOR_VERIFY_LIMIT
        )
        report.validation["vectors_verified_fully"] = verify_all
        try:
            problems.extend(
                self._compare_collections(src_client, dst_client, src_names, verify_all, report)
            )
        except ChromaInternalError:
            # Chroma 的 HNSW 分段是**懒物化**的：写入方（_stage_move_vectors）与校验方
            # 若落在不同的 PersistentClient 上，后者可能瞬时读到「sqlite 里已有
            # collection 行、分段目录还没落盘」的中间态（实测约 1/5 概率，抛
            # ``Error creating hnsw segment reader: Nothing found on disk``）。
            # 清掉 System 缓存后重开一次即可跨过这个窗口；重试仍失败则如实抛出，
            # 不掩盖真实损坏。
            SharedSystemClient.clear_system_cache()
            self._clients.clear()
            src_client, dst_client, _src, _dst = self._verify_clients(source, target_chroma)
            problems.extend(
                self._compare_collections(src_client, dst_client, src_names, verify_all, report)
            )
        return problems

    def _verify_clients(
        self, source: Path, target_chroma: Path
    ) -> tuple[Any, Any, list[str], list[str]]:
        src_client = self._chroma_client(source)
        dst_client = self._chroma_client(target_chroma)
        return (
            src_client,
            dst_client,
            self._collection_names(src_client),
            self._collection_names(dst_client),
        )

    def _compare_collections(
        self,
        src_client: Any,
        dst_client: Any,
        src_names: Sequence[str],
        verify_all: bool,
        report: MigrationReport,
    ) -> list[str]:
        problems: list[str] = []
        compared = 0
        for name in src_names:
            src_collection = src_client.get_collection(name)
            dst_collection = dst_client.get_collection(name)
            src_count = int(src_collection.count())
            dst_count = int(dst_collection.count())
            if src_count != dst_count:
                problems.append(f"collection {name} 记录数不一致：{src_count} → {dst_count}")
                continue
            limit = None if verify_all else min(64, max(1, src_count))
            if verify_all:
                raw_src = src_collection.get(include=["embeddings"])
            else:
                raw_src = src_collection.get(include=["embeddings"], limit=limit)
            src_ids = [str(item) for item in (raw_src.get("ids") or [])]
            src_embeddings = self._as_float_lists(raw_src.get("embeddings"))
            raw_dst = dst_collection.get(ids=src_ids or None, include=["embeddings"])
            dst_by_id = {
                str(block_id): vector
                for block_id, vector in zip(
                    raw_dst.get("ids") or [],
                    self._as_float_lists(raw_dst.get("embeddings")),
                    strict=False,
                )
            }
            mismatch = self._compare_vectors(src_ids, src_embeddings, dst_by_id, name)
            compared += len(src_ids)
            if mismatch:
                problems.append(mismatch)
                break
        report.validation["vectors_compared"] = compared
        return problems

    @staticmethod
    def _compare_vectors(
        ids: list[str],
        embeddings: list[list[float]],
        target: Mapping[str, list[float]],
        collection: str,
    ) -> str:
        for block_id, vector in zip(ids, embeddings, strict=False):
            other = target.get(block_id)
            if other is None or len(other) != len(vector):
                return f"collection {collection} 的 block {block_id} 缺失或维度不一致"
            if any(abs(a - b) > 1e-6 for a, b in zip(vector, other, strict=False)):
                return f"collection {collection} 的 block {block_id} 向量数值不一致"
        return ""

    def _verify_images(self, source: Path, target_root: Path, report: MigrationReport) -> int:
        """已完成识别内容与原图一致；未完成任务保留（PRD 15.3 图片行）。"""

        src_db = source / "image_state.db"
        dst_db = target_root / "image_state.db"
        if not src_db.is_file() or not dst_db.is_file():
            return 0
        with sqlite3.connect(f"file:{src_db}?mode=ro", uri=True) as src_conn:
            originals = src_conn.execute(
                "SELECT content_hash, data FROM image_originals"
            ).fetchall()
            src_tasks = int(
                src_conn.execute("SELECT COUNT(*) FROM image_tasks").fetchone()[0]
            )
        mismatches = 0
        with sqlite3.connect(f"file:{dst_db}?mode=ro", uri=True) as dst_conn:
            dst_hashes = {
                str(row[0]): bytes(row[1])
                for row in dst_conn.execute("SELECT content_hash, data FROM image_originals")
            }
            dst_tasks = int(dst_conn.execute("SELECT COUNT(*) FROM image_tasks").fetchone()[0])
        for content_hash, data in originals:
            if dst_hashes.get(str(content_hash)) != bytes(data):
                mismatches += 1
        report.images = len(originals)
        report.validation["images_compared"] = len(originals)
        report.validation["image_tasks"] = {"source": src_tasks, "target": dst_tasks}
        if src_tasks != dst_tasks:
            mismatches += 1
        return mismatches

    async def _verify_conversations(
        self, source: Path, report: MigrationReport
    ) -> list[str]:
        """内容、顺序、归属、分支、归档及引用一致（PRD 15.3 会话行）。"""

        problems: list[str] = []
        expected = self._read_json_list(source / "conversations.json")
        report.validation["conversations_source"] = len(expected)
        for raw in expected:
            conversation_id = str(raw.get("id") or "")
            record = await self._app_store.get_conversation(conversation_id)
            if record is None:
                problems.append(f"会话缺失：{conversation_id}")
                continue
            legacy_messages = raw.get("messages") or []
            stored = await self._app_store.list_messages(conversation_id)
            if len(stored) != len(legacy_messages):
                problems.append(
                    f"会话 {conversation_id} 消息数不一致："
                    f"{len(legacy_messages)} → {len(stored)}"
                )
                continue
            for index, (legacy, current) in enumerate(
                zip(legacy_messages, stored, strict=False)
            ):
                if str(legacy.get("content") or "") != current.content:
                    problems.append(f"会话 {conversation_id} 第 {index} 条消息内容不一致")
                    break
                if str(legacy.get("role") or "user") != current.role:
                    problems.append(f"会话 {conversation_id} 第 {index} 条消息角色不一致")
                    break
        return problems

    def _stage_commit(self, report: MigrationReport) -> None:
        commit_storage_version(
            self._data_dir,
            version=WORKSPACE_LAYOUT_VERSION,
            extra={
                "migration_version": MIGRATION_VERSION,
                "job_id": report.job_id,
                "backup_dir": None if report.backup_dir is None else str(report.backup_dir),
                "excluded": list(report.excluded),
                "collections": report.collections,
            },
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
