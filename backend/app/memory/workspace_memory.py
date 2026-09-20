"""``memory.json`` 工作区记忆的落地实现（PRD 11.1–11.5，Owner: T08）。

与 v1.1.0 的 ``MemoryStore``（``data_dir/memory.db``）的关系
----------------------------------------------------------
旧实现是**单库、全局**的 SQLite 表；V1.2.0 要求每工作区一个 ``memory.json``
（PRD 5.1 目录树 + 11.2 文件结构）。旧库不在这里改，迁移由 T01 的
``core/migration.py`` 按 PRD 15.2 第 8 步的类别转换表导入；本模块只认新格式，
并通过 :func:`memory_file_payload` 保证落盘字段名与顺序**逐字**等于
``contracts.MEMORY_FILE_KEYS`` / ``MEMORY_ENTRY_KEYS``（契约冻结，只读）。

三条不变量
----------
1. **原子替换**：写入走同目录 ``*.tmp`` + ``os.replace``。崩溃在替换前，用户仍拿到
   上一份完整版本；不存在"写到一半的 memory.json"（PRD 11.2）。
2. **revision 与内容同生共死**：``revision`` / ``invalidation_revision`` / 记忆条目 /
   ``processed_cursors`` / ``applied_operations`` 在**同一个文件替换**里一起提交，
   所以中断恢复时不会出现"条目写了但游标没走"或反之（PRD 11.5 / A30）。
3. **``operation_id`` 去重**：重复的后台任务带着同一 ``operation_id`` 再来时是
   **幂等空操作**（不改内容、不递增 revision），因此重放不会产生重复条目。
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.agent.contracts import (
    DEFAULT_MEMORY_MAX_CHARS,
    DEFAULT_MEMORY_SNAPSHOT_MAX_CHARS,
    MEMORY_ENTRY_KEYS,
    MEMORY_FILE_KEYS,
    MemoryEntryStatus,
    MemoryFile,
    MemoryFileEntry,
    MemoryInvalidation,
    MemoryInvalidationKind,
    MemoryKind,
    utc_now_iso,
)
from app.memory.locking import memory_maintenance_locks

__all__ = [
    "MEMORY_ENTRY_ORDER_KEYS",
    "MEMORY_SNAPSHOT_SECTIONS",
    "MemoryDisabledError",
    "MemoryFileError",
    "MemoryMutationResult",
    "MemoryRevisionConflict",
    "MemorySnapshot",
    "WorkspaceMemory",
    "build_snapshot",
    "entry_sort_key",
    "memory_file_payload",
    "new_memory_id",
    "normalize_statement",
    "snapshot_hash",
]

#: 条目字段顺序（= 契约常量，落盘顺序的唯一来源）。
MEMORY_ENTRY_ORDER_KEYS: tuple[str, ...] = MEMORY_ENTRY_KEYS

#: 快照分组顺序（PRD 11.2：用户主动保存内容 → 未完成讨论 → 最近主题；中间夹
#: 回答要求与已确认结论，因为这两类同样是"必须一直带着"的有效前提）。
MEMORY_SNAPSHOT_SECTIONS: tuple[tuple[str, tuple[MemoryKind, ...]], ...] = (
    ("用户主动保存", ()),
    (
        "回答要求",
        (MemoryKind.RESPONSE_PREFERENCE,),
    ),
    (
        "已确认结论",
        (MemoryKind.CONFIRMED_CONCLUSION,),
    ),
    (
        "讨论进展",
        (MemoryKind.PROGRESS,),
    ),
    (
        "最近主题",
        (MemoryKind.TOPIC,),
    ),
)

#: 参与快照的有效状态（forgotten / superseded / orphaned 绝不进快照）。
SNAPSHOT_STATUSES: frozenset[MemoryEntryStatus] = frozenset({MemoryEntryStatus.ACTIVE})


class MemoryFileError(RuntimeError):
    """``memory.json`` 缺失以外的读取/校验失败（不静默丢弃用户记忆）。"""


class MemoryRevisionConflict(MemoryFileError):
    """乐观并发：调用方持有的 ``revision`` 与文件当前值不一致（应由 API 映射 409）。"""


class MemoryDisabledError(MemoryFileError):
    """工作区记忆已关闭；自动提炼与记忆写入停止（PRD 11.4）。"""


def new_memory_id() -> str:
    """一条记忆的稳定 ID。"""

    return f"mem-{uuid4().hex}"


def normalize_statement(text: str) -> str:
    """陈述归一化：折叠空白（用于同内容去重，不做语义近似）。"""

    return " ".join(str(text or "").split())


def snapshot_hash(text: str) -> str:
    """快照正文 hash（PRD 11.3 ``memory_snapshot_hash``）。"""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def entry_sort_key(entry: MemoryFileEntry) -> tuple[str, str, str]:
    """分组内排序：新的在前（``updated_at`` → ``created_at`` → ``id`` 兜底确定）。"""

    return (
        entry.updated_at or entry.created_at,
        entry.created_at,
        entry.id,
    )


def memory_file_payload(memory: MemoryFile) -> dict[str, Any]:
    """按契约冻结的字段顺序导出 ``memory.json`` 的 dict。

    ``model_dump`` 已经按字段定义顺序输出，但这里**显式重排**并断言键集合相等：
    契约新增/改名时要在提交前炸掉，而不是悄悄写出一份字段不全的文件。
    """

    dumped = memory.model_dump(mode="json")
    if set(dumped) != set(MEMORY_FILE_KEYS):
        raise MemoryFileError(
            f"memory.json 字段与契约不一致：{sorted(dumped)} != {sorted(MEMORY_FILE_KEYS)}"
        )
    payload: dict[str, Any] = {}
    for key in MEMORY_FILE_KEYS:
        value = dumped[key]
        if key == "entries":
            value = [_ordered_entry(entry) for entry in value]
        payload[key] = value
    return payload


def _ordered_entry(entry: dict[str, Any]) -> dict[str, Any]:
    if set(entry) != set(MEMORY_ENTRY_ORDER_KEYS):
        raise MemoryFileError(
            f"记忆条目字段与契约不一致：{sorted(entry)} != {sorted(MEMORY_ENTRY_ORDER_KEYS)}"
        )
    return {key: entry[key] for key in MEMORY_ENTRY_ORDER_KEYS}


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """固定会话快照（PRD 11.3）：会话开始时算一次，之后不再自动替换。"""

    revision: int
    invalidation_revision: int
    text: str
    hash: str
    at: str
    truncated: bool = False
    overflow_count: int = 0
    #: 用户主动保存的内容超过单条上限 → 界面显示完整管理入口并请求整理（PRD 11.2）。
    needs_tidy: bool = False

    @property
    def empty(self) -> bool:
        return not self.text


def build_snapshot(
    memory: MemoryFile,
    *,
    at: str | None = None,
    max_chars: int = DEFAULT_MEMORY_SNAPSHOT_MAX_CHARS,
    entry_max_chars: int = DEFAULT_MEMORY_MAX_CHARS,
) -> MemorySnapshot:
    """构造会话快照正文（PRD 11.2 / 11.3）。

    - 只取 ``active`` 条目：``forgotten`` / ``superseded`` / ``orphaned`` 不得进入
      模型上下文（"自动流程不得恢复已遗忘条目"在**读取侧**也必须成立）。
    - 关闭记忆时快照为空正文（PRD 11.4：``enabled=false`` 与失效版本同时提交）。
    - 超容量时截断并回报 ``truncated`` / ``overflow_count``，让调用方给出历史查询
      与整理入口，而不是静默丢内容。
    """

    at = at or utc_now_iso()
    if not memory.enabled:
        return MemorySnapshot(
            revision=memory.revision,
            invalidation_revision=memory.invalidation_revision,
            text="",
            hash=snapshot_hash(""),
            at=at,
        )

    needs_tidy = any(
        entry.user_pinned and len(entry.text) > entry_max_chars
        for entry in memory.entries
        if entry.status in SNAPSHOT_STATUSES
    )

    lines: list[str] = [f"工作区记忆（revision {memory.revision}）"]
    overflow = 0
    used = len(lines[0])
    truncated = False

    for title, kinds in MEMORY_SNAPSHOT_SECTIONS:
        if title == "用户主动保存":
            selected = [
                entry
                for entry in memory.entries
                if entry.status in SNAPSHOT_STATUSES and entry.user_pinned
            ]
        else:
            selected = [
                entry
                for entry in memory.entries
                if entry.status in SNAPSHOT_STATUSES
                and not entry.user_pinned
                and entry.kind in kinds
            ]
        selected.sort(key=entry_sort_key, reverse=True)
        if not selected:
            continue
        # 先试算整节，整节放不下时逐条放，保证"能放几条是几条"而不是整节丢弃。
        header = f"【{title}】"
        candidate: list[str] = []
        for entry in selected:
            bullet = f"- {entry.text}"
            candidate.append(bullet)
        section_lines = [header, *candidate]
        section_len = sum(len(line) + 1 for line in section_lines)
        if used + section_len <= max_chars:
            lines.extend(section_lines)
            used += section_len
            continue
        lines.append(header)
        used += len(header) + 1
        for bullet in candidate:
            cost = len(bullet) + 1
            if used + cost > max_chars:
                truncated = True
                overflow += 1
                continue
            lines.append(bullet)
            used += cost

    text = "\n".join(lines) + "\n"
    return MemorySnapshot(
        revision=memory.revision,
        invalidation_revision=memory.invalidation_revision,
        text=text,
        hash=snapshot_hash(text),
        at=at,
        truncated=truncated,
        overflow_count=overflow,
        needs_tidy=needs_tidy,
    )


@dataclass(frozen=True, slots=True)
class MemoryMutationResult:
    """一次记忆维护的结果。"""

    #: False = 幂等空操作（``operation_id`` 已应用 / 开关关闭 / 无匹配条目）。
    applied: bool
    memory: MemoryFile
    entry: MemoryFileEntry | None = None
    reason: str = ""
    #: 受影响会话（PRD 11.4：需要递增它们的 ``context_epoch``）。
    affected_conversation_ids: tuple[str, ...] = ()
    invalidations: tuple[MemoryInvalidation, ...] = field(default=())


class WorkspaceMemory:
    """一个工作区的 ``memory.json`` 访问层。

    **所有写操作都在 PRD 11.2 的锁顺序内完成**「读当前完整版本 → 应用经校验的更新 →
    增加 revision → 原子替换」，调用方不需要（也**不应该**）自己拼锁与写文件。
    """

    def __init__(
        self,
        path: Path | str,
        *,
        storage_key: str | None = None,
        data_dir: Path | str | None = None,
    ) -> None:
        self._path = Path(path)
        self._storage_key = storage_key or self._path.parent.name
        self._data_dir = Path(data_dir) if data_dir is not None else None

    # ---------------------------------------------------------------- 基本属性

    @property
    def path(self) -> Path:
        return self._path

    @property
    def storage_key(self) -> str:
        return self._storage_key

    @property
    def data_dir(self) -> Path | None:
        return self._data_dir

    def exists(self) -> bool:
        return self._path.is_file()

    # ---------------------------------------------------------------- 读

    def load(self) -> MemoryFile:
        """读取当前版本；文件不存在返回空 ``MemoryFile``，损坏则**显式报错**。

        刻意不在损坏时返回空对象：那等于把用户的记忆静默清空，且下一次写入会把
        空版本覆盖到磁盘上，原始内容再也找不回来。
        """

        if not self._path.is_file():
            return MemoryFile()
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MemoryFileError(f"memory.json 无法解析：{self._path}（{exc}）") from exc
        if not isinstance(raw, dict):
            raise MemoryFileError(f"memory.json 根对象必须是对象：{self._path}")
        try:
            return MemoryFile.model_validate(raw)
        except ValueError as exc:
            raise MemoryFileError(f"memory.json 不符合契约：{self._path}（{exc}）") from exc

    def snapshot(
        self,
        *,
        max_chars: int = DEFAULT_MEMORY_SNAPSHOT_MAX_CHARS,
    ) -> MemorySnapshot:
        return build_snapshot(self.load(), max_chars=max_chars)

    def has_applied(self, operation_id: str) -> bool:
        if not operation_id:
            return False
        return operation_id in self.load().applied_operations

    def active_entries(self, memory: MemoryFile | None = None) -> list[MemoryFileEntry]:
        source = memory if memory is not None else self.load()
        return [entry for entry in source.entries if entry.status in SNAPSHOT_STATUSES]

    # ---------------------------------------------------------------- 写

    def write(self, memory: MemoryFile, *, expected_revision: int | None = None) -> MemoryFile:
        """原子替换整份文件（不带锁；调用方须已在维护锁内）。"""

        if expected_revision is not None and memory.revision != expected_revision:
            raise MemoryRevisionConflict(
                f"记忆版本已变化：期望 {expected_revision}，实际 {memory.revision}"
            )
        memory.updated_at = utc_now_iso()
        payload = memory_file_payload(memory)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(tmp, self._path)
        return memory

    # ---- 内部：加锁读改写 ------------------------------------------------

    def _mutate(
        self,
        mutate: Any,
        *,
        conversation_id: str | None = None,
        expected_revision: int | None = None,
        allow_disabled: bool = False,
        bump_invalidation: bool = False,
    ) -> MemoryMutationResult:
        storage_keys = [self._storage_key]
        with memory_maintenance_locks(conversation_id, storage_keys=storage_keys):
            memory = self.load()
            if not memory.enabled and not allow_disabled:
                raise MemoryDisabledError("工作区记忆已关闭，写入被拒绝（PRD 11.4）")
            # 乐观并发必须在**递增之前**比对：递增后的值永远不等于调用方持有的旧值。
            if expected_revision is not None and memory.revision != expected_revision:
                raise MemoryRevisionConflict(
                    f"记忆版本已变化：期望 {expected_revision}，实际 {memory.revision}"
                )
            before = memory.model_copy(deep=True)
            result: MemoryMutationResult = mutate(memory)
            if not result.applied:
                return result
            if memory == before:
                # 内容没变就不递增 revision：revision 是「内容版本」，不是「写入次数」，
                # 否则一次幂等重放会让所有会话的快照无谓失效。
                return result
            memory.revision = before.revision + 1
            if bump_invalidation:
                memory.invalidation_revision = before.invalidation_revision + 1
            self.write(memory)
            return result

    # ---- 条目级操作 ------------------------------------------------------

    def add_entry(
        self,
        text: str,
        kind: MemoryKind | str,
        *,
        source_conversation_id: str = "",
        source_message_ids: Sequence[str] = (),
        user_pinned: bool = False,
        operation_id: str = "",
        conversation_id: str | None = None,
        expected_revision: int | None = None,
        auto: bool | None = None,
    ) -> MemoryMutationResult:
        """新增一条记忆（PRD 11.1 / 11.2）。

        - ``user_pinned=True`` 是用户主动保存：**不截断**，超上限时由快照侧给出整理入口。
        - 自动提炼的条目：超过单条 300 字符上限时按上限截断。
        - 同 ``kind`` + 同归一化文本视为同一条：刷新时间与来源，不产生重复条目；
          已 ``forgotten`` 的同内容条目**不复活**（PRD 11.2：自动流程不得恢复已遗忘条目）。
        """

        resolved_kind = MemoryKind(kind)
        statement = normalize_statement(text)
        if not statement:
            return MemoryMutationResult(False, MemoryFile(), reason="空陈述")
        pinned = bool(user_pinned)
        auto_built = (not pinned) if auto is None else bool(auto)
        if auto_built and len(statement) > DEFAULT_MEMORY_MAX_CHARS:
            statement = statement[:DEFAULT_MEMORY_MAX_CHARS]

        def mutate(memory: MemoryFile) -> MemoryMutationResult:
            if operation_id and operation_id in memory.applied_operations:
                return MemoryMutationResult(
                    False, memory, reason=f"operation_id 已应用：{operation_id}"
                )
            existing = _find_duplicate(memory, resolved_kind, statement)
            now = utc_now_iso()
            if existing is not None:
                if existing.status is MemoryEntryStatus.FORGOTTEN:
                    return MemoryMutationResult(
                        False, memory, entry=existing, reason="同内容条目已被用户遗忘，不复活"
                    )
                merged = sorted(set(existing.source_message_ids) | set(source_message_ids))
                changed = merged != existing.source_message_ids or (
                    pinned and not existing.user_pinned
                )
                new_operation = bool(operation_id) and operation_id not in memory.applied_operations
                if not changed and not new_operation:
                    # 真正的幂等重放：一个字节都不动，revision 也不涨。
                    return MemoryMutationResult(
                        False, memory, entry=existing, reason="已存在同内容条目"
                    )
                existing.source_message_ids = merged
                existing.updated_at = now
                if pinned:
                    existing.user_pinned = True
                if new_operation:
                    memory.applied_operations.append(operation_id)
                return MemoryMutationResult(True, memory, entry=existing, reason="已存在同内容条目")
            entry = MemoryFileEntry(
                id=new_memory_id(),
                kind=resolved_kind,
                text=statement,
                source_conversation_id=source_conversation_id,
                source_message_ids=list(source_message_ids),
                created_at=now,
                updated_at=now,
                user_pinned=pinned,
                status=MemoryEntryStatus.ACTIVE,
                operation_id=operation_id,
            )
            memory.entries.append(entry)
            if operation_id and operation_id not in memory.applied_operations:
                memory.applied_operations.append(operation_id)
            return MemoryMutationResult(True, memory, entry=entry)

        return self._mutate(
            mutate,
            conversation_id=conversation_id or source_conversation_id or None,
            expected_revision=expected_revision,
            # 新增记忆**不是失效事件**：PRD 11.3 要求自动更新后"当前会话快照保持不变，
            # 新会话读取新版本"。只有人工纠正/删除/关闭才涨 invalidation_revision（11.4）。
            bump_invalidation=False,
        )

    def update_entry(
        self,
        entry_id: str,
        *,
        text: str | None = None,
        kind: MemoryKind | str | None = None,
        conversation_id: str | None = None,
        expected_revision: int | None = None,
        replacement: str = "",
    ) -> MemoryMutationResult:
        """人工纠正一条记忆：更新内容并**登记失效记录**（PRD 11.4）。

        纠正必须留下 ``MemoryInvalidation``，否则"模型读取经过失效规则处理的历史视图"
        无从知道哪条要求已被撤销。
        """

        def mutate(memory: MemoryFile) -> MemoryMutationResult:
            entry = _require_entry(memory, entry_id)
            now = utc_now_iso()
            new_text = normalize_statement(text) if text is not None else entry.text
            if not new_text:
                return MemoryMutationResult(False, memory, entry=entry, reason="空陈述")
            invalidation = MemoryInvalidation(
                memory_id=entry.id,
                source_message_ids=list(entry.source_message_ids),
                quote=entry.text,
                kind=MemoryInvalidationKind.CORRECTED,
                replacement=replacement or new_text,
                created_at=now,
            )
            memory.invalidations.append(invalidation)
            entry.text = new_text
            if kind is not None:
                entry.kind = MemoryKind(kind)
            entry.updated_at = now
            return MemoryMutationResult(
                True,
                memory,
                entry=entry,
                affected_conversation_ids=(entry.source_conversation_id,),
                invalidations=(invalidation,),
            )

        return self._mutate(
            mutate,
            conversation_id=conversation_id,
            expected_revision=expected_revision,
            # 开关关闭只停**自动**写入（PRD 11.4「自动提炼及记忆写入停止」）。用户手动
            # 纠正自己的记忆属于数据整理：此时禁掉会把用户的错误数据锁死在文件里。
            allow_disabled=True,
            bump_invalidation=True,
        )

    def forget_entry(
        self,
        entry_id: str,
        *,
        conversation_id: str | None = None,
        expected_revision: int | None = None,
    ) -> MemoryMutationResult:
        """遗忘一条记忆：状态转 ``forgotten`` 且保留记录（PRD 11.4 / v1.1.0 D7）。"""

        def mutate(memory: MemoryFile) -> MemoryMutationResult:
            entry = _require_entry(memory, entry_id)
            now = utc_now_iso()
            invalidation = MemoryInvalidation(
                memory_id=entry.id,
                source_message_ids=list(entry.source_message_ids),
                quote=entry.text,
                kind=MemoryInvalidationKind.FORGOTTEN,
                replacement="",
                created_at=now,
            )
            memory.invalidations.append(invalidation)
            entry.status = MemoryEntryStatus.FORGOTTEN
            entry.updated_at = now
            return MemoryMutationResult(
                True,
                memory,
                entry=entry,
                affected_conversation_ids=(entry.source_conversation_id,),
                invalidations=(invalidation,),
            )

        return self._mutate(
            mutate,
            conversation_id=conversation_id,
            expected_revision=expected_revision,
            # 同上：遗忘是用户对自己数据的处置权，不因开关关闭而失效。
            allow_disabled=True,
            bump_invalidation=True,
        )

    def supersede_entry(
        self,
        entry_id: str,
        by_id: str,
        *,
        expected_revision: int | None = None,
    ) -> MemoryMutationResult:
        """把一条记忆标为被 ``by_id`` 取代。"""

        def mutate(memory: MemoryFile) -> MemoryMutationResult:
            entry = _require_entry(memory, entry_id)
            if by_id == entry_id:
                return MemoryMutationResult(False, memory, entry=entry, reason="不能自我取代")
            _require_entry(memory, by_id)
            entry.status = MemoryEntryStatus.SUPERSEDED
            entry.superseded_by = by_id
            entry.updated_at = utc_now_iso()
            return MemoryMutationResult(
                True,
                memory,
                entry=entry,
                affected_conversation_ids=(entry.source_conversation_id,),
            )

        # 取代同类旧结论同样是用户可见的数据整理，不因开关关闭而阻断。
        return self._mutate(
            mutate,
            expected_revision=expected_revision,
            allow_disabled=True,
            bump_invalidation=True,
        )

    # ---- 开关 ------------------------------------------------------------

    def set_enabled(
        self,
        enabled: bool,
        *,
        expected_revision: int | None = None,
    ) -> MemoryMutationResult:
        """开关工作区记忆（PRD 11.4）。

        关闭时 **``enabled=false`` 与失效版本同时提交**（同一个原子替换），并返回全部
        受影响会话 —— 调用方据此把它们的快照置空。
        """

        def mutate(memory: MemoryFile) -> MemoryMutationResult:
            if memory.enabled == enabled:
                return MemoryMutationResult(False, memory, reason="开关无变化")
            memory.enabled = enabled
            affected = tuple(
                sorted(
                    {
                        entry.source_conversation_id
                        for entry in memory.entries
                        if entry.source_conversation_id
                    }
                )
            )
            invalidations: tuple[MemoryInvalidation, ...] = ()
            if not enabled:
                invalidation = MemoryInvalidation(
                    memory_id="",
                    source_message_ids=[],
                    quote="",
                    kind=MemoryInvalidationKind.DISABLED,
                    replacement="",
                    created_at=utc_now_iso(),
                )
                invalidations = (invalidation,)
            return MemoryMutationResult(
                True, memory, affected_conversation_ids=affected, invalidations=invalidations
            )

        return self._mutate(
            mutate,
            expected_revision=expected_revision,
            allow_disabled=True,
            bump_invalidation=True,
        )

    # ---- 提炼提交（PRD 11.5 / A30）-------------------------------------

    def apply_operations(
        self,
        *,
        operation_id: str,
        conversation_id: str,
        cursor_message_id: str = "",
        candidates: Iterable[tuple[Any, ...]] = (),
        source_message_ids: Sequence[str] = (),
        expected_revision: int | None = None,
    ) -> MemoryMutationResult:
        """一次提炼的**单次原子提交**（PRD 11.5）：条目 + 游标 + 已应用操作 ID。

        这是一处刻意的"批量接口"：如果用 ``add_entry`` 逐条写，进程在中途被杀死就会
        留下"有条目但没有游标/操作 ID"的半成品，恢复后重复提炼会产生重复条目（A30）。

        ``candidates`` 每项是 ``(text, kind)`` 或 ``(text, kind, source_message_ids)``：
        逐条携带来源，才能保证"记忆引用指向来源会话和消息"（PRD 11.1）。
        """

        def mutate(memory: MemoryFile) -> MemoryMutationResult:
            if operation_id in memory.applied_operations:
                return MemoryMutationResult(
                    False, memory, reason=f"operation_id 已应用：{operation_id}"
                )
            now = utc_now_iso()
            added: list[MemoryFileEntry] = []
            for candidate in candidates:
                text, kind = candidate[0], candidate[1]
                own_sources = list(
                    candidate[2] if len(candidate) > 2 else source_message_ids
                )
                resolved = MemoryKind(kind)
                statement = normalize_statement(text)
                if not statement:
                    continue
                if len(statement) > DEFAULT_MEMORY_MAX_CHARS:
                    statement = statement[:DEFAULT_MEMORY_MAX_CHARS]
                existing = _find_duplicate(memory, resolved, statement)
                if existing is not None:
                    if existing.status is MemoryEntryStatus.FORGOTTEN:
                        continue
                    merged = sorted(set(existing.source_message_ids) | set(own_sources))
                    if merged != existing.source_message_ids:
                        existing.source_message_ids = merged
                    continue
                entry = MemoryFileEntry(
                    id=new_memory_id(),
                    kind=resolved,
                    text=statement,
                    source_conversation_id=conversation_id,
                    source_message_ids=own_sources,
                    created_at=now,
                    updated_at=now,
                    operation_id=operation_id,
                )
                memory.entries.append(entry)
                added.append(entry)
            if cursor_message_id:
                memory.processed_cursors[conversation_id] = cursor_message_id
            memory.applied_operations.append(operation_id)
            return MemoryMutationResult(
                True,
                memory,
                entry=added[-1] if added else None,
                reason=f"新增 {len(added)} 条",
            )

        return self._mutate(
            mutate,
            conversation_id=conversation_id or None,
            expected_revision=expected_revision,
        )

    def apply_source_deletion(
        self,
        conversation_id: str,
        *,
        operation_id: str = "",
        expected_revision: int | None = None,
    ) -> MemoryMutationResult:
        """来源会话被删除：有效派生记忆转 ``orphaned``（PRD 11.5）。

        ``forgotten`` / ``superseded`` 与人工失效记录**保持原状态**（用户的遗忘意图与
        已登记的失效结论优先于系统的来源标记）。
        """

        def mutate(memory: MemoryFile) -> MemoryMutationResult:
            if operation_id and operation_id in memory.applied_operations:
                return MemoryMutationResult(
                    False, memory, reason=f"operation_id 已应用：{operation_id}"
                )
            now = utc_now_iso()
            orphaned: list[MemoryFileEntry] = []
            for entry in memory.entries:
                if entry.source_conversation_id != conversation_id:
                    continue
                if entry.status is MemoryEntryStatus.ACTIVE:
                    entry.status = MemoryEntryStatus.ORPHANED
                    entry.updated_at = now
                    orphaned.append(entry)
            if operation_id:
                memory.applied_operations.append(operation_id)
            if not orphaned and not operation_id:
                return MemoryMutationResult(False, memory, reason="无来源匹配条目")
            return MemoryMutationResult(
                True,
                memory,
                entry=orphaned[0] if orphaned else None,
                affected_conversation_ids=(conversation_id,),
                reason=f"转 orphaned {len(orphaned)} 条",
            )

        # 来源会话被删除后的清理是**系统正确性**要求（PRD 11.5 登记了持久后台任务），
        # 不能因为用户同时关了记忆开关就停在半路：那会让已删除会话的记忆继续可注入。
        return self._mutate(
            mutate,
            expected_revision=expected_revision,
            allow_disabled=True,
            bump_invalidation=True,
        )


def _require_entry(memory: MemoryFile, entry_id: str) -> MemoryFileEntry:
    for entry in memory.entries:
        if entry.id == entry_id:
            return entry
    raise MemoryFileError(f"记忆条目不存在：{entry_id}")


def _find_duplicate(
    memory: MemoryFile, kind: MemoryKind, statement: str
) -> MemoryFileEntry | None:
    """同 ``kind`` + 同归一化文本的既有条目（任何状态都算，因为"遗忘过"必须可见）。"""

    for entry in memory.entries:
        if entry.kind is kind and normalize_statement(entry.text) == statement:
            return entry
    return None
