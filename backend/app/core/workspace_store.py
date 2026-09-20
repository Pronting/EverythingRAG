"""工作区存储：data_dir/workspaces.json 持久化（对话分区，对齐 Codex 客户端）。

设计要点：
- 工作区是**轻量目录**：只存 id / name / 时间戳，对话通过 ``Conversation.workspace_id`` 归属；
- 始终保证存在唯一「默认」工作区（``is_default`` 且不可删除、不可改名冲突），
  旧数据（无 workspace_id 的会话）自动落到默认工作区；
- 删除工作区**不删对话**：其中的对话全部归档，可在设置页「归档对话」找回。

名称规则（``default`` / ``general`` / ``new``）沿用本地单用户习惯：中文名直接用，
英文名转小写连字符 slug 并去重；空名回退占位。

V1.2.0 工作区知识隔离（PRD 5.1 / 5.2，T01）：

- 每个工作区新增**服务端生成**的 ``storage_key``：稳定、唯一，目录树由它构建
  （``data_dir/workspaces/<storage_key>/``）。客户端只提交 ``workspace_id``，
  服务端验证归属后解析目录 —— 路径永远不出自客户端。
- 工作区 id 语义**保持不变**（沿用 slug 推导，既有会话归属不受影响）；PRD 5.1 的
  「新增工作区使用稳定 UUID」在 :meth:`WorkspaceStore.create_with_uuid` 与 API 层
  落地，避免破坏既有 id 与会话归属。
- 历史数据（``workspaces.json`` 里没有 ``storage_key`` 的行）按 id 确定性派生
  （``uuid5``），因此**重复加载得到同一个 key**，幂等回填、无需用户干预。
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import BaseModel

from app.agent.contracts import (
    DEFAULT_PERMISSION_MODE,
    DEFAULT_REASONING_LEVEL,
    PermissionMode,
    ReasoningLevel,
    WorkspaceStatus,
)

#: 默认工作区 id（旧会话 / 未指定归属时落这里）。
DEFAULT_WORKSPACE_ID = "default"
DEFAULT_WORKSPACE_NAME = "默认"
#: 默认工作区的 ``storage_key``：可读、稳定，且与 UUID key 天然不冲突。
DEFAULT_STORAGE_KEY = "default"
#: 名称长度上限（与前端 maxLength 对齐）。
MAX_WORKSPACE_NAME_CHARS = 20
#: 每工作区目录树（PRD 5.1）中**必须存在**的子目录；``state.db`` / ``image_state.db`` /
#: ``memory.json`` / ``checkpoints.db`` 是文件，由各自拥有者在首次写入时创建。
WORKSPACE_SUBDIRS: tuple[str, ...] = ("documents", "chroma", "artifacts", "cache/jieba")
#: 目录布局版本：1 = V1.1.0 的 ``data_dir`` 根布局；2 = PRD 5.1 的每工作区布局。
WORKSPACE_LAYOUT_VERSION = 2


def _new_storage_key() -> str:
    """生成新的 ``storage_key``：稳定 UUID4 hex（目录名安全、无分隔符）。"""

    return uuid4().hex


def _derive_storage_key(workspace_id: str) -> str:
    """由 id 确定性派生 ``storage_key``（历史数据回填用，重复调用结果相同）。"""

    if workspace_id == DEFAULT_WORKSPACE_ID:
        return DEFAULT_STORAGE_KEY
    return uuid5(NAMESPACE_URL, f"everything-rag:workspace:{workspace_id}").hex


def derive_storage_key(workspace_id: str) -> str:
    """公开别名：迁移为「缺失工作区记录」新建归档工作区时用它派生稳定 key。"""

    return _derive_storage_key(workspace_id)


class Workspace(BaseModel):
    """一个对话分区。"""

    id: str
    name: str
    created_at: str = ""
    updated_at: str = ""
    #: 默认工作区：不可删除，删/改名都会被拒绝或忽略。
    is_default: bool = False
    #: 置顶：排在其他分区之前（对齐 Codex 的「别针」图标）。
    pinned: bool = False
    #: 手工拖拽排序位；None = 未排过，按创建时间。
    order_index: float | None = None
    #: 服务端生成的稳定目录键（PRD 5.1）：``data_dir/workspaces/<storage_key>/``。
    #: 缺省时由 id 确定性派生后回填（见 ``_derive_storage_key``）。
    storage_key: str = ""
    #: 工作区状态（PRD 5.4）：删除采用可恢复的归档处理，而非物理删除。
    status: WorkspaceStatus = WorkspaceStatus.ACTIVE
    #: 工作区默认权限模式（PRD 10.1 / 18.1，默认 ask）。
    default_permission_mode: PermissionMode = DEFAULT_PERMISSION_MODE
    #: 工作区默认思考强度（PRD 9.1 / 18.1，默认 high）。
    default_reasoning_level: ReasoningLevel = DEFAULT_REASONING_LEVEL
    #: 本区知识版本（PRD 5.3.6）：导入/同步/删除提交成功后递增；缓存键的一部分。
    knowledge_revision: int = 0


def _now() -> str:
    return datetime.now(UTC).isoformat()


def workspace_order_key(item: Any) -> tuple[int, float, str]:
    """侧边栏工作区顺序的**唯一定义**：置顶在前 → 手工位次 → 创建时间。

    ``order_index`` 为 None（从未拖拽过）的用 ``+inf`` 排在手工项之后，仍按创建时间先后；
    三者类型统一为 (int, float, str)，比较时不会混类型。

    入参可以是存档行 :class:`Workspace`，也可以是 ``app.db`` 的 ``WorkspaceRecord`` /
    同名字段 mapping（D29：迁移后 pin/order 由 app.db 权威列提供，排序规则只有这一份）。
    """

    def field(name: str, default: Any = None) -> Any:
        if isinstance(item, Mapping):
            return item.get(name, default)
        return getattr(item, name, default)

    order = field("order_index")
    return (
        0 if field("pinned", False) else 1,
        float(order) if order is not None else float("inf"),
        str(field("created_at", "") or ""),
    )


def _workspace_sort_key(workspace: Workspace) -> tuple[int, float, str]:
    """兼容别名：转发到 :func:`workspace_order_key`（排序规则只有一份）。"""

    return workspace_order_key(workspace)


def _slugify(name: str) -> str:
    """把名称转成可用 id：整串英文/数字转小写连字符，否则回退自动 id。

    混合名称（如「工作区 A」）若只剥出余下的 ASCII 字母会得到 "a" 这种无意义 id，
    因此这里要求整串都是 ASCII 字母数字才做 slug，其余交给 ``workspace[-N]`` 兜底。
    """

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\s_-]*", name.strip()):
        return ""
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")[:40]


def workspace_from_record(record: Any) -> Workspace:
    """把 ``app.db`` ``workspaces`` 行转成存档 :class:`Workspace`（单向适配，D19/A3）。

    接受 ``WorkspaceRecord`` 或同名字段的 mapping（例如读路径用的只读 SQL 行）。
    ``pinned`` / ``order_index`` 缺失时取默认值（``False`` / ``None``），因此 **v1 老库
    升级前后的行都能安全转换**。
    """

    def field(name: str, default: Any = None) -> Any:
        if isinstance(record, Mapping):
            return record.get(name, default)
        return getattr(record, name, default)

    workspace_id = str(field("id", "") or "")
    return Workspace(
        id=workspace_id,
        name=str(field("name", "") or ""),
        created_at=str(field("created_at", "") or ""),
        updated_at=str(field("updated_at", "") or ""),
        is_default=workspace_id == DEFAULT_WORKSPACE_ID,
        pinned=bool(field("pinned", False)),
        order_index=field("order_index"),
        storage_key=str(field("storage_key", "") or "") or _derive_storage_key(workspace_id),
        status=WorkspaceStatus(str(field("status", WorkspaceStatus.ACTIVE.value))),
        default_permission_mode=PermissionMode(
            str(field("default_permission_mode", DEFAULT_PERMISSION_MODE.value))
        ),
        default_reasoning_level=ReasoningLevel(
            str(field("default_reasoning_level", DEFAULT_REASONING_LEVEL.value))
        ),
        knowledge_revision=int(field("knowledge_revision", 0) or 0),
    )


class WorkspaceStore:
    """workspaces.json 的读写（thread-safe，低频写）。"""

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "workspaces.json"
        self._lock = threading.Lock()
        self._workspaces: dict[str, Workspace] = {}
        self._load()
        with self._lock:
            if DEFAULT_WORKSPACE_ID not in self._workspaces:
                self._workspaces[DEFAULT_WORKSPACE_ID] = self._default_workspace()
                self._persist()

    @property
    def path(self) -> Path:
        return self._path

    def _default_workspace(self) -> Workspace:
        now = _now()
        return Workspace(
            id=DEFAULT_WORKSPACE_ID,
            name=DEFAULT_WORKSPACE_NAME,
            created_at=now,
            updated_at=now,
            is_default=True,
            # 默认分区初始排最前；用户拖拽/置顶后按新的位次走
            order_index=0.0,
            storage_key=DEFAULT_STORAGE_KEY,
        )

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for item in raw:
                workspace = Workspace.model_validate(item)
                self._workspaces[workspace.id] = workspace
        except (OSError, json.JSONDecodeError):
            self._workspaces = {}
            return
        self._backfill_storage_keys()

    def _backfill_storage_keys(self) -> None:
        """回填历史缺失的 ``storage_key`` 并保证唯一（PRD 5.1）。

        历史行按 id **确定性派生**（:func:`_derive_storage_key`）：同一个 id 每次加载都
        得到同一个 key，因此**刻意不落盘** —— 否则「只是构造 ``WorkspaceStore``」（例如
        ``import app.api.deps`` 会在模块级构造它）就会改写用户的 ``workspaces.json``。
        只有出现 key 撞车、必须换成随机新 key 时才写盘，否则下次加载会得到不同的 key。
        """

        repaired = False
        seen: set[str] = set()
        for workspace in self._workspaces.values():
            key = workspace.storage_key or _derive_storage_key(workspace.id)
            if key in seen:
                key = _new_storage_key()
                repaired = True
            seen.add(key)
            workspace.storage_key = key
        if repaired:
            self._persist()

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [workspace.model_dump() for workspace in self._workspaces.values()]
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)

    def list(self) -> list[Workspace]:
        """返回工作区列表：置顶在前 → 手工位次 → 创建时间（默认工作区不特殊对待）。

        默认工作区也参与排序（用户可以把它拖到别处或置顶），只保证它不可删除。
        """

        with self._lock:
            return sorted(self._workspaces.values(), key=_workspace_sort_key)

    def set_pinned(self, workspace_id: str, pinned: bool) -> Workspace | None:
        """置顶 / 取消置顶（不改动手工位次，取消后回到原位次）。"""

        with self._lock:
            workspace = self._workspaces.get(workspace_id)
            if workspace is None:
                return None
            workspace.pinned = pinned
            workspace.updated_at = _now()
            self._persist()
            return workspace

    def reorder(self, ordered_ids: list[str]) -> list[Workspace]:
        """按给定顺序重写手工位次（只认已存在的 id，未列出的保持原位次）。

        置顶与否由 ``pinned`` 单独表达，因此重排只写 order_index：
        调用方传进来的就是「当前可见顺序」，置顶项若混在其中也按它给的位次落位。
        """

        with self._lock:
            for position, workspace_id in enumerate(ordered_ids):
                workspace = self._workspaces.get(workspace_id)
                if workspace is None:
                    continue
                workspace.order_index = float(position)
            self._persist()
            return sorted(self._workspaces.values(), key=_workspace_sort_key)

    def get(self, workspace_id: str) -> Workspace | None:
        with self._lock:
            return self._workspaces.get(workspace_id)

    def ensure_default(self) -> Workspace:
        """返回默认工作区（启动/迁移时确保存在）。"""

        with self._lock:
            existing = self._workspaces.get(DEFAULT_WORKSPACE_ID)
            if existing is not None:
                return existing
            workspace = self._default_workspace()
            self._workspaces[workspace.id] = workspace
            self._persist()
            return workspace

    def create(self, name: str, workspace_id: str | None = None) -> Workspace:
        """新建工作区；id 缺省由名称推导（冲突则追加序号），名称为空回退「新工作区」。"""

        cleaned = name.strip()[:MAX_WORKSPACE_NAME_CHARS] or "新工作区"
        with self._lock:
            candidate = (workspace_id or _slugify(cleaned)).strip()
            if candidate == "" or candidate in self._workspaces:
                candidate = self._unique_id(candidate or "workspace")
            now = _now()
            # 新分区排到末尾：位次取当前最大 +1，避免和已有手工位次打平
            existing_indexes = [
                item.order_index for item in self._workspaces.values() if item.order_index is not None
            ]
            next_index = (max(existing_indexes) + 1.0) if existing_indexes else float(len(self._workspaces))
            workspace = Workspace(
                id=candidate,
                name=cleaned,
                created_at=now,
                updated_at=now,
                order_index=next_index,
                storage_key=_new_storage_key(),
            )
            self._workspaces[workspace.id] = workspace
            self._persist()
            return workspace

    def create_with_uuid(self, name: str, workspace_id: str | None = None) -> Workspace:
        """新建工作区并分配**稳定 UUID id**（PRD 5.1：新增工作区使用稳定 UUID）。

        :meth:`create` 保留「按名称推导 id」的既有语义（历史数据与既有会话归属依赖
        它，且已有单测锁死）；PRD 5.1 的 UUID 要求由本方法在 API 层落地。
        """

        return self.create(name, workspace_id or uuid4().hex)

    def get_by_storage_key(self, storage_key: str) -> Workspace | None:
        """按目录键反查工作区（目录 → 归属校验用）。"""

        with self._lock:
            for workspace in self._workspaces.values():
                if workspace.storage_key == storage_key:
                    return workspace
            return None

    def adopt(self, workspace: Workspace) -> Workspace:
        """写入/覆盖一条工作区记录（**单向适配器**：app.db 权威 → 本存档）。

        架构 §6.4 裁定 ``app.db.workspaces`` 是唯一权威、``workspaces.json`` 迁移后
        降级为只读存档。迁移完成后由迁移任务按 app.db 的记录调用本方法刷新存档，
        方向永远是 app.db → 本文件；本模块不反向写 app.db（``app_store`` 也不依赖
        ``app/core``）。
        """

        with self._lock:
            self._workspaces[workspace.id] = workspace
            self._persist()
            return workspace

    def adopt_record(self, record: Any) -> Workspace:
        """把 ``app.db`` 权威行（``WorkspaceRecord`` / 同名字段 mapping）单向写回存档。

        D29：迁移后所有工作区写入都走 AppStore，写完调用本方法刷新存档，读入口
        （``WorkspaceContextFactory`` → 本对象）因此始终看到权威值。
        """

        return self.adopt(workspace_from_record(record))

    def replace_all(self, workspaces: list[Workspace]) -> list[Workspace]:
        """用给定集合整体替换存档内容（迁移刷新时用；仍保证默认工作区存在）。"""

        with self._lock:
            self._workspaces = {workspace.id: workspace for workspace in workspaces}
            if DEFAULT_WORKSPACE_ID not in self._workspaces:
                self._workspaces[DEFAULT_WORKSPACE_ID] = self._default_workspace()
            self._persist()
            return sorted(self._workspaces.values(), key=_workspace_sort_key)

    def storage_key(self, workspace_id: str) -> str | None:
        """返回工作区目录键；工作区不存在返回 None。"""

        workspace = self.get(workspace_id)
        return None if workspace is None else workspace.storage_key

    def workspace_root(self, data_dir: Path, workspace_id: str) -> Path | None:
        """``data_dir/workspaces/<storage_key>/``（PRD 5.1）；工作区不存在返回 None。"""

        key = self.storage_key(workspace_id)
        return None if key is None else Path(data_dir) / "workspaces" / key

    def ensure_dirs(self, data_dir: Path, workspace_id: str) -> Path:
        """创建并返回本工作区目录树（幂等）；工作区不存在时抛 ``KeyError``。"""

        root = self.workspace_root(data_dir, workspace_id)
        if root is None:
            raise KeyError(f"工作区不存在：{workspace_id}")
        for relative in WORKSPACE_SUBDIRS:
            (root / relative).mkdir(parents=True, exist_ok=True)
        return root

    def bump_knowledge_revision(self, workspace_id: str, *, delta: int = 1) -> int | None:
        """递增本区知识版本（PRD 5.3.6：导入/同步/删除**提交成功后**调用）。

        返回新版本；工作区不存在返回 None。递增后本区的 BM25/目录/术语缓存键
        必须重新计算（PRD 5.3.4）。
        """

        if delta < 0:
            raise ValueError("knowledge_revision 只能递增")
        with self._lock:
            workspace = self._workspaces.get(workspace_id)
            if workspace is None:
                return None
            workspace.knowledge_revision += delta
            workspace.updated_at = _now()
            self._persist()
            return workspace.knowledge_revision

    def rename(self, workspace_id: str, name: str) -> Workspace | None:
        """改名（默认工作区允许改名，但仍不可删除）。"""

        cleaned = name.strip()[:MAX_WORKSPACE_NAME_CHARS]
        if cleaned == "":
            return None
        with self._lock:
            workspace = self._workspaces.get(workspace_id)
            if workspace is None:
                return None
            workspace.name = cleaned
            workspace.updated_at = _now()
            self._persist()
            return workspace

    def delete(self, workspace_id: str) -> bool:
        """删除工作区；默认工作区不可删（返回 False）。"""

        with self._lock:
            workspace = self._workspaces.get(workspace_id)
            if workspace is None or workspace.is_default:
                return False
            del self._workspaces[workspace_id]
            self._persist()
            return True

    def _unique_id(self, base: str) -> str:
        """在既有 id 集合中生成唯一 id（base、base-2、base-3…）。"""

        candidate = base
        index = 2
        while candidate in self._workspaces:
            candidate = f"{base}-{index}"
            index += 1
        return candidate
