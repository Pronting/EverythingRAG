"""工作区存储：data_dir/workspaces.json 持久化（对话分区，对齐 Codex 客户端）。

设计要点：
- 工作区是**轻量目录**：只存 id / name / 时间戳，对话通过 ``Conversation.workspace_id`` 归属；
- 始终保证存在唯一「默认」工作区（``is_default`` 且不可删除、不可改名冲突），
  旧数据（无 workspace_id 的会话）自动落到默认工作区；
- 删除工作区**不删对话**：其中的对话全部归档，可在设置页「归档对话」找回。

名称规则（``default`` / ``general`` / ``new``）沿用本地单用户习惯：中文名直接用，
英文名转小写连字符 slug 并去重；空名回退占位。
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

#: 默认工作区 id（旧会话 / 未指定归属时落这里）。
DEFAULT_WORKSPACE_ID = "default"
DEFAULT_WORKSPACE_NAME = "默认"
#: 名称长度上限（与前端 maxLength 对齐）。
MAX_WORKSPACE_NAME_CHARS = 20


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


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _workspace_sort_key(workspace: Workspace) -> tuple[int, float, str]:
    """侧边栏工作区顺序：置顶在前 → 手工位次 → 创建时间。

    ``order_index`` 为 None（从未拖拽过）的用 ``+inf`` 排在手工项之后，
    仍按创建时间先后；三者类型统一为 (int, float, str)，比较时不会混类型。
    """

    order = workspace.order_index
    return (
        0 if workspace.pinned else 1,
        float(order) if order is not None else float("inf"),
        workspace.created_at,
    )


def _slugify(name: str) -> str:
    """把名称转成可用 id：整串英文/数字转小写连字符，否则回退自动 id。

    混合名称（如「工作区 A」）若只剥出余下的 ASCII 字母会得到 "a" 这种无意义 id，
    因此这里要求整串都是 ASCII 字母数字才做 slug，其余交给 ``workspace[-N]`` 兜底。
    """

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\s_-]*", name.strip()):
        return ""
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")[:40]


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
            )
            self._workspaces[workspace.id] = workspace
            self._persist()
            return workspace

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
