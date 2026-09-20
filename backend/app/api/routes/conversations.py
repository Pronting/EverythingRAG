"""会话接口：对话历史 + AI 标题 + 归档 + 分支 + 工作区（对齐 Codex 客户端体验）。

- GET/POST /api/conversations：活跃会话列表（可按 workspace 过滤，不含归档/空会话）/ 新建
- POST /api/conversations/{id}/workspace：把会话移动到指定工作区
- GET /api/conversations/archived：已归档会话列表（设置页「归档对话」）
- GET /api/conversations/{id}：完整会话（含消息）
- POST /api/conversations/{id}/messages：追加消息
- POST /api/conversations/{id}/generate-title：用对话模型总结标题
- PUT /api/conversations/{id}/title：手动设标题
- POST /api/conversations/{id}/archive：归档（从侧边栏隐藏）
- POST /api/conversations/{id}/restore：从归档恢复
- POST /api/conversations/{id}/fork：分支出新会话（深拷贝消息，立即打开继续追问）
- DELETE /api/conversations/{id}：删除
- GET/POST /api/workspaces：工作区列表（含会话数）/ 新建
- PUT/DELETE /api/workspaces/{id}：改名 / 删除（删除时其中对话全部归档）
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.agent.contracts import (
    ConversationStatus,
    PermissionMode,
    ReasoningLevel,
    WorkspaceStatus,
)
from app.api.settings_scope import resolve_conversation_settings
from app.api.workspace_bridge import WorkspaceRowMissingError, ensure_workspace_row
from app.core.config import settings
from app.core.conversation_store import ConversationMessage, ConversationStore
from app.core.settings_store import get_settings_store
from app.core.workspace_context import (
    WorkspaceContextFactory,
    WorkspaceNotFoundError,
    committed_storage_version,
)
from app.core.workspace_store import (
    DEFAULT_STORAGE_KEY,
    DEFAULT_WORKSPACE_ID,
    DEFAULT_WORKSPACE_NAME,
    MAX_WORKSPACE_NAME_CHARS,
    WORKSPACE_LAYOUT_VERSION,
    Workspace,
    WorkspaceStore,
)
from app.generation.providers import create_chat_model_from_config

router = APIRouter()


def get_conversation_store() -> ConversationStore:
    from app.api.deps import conversation_store

    return conversation_store


def get_workspace_store() -> WorkspaceStore:
    from app.api.deps import workspace_store

    return workspace_store


# =====================================================================
# D25 / D29：工作区写路径的权威源（过渡期双源 → app.db 唯一权威）
# =====================================================================
#
# 判定唯一依据是 ``data_dir/storage.json`` 的 ``storage_schema_version``（D25）：
#   <2  迁移前：``workspaces.json`` 是**活数据源**，本模块保持既有 JSON 语义；
#   >=2 迁移后：``app.db`` 是唯一权威，工作区写路径改为 ``await AppStore.*``，
#        写完后把权威行**单向**镜像回 JSON 存档（D19/A3：方向恒为 app.db → 存档），
#        这样 D25 的唯一读取入口（``WorkspaceContextFactory`` → ``WorkspaceStore``）
#        在迁移前后返回一致结果，调用方不需要分支判断。


def _app_db_authority() -> Any | None:
    """返回 app.db 写权威源；迁移未提交或 app.db 未注册时返回 None（走 JSON）。

    ``app.db`` 未注册（进程启动早期、测试未注入单例）时**不翻转**写路径：此时降级为
    过渡态写存档，而不是把写入变成失败。生产路径由 lifespan 注册单例（D28/D31）。
    """

    if committed_storage_version(settings.data_dir) < WORKSPACE_LAYOUT_VERSION:
        return None
    from app.api.deps import get_app_store

    return get_app_store()


def _context_factory(store: WorkspaceStore) -> WorkspaceContextFactory:
    """以**本路由注入的存档实例**构造上下文工厂（D25 的唯一读取入口）。

    刻意不复用 ``deps.get_workspace_context_factory()``：那个函数绑定
    ``deps.workspace_store`` 模块单例，而本模块的 ``get_workspace_store`` 可被
    ``dependency_overrides`` 换成另一个指向同一目录的实例；混用两个实例会读到彼此
    陈旧的内存副本（写进去的新工作区在另一个实例里看不到）。

    ``app_db_reader`` 由 deps 提供：迁移后工厂会先用 app.db 权威行按需刷新存档
    （存档是**可再生**的物化视图，被删/损坏/过期都能自愈，读入口因此不依赖它）。
    """

    from app.api.deps import get_app_store, read_app_db_workspaces

    return WorkspaceContextFactory(
        settings.data_dir,
        store,
        app_store=get_app_store(),
        app_db_reader=read_app_db_workspaces,
    )


def _view_of(store: WorkspaceStore, workspace_id: str) -> Any | None:
    """经工厂解析工作区视图；不存在返回 None（绝不静默回落到默认区）。"""

    try:
        return _context_factory(store).view_for(workspace_id)
    except WorkspaceNotFoundError:
        return None


def _normalize_workspace_name(name: str) -> str:
    """名称规范化：去空白、截断到上限、空名回退占位（与存档层同规则）。"""

    return name.strip()[:MAX_WORKSPACE_NAME_CHARS] or "新工作区"


async def _unique_workspace_id(app_store: Any, base: str) -> str:
    """在 app.db 里生成唯一工作区 id（``base`` / ``base-2`` / ``base-3``…）。"""

    candidate = base
    index = 2
    while await app_store.get_workspace(candidate) is not None:
        candidate = f"{base}-{index}"
        index += 1
    return candidate


async def _ensure_default_workspace(app_store: Any, store: WorkspaceStore) -> None:
    """迁移后确保 app.db 里有默认工作区行（PRD 15.2 第 2 步保证；缺失时自愈）。

    默认区若不在 app.db 里，它就无法参与 app.db 权威排序/置顶（会退回存档的初始位次），
    权威源与读入口的侧边栏顺序就会不一致。生产路径上迁移已建好该行，这里是**幂等**兜底。

    自愈实现统一在 :mod:`app.api.workspace_bridge`（run API 与工作区端点共用一处），
    并且会把存档里的**默认权限/强度、知识版本**一并带进新行 —— 否则自愈本身会把用户在
    设置页写下的默认值重置回 PRD 18.1 兜底。
    """

    await ensure_workspace_row(
        app_store,
        store,
        DEFAULT_WORKSPACE_ID,
        allow_archive_missing=True,
        storage_key_fallback=DEFAULT_STORAGE_KEY,
        name_fallback=DEFAULT_WORKSPACE_NAME,
    )


def _mirror_from_app_db(store: WorkspaceStore, record: Any) -> Workspace:
    """把 app.db 权威行**单向**镜像进 JSON 存档（D19/A3）。

    转换规则由 :func:`app.core.workspace_store.workspace_from_record` **唯一定义**
    （``pinned`` / ``order_index`` 自 schema v2 起是 app.db 权威列）。
    """

    return store.adopt_record(record)


class MessageIn(BaseModel):
    role: str
    content: str
    sources: list[dict[str, Any]] | None = None
    answer_basis: str | None = None
    policy_version: str | None = None
    #: 推理模型思维链（本地展示，前端默认折叠）。
    reasoning: str | None = None


class AppendMessagesRequest(BaseModel):
    messages: list[MessageIn]


class SetTitleRequest(BaseModel):
    title: str


class MoveWorkspaceRequest(BaseModel):
    """把会话移动到目标工作区（PRD 5.4 / 13.2）。

    PRD 13.2 末段要求移动接口带**确认标识**与**预期 ``context_epoch``**：

    - ``confirm``：会话携带历史或存在活动任务时必须为 ``true``（前端先展示"会携带历史内容"
      的说明，用户确认后再发）。空会话（无消息、无活动任务）无需确认，保持既有交互。
    - ``expected_context_epoch``：乐观并发校验，不一致返回 409 而不是静默覆盖。
    """

    workspace_id: str = Field(min_length=1)
    confirm: bool = False
    expected_context_epoch: int | None = Field(default=None, ge=1)


class WorkspacePinRequest(BaseModel):
    pinned: bool


class WorkspaceReorderRequest(BaseModel):
    """按给定顺序重写工作区位次（侧边栏拖拽排序落库）。"""

    ids: list[str] = Field(min_length=1, max_length=200)


class TruncateRequest(BaseModel):
    """从第 index 条消息起截断（用户点铅笔编辑重发时调用）。"""

    index: int = Field(ge=0)


class MoveRequest(BaseModel):
    """侧边栏拖拽落点：目标工作区（可选）+ 目标位置（该区第几条）。

    **跨区**拖拽与 ``POST /{id}/workspace`` 是同一条状态机（PRD 5.4），因此同样接受
    ``confirm`` 与 ``expected_context_epoch``（PRD 13.2 末段）。仅``target_index`` 的
    同区排序不涉及归属，不需要确认。
    """

    workspace_id: str | None = Field(default=None, max_length=40)
    target_index: int | None = Field(default=None, ge=0)
    confirm: bool = False
    expected_context_epoch: int | None = Field(default=None, ge=1)


class WorkspaceDefaultsRequest(BaseModel):
    """工作区默认权限与思考强度（PRD 1.3 / 14.1）。

    两者都可选：只改其中一个时另一个保持现值（不传 = 不改，而不是重置为默认）。
    """

    permission_mode: PermissionMode | None = None
    reasoning_level: ReasoningLevel | None = None


class WorkspaceRequest(BaseModel):
    name: str = Field(min_length=1, max_length=MAX_WORKSPACE_NAME_CHARS)
    #: 可选自定义 id（缺省由名称推导）；前端一般只传 name。
    id: str | None = Field(default=None, max_length=40)


@router.get("/api/workspaces")
async def list_workspaces(
    store: WorkspaceStore = Depends(get_workspace_store),  # noqa: B008
    conversations: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> list[dict[str, Any]]:
    """返回全部**有效**工作区（置顶在前 → 手工位次 → 创建时间），带各分区活跃会话数。

    D25 唯一读取入口：身份字段（``id`` / ``name`` / ``status``）取自
    ``WorkspaceContextFactory.view_for(id)``；排序与创建时间等 UI 元数据仍取存档行。
    PRD 5.4 的「删除」是可恢复归档，归档工作区不再出现在侧边栏列表里。
    """
    counts = conversations.count_by_workspace()
    factory = _context_factory(store)
    listed: list[dict[str, Any]] = []
    for workspace in store.list():
        if workspace.status is WorkspaceStatus.ARCHIVED:
            continue
        try:
            view = factory.view_for(workspace.id)
        except WorkspaceNotFoundError:
            continue
        listed.append(_workspace_payload(workspace, counts.get(workspace.id, 0), view=view))
    return listed


@router.put("/api/workspaces/order")
async def reorder_workspaces(
    request: WorkspaceReorderRequest,
    store: WorkspaceStore = Depends(get_workspace_store),  # noqa: B008
    conversations: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> list[dict[str, Any]]:
    """按给定顺序重写工作区位次（侧边栏拖拽排序），返回新的完整顺序。

    必须注册在 ``/api/workspaces/{workspace_id}`` 之前，否则会被当成工作区 id。

    迁移后（D29）：``await AppStore.reorder_workspaces`` 落 app.db 的 ``order_index``
    权威列再单向镜像存档；过渡期保持既有 JSON 语义。**排序规则一字未改**：
    只写 ``order_index``，置顶由 ``pinned`` 单独表达（唯一排序定义见
    ``core.workspace_store.workspace_order_key``）。
    """
    counts = conversations.count_by_workspace()
    app_store = _app_db_authority()
    if app_store is None:
        store.reorder(request.ids)
    else:
        await _ensure_default_workspace(app_store, store)
        for record in await app_store.reorder_workspaces(request.ids):
            _mirror_from_app_db(store, record)
    return [
        _workspace_payload(workspace, counts.get(workspace.id, 0), view=_view_of(store, workspace.id))
        for workspace in store.list()
        if workspace.status is not WorkspaceStatus.ARCHIVED
    ]


@router.post("/api/workspaces", status_code=201)
async def create_workspace(
    request: WorkspaceRequest,
    store: WorkspaceStore = Depends(get_workspace_store),  # noqa: B008
) -> dict[str, Any]:
    """新建工作区（名称为空/重复由存储层兜底，id 冲突自动追加序号）。

    迁移后（D29）：``await AppStore.create_workspace`` 落 app.db，再单向镜像存档；
    过渡期（``storage_schema_version<2``）保持既有 JSON 语义。
    """
    app_store = _app_db_authority()
    if app_store is None:
        workspace = store.create(request.name, request.id)
        return _workspace_payload(workspace, 0)

    base = (request.id or "").strip() or uuid4().hex
    await _ensure_default_workspace(app_store, store)
    workspace_id = await _unique_workspace_id(app_store, base)
    record = await app_store.create_workspace(
        workspace_id,
        uuid4().hex,
        name=_normalize_workspace_name(request.name),
    )
    return _workspace_payload(_mirror_from_app_db(store, record), 0)


@router.put("/api/workspaces/{workspace_id}/pin")
async def pin_workspace(
    workspace_id: str,
    request: WorkspacePinRequest,
    store: WorkspaceStore = Depends(get_workspace_store),  # noqa: B008
) -> dict[str, Any]:
    """置顶 / 取消置顶工作区（别针图标）。

    迁移后（D29）：``await AppStore.update_workspace(pinned=…)`` 落 app.db 权威列再
    单向镜像存档（排序规则不变：置顶在前 → 手工位次 → 创建时间）；过渡期保持既有语义。
    """
    app_store = _app_db_authority()
    if app_store is None:
        workspace = store.set_pinned(workspace_id, request.pinned)
        if workspace is None:
            raise HTTPException(status_code=404, detail="工作区不存在")
        return _workspace_payload(workspace, 0, view=_view_of(store, workspace_id))

    await _ensure_default_workspace(app_store, store)
    if await app_store.get_workspace(workspace_id) is None:
        raise HTTPException(status_code=404, detail="工作区不存在")
    record = await app_store.update_workspace(workspace_id, pinned=request.pinned)
    return _workspace_payload(_mirror_from_app_db(store, record), 0)


@router.put("/api/workspaces/{workspace_id}")
async def rename_workspace(
    workspace_id: str,
    request: WorkspaceRequest,
    store: WorkspaceStore = Depends(get_workspace_store),  # noqa: B008
) -> dict[str, Any]:
    """重命名工作区（默认工作区也可改名，但仍不可删除）。

    迁移后（D29）：``await AppStore.update_workspace(name=…)`` 落 app.db，再单向镜像存档。
    """
    app_store = _app_db_authority()
    if app_store is None:
        workspace = store.rename(workspace_id, request.name)
        if workspace is None:
            raise HTTPException(status_code=404, detail="工作区不存在或名称为空")
        return _workspace_payload(workspace, 0)

    cleaned = request.name.strip()[:MAX_WORKSPACE_NAME_CHARS]
    await _ensure_default_workspace(app_store, store)
    if not cleaned or await app_store.get_workspace(workspace_id) is None:
        raise HTTPException(status_code=404, detail="工作区不存在或名称为空")
    record = await app_store.update_workspace(workspace_id, name=cleaned)
    return _workspace_payload(_mirror_from_app_db(store, record), 0)


@router.put("/api/workspaces/{workspace_id}/defaults")
async def update_workspace_defaults(
    workspace_id: str,
    request: WorkspaceDefaultsRequest,
    store: WorkspaceStore = Depends(get_workspace_store),  # noqa: B008
) -> dict[str, Any]:
    """写工作区默认权限模式 / 思考强度（PRD 1.3「工作区默认值」+ 14.1「设置页」）。

    权限与思考强度占位层次（PRD 1.3）：请求显式值 > 会话保存值 > **工作区默认值**。
    本端点负责第三层的写；取值为 contracts 的 ``PermissionMode`` / ``ReasoningLevel``，
    非法值由请求模型直接 422（不在运行期兜底，PRD 12.5）。

    写路径与 D25/D29 一致：迁移后（``storage_schema_version >= 2``）由 ``app.db`` 权威，
    ``await AppStore.update_workspace`` 落库后单向镜像回 JSON 存档；迁移前保持既有
    JSON 语义（存档本身是权威源）。两者都返回同一投影，调用方无需分支。
    """
    if request.permission_mode is None and request.reasoning_level is None:
        raise HTTPException(status_code=400, detail="至少给出一个默认值")

    app_store = _app_db_authority()
    if app_store is None:
        # 迁移前 / app.db 未注册：存档是活数据源，直接更新它（adopt 会落盘）
        workspace = store.get(workspace_id)
        if workspace is None:
            raise HTTPException(status_code=404, detail="工作区不存在")
        updated = workspace.model_copy(
            update={
                "default_permission_mode": request.permission_mode
                or workspace.default_permission_mode,
                "default_reasoning_level": request.reasoning_level
                or workspace.default_reasoning_level,
            }
        )
        mirrored = store.adopt(updated)
        return _workspace_payload(mirrored, 0, view=_view_of(store, workspace_id))

    await _ensure_default_workspace(app_store, store)
    if await app_store.get_workspace(workspace_id) is None:
        raise HTTPException(status_code=404, detail="工作区不存在")
    record = await app_store.update_workspace(
        workspace_id,
        default_permission_mode=request.permission_mode,
        default_reasoning_level=request.reasoning_level,
    )
    return _workspace_payload(_mirror_from_app_db(store, record), 0)


@router.delete("/api/workspaces/{workspace_id}")
async def delete_workspace(
    workspace_id: str,
    store: WorkspaceStore = Depends(get_workspace_store),  # noqa: B008
    conversations: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> dict[str, Any]:
    """删除工作区：其中对话全部归档（可在设置页「归档对话」找回）；默认工作区不可删。

    迁移后（D29）按 PRD 5.4 走**可恢复归档**：``AppStore.update_workspace(status=archived)``
    保留工作区行归属，其 app.db 会话一并归档，再由存档单向镜像；过渡期保持既有语义。
    """
    app_store = _app_db_authority()
    if app_store is None:
        if store.get(workspace_id) is None:
            raise HTTPException(status_code=404, detail="工作区不存在")
        if not store.delete(workspace_id):
            raise HTTPException(status_code=400, detail="默认工作区不可删除")
        archived = conversations.archive_workspace(workspace_id)
        return {"archived": archived}

    record = await app_store.get_workspace(workspace_id)
    if workspace_id == DEFAULT_WORKSPACE_ID:
        raise HTTPException(status_code=400, detail="默认工作区不可删除")
    if record is None:
        raise HTTPException(status_code=404, detail="工作区不存在")

    await _ensure_default_workspace(app_store, store)
    archived_record = await app_store.update_workspace(
        workspace_id, status=WorkspaceStatus.ARCHIVED
    )
    _mirror_from_app_db(store, archived_record)
    # 归档本区 app.db 会话（保留归属、可恢复）＋ 侧边栏存档里的会话（用户可见的归档列表）
    for conversation in await app_store.list_conversations(workspace_id):
        if conversation.status is ConversationStatus.ACTIVE:
            await app_store.update_conversation(
                conversation.id, status=ConversationStatus.ARCHIVED
            )
    archived = conversations.archive_workspace(workspace_id)
    return {"archived": archived}


@router.get("/api/conversations")
async def list_conversations(
    q: str = Query(default="", max_length=200),
    workspace: str | None = Query(default=None, max_length=40),
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> list[dict[str, Any]]:
    """返回活跃会话摘要列表（按最近更新倒序，不含归档与空会话）。

    给定 ``workspace`` 时只返回该工作区的会话（侧边栏分区切换）。
    """
    if q.strip():
        return store.search(q, workspace)
    return store.list_summaries(workspace)


@router.get("/api/conversations/archived")
async def list_archived_conversations(
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> list[dict[str, Any]]:
    """返回已归档会话摘要列表（按归档时间倒序）。

    必须注册在 `/api/conversations/{conversation_id}` 之前，否则会被当成会话 id。
    """
    return store.list_archived_summaries()


@router.post("/api/conversations", status_code=201)
async def create_conversation(
    workspace: str = Query(default=DEFAULT_WORKSPACE_ID, max_length=40),
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
    workspace_store: WorkspaceStore = Depends(get_workspace_store),  # noqa: B008
) -> dict[str, Any]:
    """在指定工作区（缺省默认工作区）新建空会话，返回其摘要。

    **工作区默认值在此生效**（PRD 1.3 / 14.1 / 18.1）：迁移后（app.db 权威）新会话的
    权限与思考强度按 ``工作区默认值 → PRD 18.1 兜底`` 解析，并随 app.db 行一起落库 ——
    这正是「工作区设置页提供默认值」的落点（此前只写默认值却不被应用 ⇒ 设置形同装饰）。

    过渡期（``storage_schema_version < 2`` / app.db 未注册）保持既有 JSON 语义：会话
    只进存档；工作区默认值不在此处落库（存档行没有这两个字段，且 app.db 当时非权威）。

    双写顺序：先建存档行（拿到 id，侧边栏立即可见），再用**同一个 id** 写 app.db 行
    （带解析后的默认值）。id 一致是硬要求 —— run API 按 app.db 的会话 id 工作。
    """
    settings = resolve_conversation_settings(_view_of(workspace_store, workspace))
    app_store = _app_db_authority()

    if app_store is None:
        conv = store.create(workspace)
        return _summary(conv)

    await _ensure_default_workspace(app_store, workspace_store)
    try:
        await ensure_workspace_row(app_store, workspace_store, workspace)
    except WorkspaceRowMissingError as exc:
        raise HTTPException(status_code=404, detail="工作区不存在") from exc
    conversation_id = uuid4().hex
    conv = store.create(workspace, conversation_id=conversation_id)
    await app_store.create_conversation(
        conversation_id,
        workspace,
        permission_mode=settings.permission_mode,
        reasoning_level=settings.reasoning_level,
    )
    return _summary(conv)


@router.get("/api/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> dict[str, Any]:
    """返回完整会话（含消息列表）。"""
    conv = store.get(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return conv.model_dump()


@router.post("/api/conversations/{conversation_id}/messages")
async def append_messages(
    conversation_id: str,
    request: AppendMessagesRequest,
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> dict[str, Any]:
    """追加一轮消息（user + assistant），返回更新后的会话摘要。"""
    messages = [
        ConversationMessage(
            role=m.role,
            content=m.content,
            sources=m.sources,
            answer_basis=m.answer_basis,
            policy_version=m.policy_version,
            reasoning=m.reasoning,
        )
        for m in request.messages
    ]
    conv = store.append_messages(conversation_id, messages)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _summary(conv)


@router.put("/api/conversations/{conversation_id}/title")
async def set_title(
    conversation_id: str,
    request: SetTitleRequest,
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> dict[str, Any]:
    """手动设置标题。"""
    conv = store.set_title(conversation_id, request.title)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _summary(conv)


@router.post("/api/conversations/{conversation_id}/truncate")
async def truncate_conversation(
    conversation_id: str,
    request: TruncateRequest,
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> dict[str, Any]:
    """从第 index 条消息起截断会话（前端「编辑重发」：该条及其之后的轮次作废）。"""
    conv = store.truncate_from(conversation_id, request.index)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"message_count": len(conv.messages)}


@router.post("/api/conversations/{conversation_id}/move")
async def move_conversation_in_sidebar(
    conversation_id: str,
    request: MoveRequest,
    http_request: Request,
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
    workspaces: WorkspaceStore = Depends(get_workspace_store),  # noqa: B008
) -> dict[str, Any]:
    """侧边栏拖拽落点：**同区排序** 或 **跨区移动**（PRD 5.4 / 13.2）。

    ⚠️ **跨区必须与 ``POST /api/conversations/{id}/workspace`` 走同一条状态机**
    （``_move_via_state_machine``）。早先这里直接 ``store.move_in_sidebar`` 改存档归属：
    不停止 run、不递增 ``context_epoch``、不 supersede 批准/澄清、不写 app.db ⇒
    **侧边栏与 app.db 权威行分叉**，A27 在**最常用的拖拽路径**上被整条绕过（t35 修正）。
    仅同区 ``target_index`` 排序不含归属变更，因此不触发状态机（也不会碰权威行）。
    """

    current = store.get(conversation_id)
    target = request.workspace_id
    # 归属的**权威来源**必须与状态机同一口径：app.db 行优先，未迁移（未注册）时退回存档
    # （D19 / D25）。若这里读存档而状态机读 app.db，分叉状态下的「同区判断」会不一致：
    # 存档已被旧旁路改成目标区时，跨区移动会被误判成同区排序而**整条跳过状态机**。
    source = await _authoritative_workspace_id(current, conversation_id)
    if source is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    if target is None or target == source:
        # 同区：只改侧边栏顺序（PRD 5.4 的「移动」语义不成立）。
        conv = store.move_in_sidebar(
            conversation_id, workspace_id=source, target_index=request.target_index
        )
        if conv is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return _summary(conv)

    payload = await _move_via_state_machine(
        conversation_id=conversation_id,
        request=http_request,
        target_workspace_id=str(target),
        confirm=request.confirm,
        expected_context_epoch=request.expected_context_epoch,
        store=store,
        workspaces=workspaces,
    )
    if request.target_index is not None:
        # 落点位置在同区生效（归属已由状态机在 app.db 权威 + 存档镜像改完）。
        placed = store.move_in_sidebar(
            conversation_id, workspace_id=str(target), target_index=request.target_index
        )
        if placed is not None:
            summary = _summary(placed)
            summary.update({key: value for key, value in payload.items() if key not in summary})
            return summary
    return payload


async def _authoritative_workspace_id(
    sidebar: Any, conversation_id: str
) -> str | None:
    """会话当前归属：**app.db 权威行优先**，未迁移（未注册 app.db）时退回存档（D19 / D25）。

    与 ``conversation_move.move_conversation`` 的 ``source_workspace_id`` 同一口径 ——
    两个入口对「是否同区」的判断必须一致，否则分叉状态下会跳过状态机（t35）。
    """

    app_store = _registered_app_store()
    if app_store is not None:
        record = await app_store.get_conversation(conversation_id)
        if record is not None:
            return str(record.workspace_id)
    if sidebar is not None:
        return str(sidebar.workspace_id)
    return None


async def _move_via_state_machine(
    *,
    conversation_id: str,
    request: Request,
    target_workspace_id: str,
    confirm: bool,
    expected_context_epoch: int | None,
    store: ConversationStore,
    workspaces: WorkspaceStore,
) -> dict[str, Any]:
    """**跨区移动的唯一实现路径**：``/move`` 与 ``/workspace`` 共用（不得再有第二条旁路）。

    顺序（PRD 5.4）：停止主任务及子任务 → 等待活动请求退出 → 改归属（app.db 权威，
    ``context_epoch`` +1）→ 侧边栏镜像 → 目标区记忆快照。原 run 的工作区归属不变，
    仅历史可读且不可恢复。
    """

    from app.api.conversation_move import (
        ConversationMoveConflict,
        ConversationMoveError,
        ConversationMoveNotFound,
        MoveConfirmationRequired,
        move_conversation,
    )

    try:
        outcome = await move_conversation(
            conversation_id=conversation_id,
            target_workspace_id=target_workspace_id,
            store=store,
            workspaces=workspaces,
            manager=_optional_run_manager(request),
            app_store=_registered_app_store(),
            confirm=confirm,
            expected_context_epoch=expected_context_epoch,
            target_memory=_target_memory_factory(workspaces),
        )
    except ConversationMoveNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (MoveConfirmationRequired, ConversationMoveConflict) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ConversationMoveError as exc:  # pragma: no cover - 兜底 409 家族
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    conversation = outcome.conversation
    if conversation is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    payload = _summary(conversation)
    payload.update(
        {
            "from_workspace_id": outcome.from_workspace_id,
            "workspace_id": outcome.to_workspace_id,
            "context_epoch": outcome.context_epoch,
            "previous_context_epoch": outcome.previous_context_epoch,
            "stopped_run_ids": list(outcome.stopped_run_ids),
            "superseded_approvals": outcome.superseded_approvals,
            "superseded_inputs": outcome.superseded_inputs,
            "memory_snapshot_rebound": outcome.memory_snapshot_rebound,
            "permission_rebound": outcome.permission_rebound,
            "steps": list(outcome.steps),
        }
    )
    return payload


@router.post("/api/conversations/{conversation_id}/workspace")
async def move_conversation_workspace(
    conversation_id: str,
    request: MoveWorkspaceRequest,
    http_request: Request,
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
    workspaces: WorkspaceStore = Depends(get_workspace_store),  # noqa: B008
) -> dict[str, Any]:
    """把会话移动到另一个工作区（PRD 5.4 / 13.2；A27 / A28）。"""

    return await _move_via_state_machine(
        conversation_id=conversation_id,
        request=http_request,
        target_workspace_id=request.workspace_id,
        confirm=request.confirm,
        expected_context_epoch=request.expected_context_epoch,
        store=store,
        workspaces=workspaces,
    )


def _optional_run_manager(http_request: Request) -> Any:
    """取 RunManager；未注册（裸路由测试/未启动 lifespan）返回 None（不做假装停止）。"""

    manager = getattr(http_request.app.state, "run_manager", None)
    if manager is not None:
        return manager
    from app.agent.runtime import get_run_manager

    return get_run_manager()


def _registered_app_store() -> Any:
    """当前 app.db 访问层单例；未注册返回 None（D25 过渡态：JSON 仍是活源）。"""

    from app.api.deps import get_app_store

    return get_app_store()


def _target_memory_factory(store: WorkspaceStore) -> Any:
    """目标区 ``memory.json`` 句柄工厂（按 ``workspace_id`` 解析，PRD 5.4）。

    刻意用**本路由注入的**存档实例构造上下文工厂（与 ``_context_factory`` 同一理由）：
    ``deps.workspace_store`` 是模块单例，可能已被 ``dependency_overrides`` 换成另一个
    指向同目录的实例；混用会解析不到刚创建的工作区。
    """

    def factory(workspace_id: str) -> Any:
        from app.api import deps as deps_mod

        context = deps_mod.get_workspace_context_factory(store).resolve(workspace_id)
        return deps_mod.workspace_memory(context)

    return factory


@router.post("/api/conversations/{conversation_id}/archive")
async def archive_conversation(
    conversation_id: str,
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> dict[str, Any]:
    """归档会话：从侧边栏隐藏，可在设置页「归档对话」恢复。"""
    conv = store.set_archived(conversation_id, True)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _summary(conv)


@router.post("/api/conversations/{conversation_id}/restore")
async def restore_conversation(
    conversation_id: str,
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> dict[str, Any]:
    """恢复已归档会话，重新回到侧边栏。"""
    conv = store.set_archived(conversation_id, False)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _summary(conv)


@router.post("/api/conversations/{conversation_id}/fork", status_code=201)
async def fork_conversation(
    conversation_id: str,
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> dict[str, Any]:
    """分支会话：深拷贝源会话消息为独立新会话（侧边栏折叠在源会话下方）。"""
    branch = store.fork(conversation_id)
    if branch is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _summary(branch)


@router.post("/api/conversations/{conversation_id}/generate-title")
async def generate_title(
    conversation_id: str,
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> dict[str, str]:
    """用对话模型从前 2 条用户消息总结标题；模型不可用时回退首个用户消息截断。"""
    conv = store.get(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    title = await _ai_title(conv)
    store.set_title(conversation_id, title)
    return {"title": title}


@router.delete("/api/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    store: ConversationStore = Depends(get_conversation_store),  # noqa: B008
) -> None:
    """删除会话（其分支提升为顶层，不级联删除）。"""
    if not store.delete(conversation_id):
        raise HTTPException(status_code=404, detail="会话不存在")


async def _ai_title(conv: Any) -> str:
    """生成标题：优先对话模型总结，回退首个用户消息前 30 字。"""
    user_texts = [m.content for m in conv.messages if m.role == "user"][:2]
    if not user_texts:
        return "新对话"
    try:
        chat = create_chat_model_from_config(get_settings_store().load().chat)
        prompt = (
            "请用不超过 15 字的短语概括这段对话的主题，只返回标题本身，"
            "不要加引号、标点或解释。\n用户：" + "\n用户：".join(user_texts)
        )
        tokens: list[str] = []
        async for chunk in chat.stream_chat([{"role": "user", "content": prompt}]):
            if chunk.kind == "content":
                tokens.append(chunk.text)
        title = "".join(tokens).strip()
        if title:
            return title[:60]
    except Exception:  # noqa: BLE001, S110 -- 标题生成失败回退，不影响主流程
        pass
    return user_texts[0][:30]


def _summary(conv: Any) -> dict[str, Any]:
    return {
        "id": conv.id,
        "title": conv.title,
        "created_at": conv.created_at,
        "updated_at": conv.updated_at,
        "message_count": len(conv.messages),
        "archived": conv.archived,
        "archived_at": conv.archived_at,
        "parent_id": conv.parent_id,
        "forked_from_title": conv.forked_from_title,
        "workspace_id": conv.workspace_id,
        "sort_index": conv.sort_index,
    }


def _workspace_payload(
    workspace: Any, conversation_count: int, *, view: Any | None = None
) -> dict[str, Any]:
    """工作区响应投影。

    身份字段（``id`` / ``name`` / ``status``）优先取 D25 唯一读取入口
    （``WorkspaceContextFactory`` 的视图）；排序与时间戳等 UI 元数据取存档行。
    """

    return {
        "id": workspace.id if view is None else view.workspace_id,
        "name": workspace.name if view is None else view.name,
        "created_at": workspace.created_at,
        "updated_at": workspace.updated_at,
        "is_default": workspace.is_default,
        "pinned": workspace.pinned,
        "order_index": workspace.order_index,
        "status": str(workspace.status if view is None else view.status),
        "conversation_count": conversation_count,
    }
