"""V1.2.0 Agent 工作流共享测试夹具（T00a 建立，T00b 扩充，供全部工作流复用）。

提供五类共享夹具：

1. **tmp 工作区目录** —— :func:`make_workspace` / fixture ``agent_workspace``：
   按 PRD 5.1 的目录形状创建 ``documents/`` ``chroma/`` ``artifacts/`` ``cache/``
   与 ``state.db`` / ``image_state.db`` / ``memory.json`` / ``checkpoints.db`` 路径。
2. **内存 app.db** —— fixture ``memory_app_db``（``aiosqlite`` 单连接，外键开启），
   以及需要真实文件的 ``app_db_file``（T00a 起可用）。
3. **业务库夹具（T00b 新增）** —— fixture ``app_store`` / ``app_store_file``：
   直接拿到建好 14 张表的 :class:`app.agent.app_store.AppStore`；``seeded_run``
   与 :func:`seed_run` 一秒钟建好「工作区 + 会话 + 用户消息 + run」。
4. **假 Provider 能力声明** —— :class:`FakeProviderCapabilities`（PRD 9.2 七个字段）
   与 :class:`FakeChatProvider`（记录调用、返回可编排的响应）。
5. **事件收集器** —— :class:`EventCollector` / fixture ``event_collector``：
   收集 ``run_events`` / SSE 事件，提供类型筛选、序号单调性与顺序断言。

用法（pytest 只自动发现 ``conftest.py`` 的夹具，因此这里必须显式导入）::

    from tests.fakes_agent import app_store, event_collector, seeded_run

约束：本模块只依赖标准库、``aiosqlite``、t1 冻结契约 :mod:`app.agent.contracts`
与 T00b 已交付的 :mod:`app.agent.app_store`；**不导入**尚未实现的工作流模块
（``runtime`` / ``graph`` / ``tools`` ...），因此任何工作流都能复用它。
"""

from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import aiosqlite
import pytest

from app.agent.app_store import AppStore
from app.agent.contracts import (
    AgentKind,
    AnswerBasis,
    ApprovalStatus,
    EventType,
    EvidenceSourceType,
    Freshness,
    InputRequestStatus,
    IntentKind,
    PermissionMode,
    ReasoningLevel,
    RunStatus,
    SourceType,
    TaskStatus,
    VerifierVerdict,
    main_thread_id,
    task_thread_id,
)

__all__ = [
    "CONTRACT_SYMBOLS",
    "TEST_EXECUTION_ID",
    "TEST_RUN_ID",
    "TEST_STORAGE_KEY",
    "TEST_TASK_ID",
    "TEST_WORKSPACE_ID",
    "EventCollector",
    "FakeChatProvider",
    "FakeProviderCapabilities",
    "SeededRun",
    "WorkspaceFixture",
    "agent_workspace",
    "app_db_file",
    "app_store",
    "app_store_file",
    "configure_connection",
    "dump_json",
    "event_collector",
    "install_agent_schema",
    "make_workspace",
    "memory_app_db",
    "open_memory_app_db",
    "seed_run",
    "seeded_run",
]

#: 夹具使用的稳定身份（thread_id 规则见 contracts.main_thread_id，PRD 12.1）。
TEST_WORKSPACE_ID: Final[str] = "ws-0001"
TEST_STORAGE_KEY: Final[str] = "ws0001storagkey"
TEST_RUN_ID: Final[str] = "run-0001"
TEST_TASK_ID: Final[str] = "task-0001"
TEST_EXECUTION_ID: Final[str] = "exec-0001"


# =====================================================================
# 1. tmp 工作区目录（PRD 5.1）
# =====================================================================


@dataclass(frozen=True)
class WorkspaceFixture:
    """一个 tmp 工作区的目录与文件路径（不触碰真实 ``~/.everything-rag``）。"""

    workspace_id: str
    storage_key: str
    root: Path

    @property
    def documents_dir(self) -> Path:
        return self.root / "documents"

    @property
    def chroma_dir(self) -> Path:
        return self.root / "chroma"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def jieba_cache_dir(self) -> Path:
        return self.cache_dir / "jieba"

    @property
    def state_db(self) -> Path:
        return self.root / "state.db"

    @property
    def image_state_db(self) -> Path:
        return self.root / "image_state.db"

    @property
    def memory_json(self) -> Path:
        return self.root / "memory.json"

    @property
    def checkpoints_db(self) -> Path:
        return self.root / "checkpoints.db"

    def ensure(self) -> WorkspaceFixture:
        """创建全部目录（文件由各自模块按需创建）。"""

        for path in (
            self.documents_dir,
            self.chroma_dir,
            self.artifacts_dir,
            self.jieba_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def main_thread_id(self) -> str:
        return main_thread_id(TEST_RUN_ID)

    def task_thread_id(self, task_id: str = TEST_TASK_ID) -> str:
        return task_thread_id(TEST_RUN_ID, task_id)


def make_workspace(
    tmp_path: Path,
    *,
    workspace_id: str = TEST_WORKSPACE_ID,
    storage_key: str = TEST_STORAGE_KEY,
) -> WorkspaceFixture:
    """在工作区根目录下创建 `data_dir/workspaces/<storage_key>/` 结构（PRD 5.1）。"""

    root = Path(tmp_path) / "workspaces" / storage_key
    return WorkspaceFixture(
        workspace_id=workspace_id, storage_key=storage_key, root=root
    ).ensure()


@pytest.fixture
def agent_workspace(tmp_path: Path) -> WorkspaceFixture:
    """tmp 工作区目录夹具（PRD 5.1 形状，已创建目录）。"""

    return make_workspace(tmp_path)


# =====================================================================
# 2. 内存 app.db（PRD 13.1）
# =====================================================================

#: ``app_store`` 实现时可能提供的建表入口名（按此顺序探测，避免夹具绑定单一实现）。
_SCHEMA_ENTRYPOINTS: Final[tuple[str, ...]] = (
    "ensure_schema",
    "init_schema",
    "create_schema",
    "initialize",
    "initialize_schema",
)


async def configure_connection(conn: aiosqlite.Connection, *, wal: bool = False) -> None:
    """统一连接配置：外键开启；``wal=True`` 时启用 WAL（``:memory:`` 不支持 WAL）。"""

    conn.row_factory = aiosqlite.Row
    if wal:
        await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")


async def install_agent_schema(conn: aiosqlite.Connection) -> bool:
    """为连接安装 ``app.db`` 表结构；工作流尚未实现时返回 ``False``。

    T00b 实现 ``app.agent.app_store`` 后，本函数自动调用其建表入口，
    其它工作流的夹具无需改动即可用上真实 schema。
    """

    try:
        app_store = importlib.import_module("app.agent.app_store")
    except ImportError:
        return False
    for name in _SCHEMA_ENTRYPOINTS:
        candidate = getattr(app_store, name, None)
        if not callable(candidate):
            continue
        result = candidate(conn)
        if inspect.isawaitable(result):
            await result
        return True
    return False


async def open_memory_app_db(*, install_schema: bool = True) -> aiosqlite.Connection:
    """打开一个外键开启的内存 ``app.db`` 连接（调用方负责 ``await conn.close()``）。"""

    conn = await aiosqlite.connect(":memory:")
    await configure_connection(conn)
    if install_schema:
        await install_agent_schema(conn)
    return conn


@pytest.fixture
async def memory_app_db() -> Iterator[aiosqlite.Connection]:
    """内存 ``app.db`` 夹具：外键开启，``app_store`` 就绪时自动建表。"""

    conn = await open_memory_app_db()
    try:
        yield conn
    finally:
        await conn.close()


@pytest.fixture
async def app_store() -> Iterator[AppStore]:
    """内存 ``AppStore`` 夹具（T00b 起可用）：14 张表就绪，彼此测试互不干扰。

    需要直接跑 SQL 的用例请用上面的 ``memory_app_db``；需要建表 + 业务接口的用例
    用本夹具。
    """

    store = await AppStore.open_memory()
    try:
        yield store
    finally:
        await store.close()


@pytest.fixture
async def app_store_file(tmp_path: Path) -> Iterator[AppStore]:
    """磁盘 ``AppStore`` 夹具（WAL、重启恢复、并发锁场景使用）。"""

    store = await AppStore.open(Path(tmp_path) / "app.db")
    try:
        yield store
    finally:
        await store.close()


@dataclass(frozen=True)
class SeededRun:
    """:func:`seed_run` 建好的最小可用数据图（工作区 → 会话 → 消息 → run）。"""

    workspace_id: str
    storage_key: str
    conversation_id: str
    user_message_id: str
    run_id: str


async def seed_run(
    store: AppStore,
    *,
    workspace_id: str = TEST_WORKSPACE_ID,
    storage_key: str = TEST_STORAGE_KEY,
    conversation_id: str = "conv-0001",
    run_id: str = TEST_RUN_ID,
    question: str = "这是一条测试问题",
    client_request_id: str | None = None,
) -> SeededRun:
    """建好工作区、会话、用户消息与 run，供 RunManager / 工具 / 记忆用例复用。"""

    await store.create_workspace(workspace_id, storage_key, name="测试工作区")
    await store.create_conversation(conversation_id, workspace_id)
    message = await store.append_message(
        conversation_id=conversation_id, role="user", content=question
    )
    run = await store.create_run(
        conversation_id=conversation_id,
        workspace_id=workspace_id,
        client_request_id=client_request_id,
        input_message_id=message.id,
        run_id=run_id,
    )
    return SeededRun(
        workspace_id=workspace_id,
        storage_key=storage_key,
        conversation_id=conversation_id,
        user_message_id=message.id,
        run_id=run.id,
    )


@pytest.fixture
async def seeded_run(app_store: AppStore) -> SeededRun:
    """最小可用数据图夹具（工作区 + 会话 + 用户消息 + run）。"""

    return await seed_run(app_store)



@pytest.fixture
async def app_db_file(tmp_path: Path) -> Iterator[aiosqlite.Connection]:
    """磁盘 ``app.db`` 夹具（WAL 场景、重启恢复场景使用）。"""

    conn = await aiosqlite.connect(str(Path(tmp_path) / "app.db"))
    await configure_connection(conn, wal=True)
    await install_agent_schema(conn)
    try:
        yield conn
    finally:
        await conn.close()


# =====================================================================
# 3. 假 Provider 能力声明（PRD 9.2）
# =====================================================================


@dataclass(frozen=True)
class FakeProviderCapabilities:
    """PRD 9.2 ``ModelCapabilities`` 的七个字段（不依赖尚未实现的模型模块）。

    ``declaration()`` 返回与 PRD 9.2 同名的普通 dict，T03 的
    ``ModelCapabilities`` 可直接由它构造，避免两份字段名漂移。
    """

    supports_tools: bool = True
    supports_thinking_toggle: bool = True
    reasoning_levels: tuple[ReasoningLevel, ...] = (
        ReasoningLevel.OFF,
        ReasoningLevel.LOW,
        ReasoningLevel.HIGH,
        ReasoningLevel.MAX,
    )
    supports_image_input: bool = False
    context_limit: int = 131_072
    max_output_tokens: int = 8_192
    history_protocol: str = "deepseek_tool_calls_v1"

    def declaration(self) -> dict[str, Any]:
        """PRD 9.2 字段名的普通 dict。"""

        return {
            "supports_tools": self.supports_tools,
            "supports_thinking_toggle": self.supports_thinking_toggle,
            "reasoning_levels": [level.value for level in self.reasoning_levels],
            "supports_image_input": self.supports_image_input,
            "context_limit": self.context_limit,
            "max_output_tokens": self.max_output_tokens,
            "history_protocol": self.history_protocol,
        }

    def supports_level(self, level: ReasoningLevel) -> bool:
        return level in self.reasoning_levels


@dataclass
class FakeChatProvider:
    """记录调用的假 Provider：不联网、不调用真实模型。

    验收要求真实模型调用（PRD 16.1），因此本夹具只用于**单元测试**的接线与
    协议形状校验；真实能力验证走实际 Provider。
    """

    capabilities: FakeProviderCapabilities = field(default_factory=FakeProviderCapabilities)
    #: 依次弹出的响应（``None`` 表示用 ``default_response``）。
    responses: list[dict[str, Any]] = field(default_factory=list)
    default_response: dict[str, Any] = field(
        default_factory=lambda: {
            "content": "fake answer",
            "reasoning_content": "",
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "provider_request_id": "fake-req-0001",
        }
    )
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        """记录一次请求并返回完整响应对象（PRD 9.2 字段形状）。"""

        self.calls.append(dict(request))
        if self.responses:
            return dict(self.responses.pop(0))
        return dict(self.default_response)

    def tool_call_sequence(self, *tool_calls: dict[str, Any]) -> None:
        """队列入若干响应：每轮返回一个 ``tool_calls``，最后一轮 ``stop``。"""

        for index, call in enumerate(tool_calls):
            last = index == len(tool_calls) - 1
            self.responses.append(
                {
                    "content": "",
                    "reasoning_content": f"reasoning-{index}",
                    "tool_calls": [call],
                    "finish_reason": "stop" if last else "tool_calls",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    "provider_request_id": f"fake-req-{index:04d}",
                }
            )


# =====================================================================
# 4. 事件收集器（PRD 13.3）
# =====================================================================


def _event_type_of(event: Any) -> str:
    value = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)
    return value.value if isinstance(value, EventType) else str(value)


def _event_seq_of(event: Any) -> int | None:
    value = (
        event.get("event_seq")
        if isinstance(event, dict)
        else getattr(event, "event_seq", None)
    )
    return None if value is None else int(value)


@dataclass
class EventCollector:
    """收集 ``run_events`` / SSE 事件，供断言事件序列与补读语义。

    事件可以是 dict，也可以是带 ``type`` / ``event_seq`` / ``payload`` 属性的
    对象（如 T00b 的 ``EventEnvelope``）；两种形状都能被筛选与断言。
    """

    events: list[Any] = field(default_factory=list)

    def append(self, event: Any) -> Any:
        self.events.append(event)
        return event

    async def emit(self, event: Any) -> None:
        """异步事件汇（可直接传给 RunManager / SSE 生成器）。"""

        self.append(event)

    def extend(self, events: Iterable[Any]) -> None:
        for event in events:
            self.append(event)

    def clear(self) -> None:
        self.events.clear()

    def types(self) -> list[str]:
        return [_event_type_of(event) for event in self.events]

    def by_type(self, event_type: EventType | str) -> list[Any]:
        wanted = event_type.value if isinstance(event_type, EventType) else str(event_type)
        return [event for event in self.events if _event_type_of(event) == wanted]

    def payloads(self, event_type: EventType | str) -> list[Any]:
        return [
            event.get("payload") if isinstance(event, dict) else getattr(event, "payload", None)
            for event in self.by_type(event_type)
        ]

    def seqs(self) -> list[int]:
        return [seq for seq in (_event_seq_of(event) for event in self.events) if seq is not None]

    def last_seq(self) -> int:
        seqs = self.seqs()
        return seqs[-1] if seqs else 0

    def assert_monotonic_seq(self) -> None:
        """``event_seq`` 必须严格单调递增（PRD 12.1：``(run_id, event_seq)`` 唯一）。"""

        seqs = self.seqs()
        assert seqs == sorted(set(seqs)), f"event_seq 不单调或重复: {seqs}"

    def assert_event_order(self, *expected: EventType | str) -> None:
        """断言给出的类型按此顺序出现（可跳过中间事件，PRD 13.1 末段的提交顺序）。"""

        wanted = [
            item.value if isinstance(item, EventType) else str(item) for item in expected
        ]
        remaining = list(wanted)
        for event_type in self.types():
            if remaining and event_type == remaining[0]:
                remaining.pop(0)
        assert not remaining, f"缺少或顺序错误的事件: {remaining}（实际 {self.types()}）"

    def __len__(self) -> int:
        return len(self.events)


@pytest.fixture
def event_collector() -> EventCollector:
    """事件收集器夹具（PRD 13.3）。"""

    return EventCollector()


# =====================================================================
# 契约冒烟：确保夹具引用的枚举可用（防止 t1 契约被改坏后夹具静默失效）
# =====================================================================


def dump_json(payload: Any) -> str:
    """稳定 JSON 序列化（``sort_keys``），用于 ``content_hash`` 类断言。"""

    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


#: 夹具覆盖的契约符号（供契约测试断言「夹具与契约同步」）。
CONTRACT_SYMBOLS: Final[tuple[Any, ...]] = (
    AgentKind,
    AnswerBasis,
    ApprovalStatus,
    EvidenceSourceType,
    EventType,
    Freshness,
    InputRequestStatus,
    IntentKind,
    PermissionMode,
    ReasoningLevel,
    RunStatus,
    SourceType,
    TaskStatus,
    VerifierVerdict,
)
