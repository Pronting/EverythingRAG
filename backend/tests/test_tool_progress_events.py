"""权限门发布过程事件（PRD 14.2 的工具条目）。

过程时间线里「检索 · 正在检索知识库 / 检索：命中 2 条资料」这样的环节行，唯一来源是运行层
发布的 ``tool.started`` / ``tool.completed``。此前**只有 rerank 这一个工具发过**这两类事件
（``runtime._rerank_progress_sink``），其余工具一次都不发，时间线于是只剩粗糙的阶段叙述。

本文件守住 ``ApprovalGate.execute`` 这个**唯一收口点**上的四条不变量：

1. 一次工具调用 = 一对开始/结束事件，且两个事件的 ``operation_id`` 相同（界面据此原地收口）；
2. 同一工具的**多次**调用拿到**不同** ``operation_id``（否则界面会把两次调用并成一行）；
3. 失败如实收口成 ``status="failed"`` + 稳定 ``error_code``，并**原样上抛**异常（不吞错误）；
4. ``args_summary`` 与批准卡片共用脱敏规则（密钥/认证头/绝对路径不进过程区域）。

外加一个纯函数守卫：``_tool_result_counts`` 只认真实存在的计数，认不出就是 0（不猜数字）。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.agent import runtime as agent_runtime
from app.agent.contracts import EventType
from app.agent.tools import ToolError, ToolErrorCode

# =====================================================================
# 替身（零出网、零数据库：只观察发布出来的事件）
# =====================================================================


class _FakeParams:
    """被 ``parse_arguments`` 返回的参数对象（只需 ``model_dump``）。"""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        del mode
        return dict(self._payload)


class _FakeSpec:
    def __init__(self, name: str, *, requires_approval: bool = False) -> None:
        self.name = name
        self.requires_approval = requires_approval


class _FakeRegistry:
    """最小工具注册表：记录每次 execute 的真实入参，按脚本返回结果或抛错。"""

    def __init__(self, result: Any = None, error: BaseException | None = None) -> None:
        self._result = result
        self._error = error
        self.executed: list[dict[str, Any]] = []

    def get(self, name: str) -> _FakeSpec:
        return _FakeSpec(name)

    def parse_arguments(self, spec: _FakeSpec, arguments: Any) -> _FakeParams:
        del spec
        if isinstance(arguments, str):
            return _FakeParams({"query": arguments})
        return _FakeParams(dict(arguments))

    async def execute(self, name: str, safe: Any, context: Any, **kwargs: Any) -> Any:
        self.executed.append({"name": name, "safe": dict(safe), "context": context, "kwargs": kwargs})
        if self._error is not None:
            raise self._error
        return self._result


class _FakeManager:
    """只实现发布口：``publish_event`` 是运行层与事件流的唯一接缝。"""

    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []

    async def publish_event(
        self,
        run_id: str,
        event_type: Any,
        payload: Any,
        *,
        task_id: str | None = None,
        **_ignored: Any,
    ) -> None:
        self.published.append(
            {
                "run_id": run_id,
                "type": str(getattr(event_type, "value", event_type)),
                "payload": dict(payload),
                "task_id": task_id,
            }
        )

    def types(self) -> list[str]:
        return [item["type"] for item in self.published]


def _gate(registry: _FakeRegistry, manager: _FakeManager) -> Any:
    return agent_runtime.ApprovalGate(
        manager=manager,
        registry=registry,
        run=SimpleNamespace(id="run-1"),
        state={"question": "测试问题"},  # type: ignore[arg-type]
        question_scope="测试问题",
    )


def _context() -> Any:
    return SimpleNamespace(run_id="run-1", task_id="t1")


# =====================================================================
# 1. 成功路径：一对事件、同一个 operation_id、真实计数
# =====================================================================


async def test_tool_call_publishes_started_then_completed_with_real_counts() -> None:
    manager = _FakeManager()
    registry = _FakeRegistry(
        result=SimpleNamespace(data={"chunks": [{"a": 1}, {"a": 2}], "count": 2, "sources": [{"s": 1}]})
    )
    gate = _gate(registry, manager)

    returned = await gate.execute("search_knowledge", {"query": "限流", "top_k": 5}, _context())

    assert returned is not None
    assert manager.types() == [EventType.TOOL_STARTED.value, EventType.TOOL_COMPLETED.value]

    started, completed = manager.published[0]["payload"], manager.published[1]["payload"]
    assert started["tool_name"] == "search_knowledge"
    assert started["operation_id"] == completed["operation_id"] != ""
    assert started["task_id"] == "t1"
    # 参数摘要与批准卡片同一套规则：只出现真实参数，不出现密钥/认证头/绝对路径。
    assert "query=限流" in started["args_summary"]
    assert "top_k=5" in started["args_summary"]

    assert completed["status"] == "succeeded"
    # 计数是真实值（2 条结果、1 个来源），而不是缺省的 0。
    assert completed["result_count"] == 2
    assert completed["source_count"] == 1
    assert completed["error_code"] == ""


async def test_args_summary_never_carries_hidden_keys() -> None:
    """脱敏是过程区域的硬约束（PRD 12.5）：密钥与认证头不得进入 args_summary。"""

    manager = _FakeManager()
    registry = _FakeRegistry(result=SimpleNamespace(data={"count": 1}))
    gate = _gate(registry, manager)

    await gate.execute(
        "web_search",
        {"query": "张三 近况", "api_key": "sk-secret", "headers": {"Authorization": "Bearer x"}},
        _context(),
    )

    summary = manager.published[0]["payload"]["args_summary"]
    assert "sk-secret" not in summary
    assert "Authorization" not in summary
    assert "query=张三 近况" in summary


# =====================================================================
# 2. 同一工具的多次调用必须各占一行
# =====================================================================


async def test_repeated_calls_get_distinct_operation_ids() -> None:
    """界面按 ``operation_id`` 让每次调用独占一个环节行（``processState`` 的条目 id）。"""

    manager = _FakeManager()
    registry = _FakeRegistry(result=SimpleNamespace(data={"count": 1}))
    gate = _gate(registry, manager)

    await gate.execute("read_document", {"source_ref": "a.md"}, _context())
    await gate.execute("read_document", {"source_ref": "b.md"}, _context())

    ids = [
        item["payload"]["operation_id"]
        for item in manager.published
        if item["type"] == EventType.TOOL_STARTED.value
    ]
    assert len(ids) == 2
    assert len(set(ids)) == 2, f"两次调用拿到了同一个 operation_id：{ids}"


# =====================================================================
# 3. 失败如实收口并原样上抛
# =====================================================================


async def test_failed_tool_publishes_failure_and_reraises() -> None:
    manager = _FakeManager()
    registry = _FakeRegistry(error=ToolError("服务不可用", code=ToolErrorCode.SERVICE_ERROR))
    gate = _gate(registry, manager)

    with pytest.raises(ToolError):
        await gate.execute("search_knowledge", {"query": "限流"}, _context())

    assert manager.types() == [EventType.TOOL_STARTED.value, EventType.TOOL_COMPLETED.value]
    completed = manager.published[1]["payload"]
    assert completed["status"] == "failed"
    assert completed["error_code"] == str(ToolErrorCode.SERVICE_ERROR)
    # 失败不是「没找到资料」：结果计数不得被当成成功计数展示。
    assert manager.published[1]["payload"]["result_count"] == 0


async def test_missing_publish_seam_does_not_break_tool_execution() -> None:
    """发布口缺失（单测替身）时工具照常执行，不因过程事件缺席而改变行为。"""

    class _NoPublisher:
        pass

    registry = _FakeRegistry(result=SimpleNamespace(data={"count": 3}))
    gate = agent_runtime.ApprovalGate(
        manager=_NoPublisher(),
        registry=registry,
        run=SimpleNamespace(id="run-1"),
        state={},  # type: ignore[arg-type]
    )

    returned = await gate.execute("lookup_catalog", {"query": "限流"}, _context())
    assert returned is not None
    assert len(registry.executed) == 1


# =====================================================================
# 4. 计数只认真实字段（纯函数守卫）
# =====================================================================


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"count": 3}, (3, 0)),
        ({"count": 0}, (0, 0)),
        ({"results": [{}, {}]}, (2, 0)),
        ({"pages": [{}], "count": 1}, (1, 0)),
        ({"chunks": [], "count": 0, "sources": [{"a": 1}, {"b": 2}]}, (0, 2)),
        # 认不出条数的工具（概览/确认/图片核对）：报 0，由界面收口成「已完成」。
        ({"categories": {"documents": 2}}, (0, 0)),
        ({}, (0, 0)),
    ],
)
def test_tool_result_counts_reads_only_real_numbers(
    data: dict[str, Any], expected: tuple[int, int]
) -> None:
    assert agent_runtime._tool_result_counts(SimpleNamespace(data=data)) == expected


def test_tool_result_counts_tolerates_non_mapping_data() -> None:
    assert agent_runtime._tool_result_counts(SimpleNamespace(data=None)) == (0, 0)
    assert agent_runtime._tool_result_counts(object()) == (0, 0)
