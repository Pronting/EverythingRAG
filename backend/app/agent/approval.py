"""权限判断、批准暂停与恢复、人名脱敏（PRD 10）。Owner: T09。

职责：
- 三种权限模式（``ask`` / ``auto`` / ``full``，取值见 :mod:`app.agent.contracts`）
  下的决策，联网动作执行前重新检查（PRD 10.1）。
- ``ApprovalRequest`` 的状态流转本体在 :mod:`app.agent.app_store`（单事务 + 唯一
  约束），本模块只做**决定语义**与卡片构造：``pending → approved/denied/superseded``，
  重复提交相同决定幂等返回、冲突决定 409（PRD 10.2）。
- 独立批准节点调用 LangGraph ``interrupt()``；恢复使用原图身份与
  ``Command(resume=...)``，因此模型调用与外部工具操作必须在各自独立节点（PRD 10.3）。
- 人名脱敏：非公开姓名替换为稳定称谓，记录原始查询与实际联网查询的对应关系
  （PRD 10.4）。用户原文 / 聊天记录 / 知识索引**保持原内容** —— 脱敏只作用于
  **出网参数**，不改本地数据。

关键取舍（写进架构文档 §6 D35）：

1. 「本轮同类操作」的授权范围严格限定 ``run_id + 工具类别 + 问题范围 + 规则版本``
   四项（:func:`scope_matches`）。任一项不同即视为未授权：范围放宽会让用户在 A 问题
   上点的「允许」被 B 问题的联网调用复用。
2. 脱敏只用**确定性**替换：同一姓名在本实例内始终映射到同一称谓，且逐字匹配而非
   正则模糊匹配（PRD 10.4 要求可复核）。
3. URL 里出现非公开姓名时**不伪造**可用地址：能安全替换就保留地址，否则改用脱敏后的
   搜索目标并把原因写进 :class:`NameRedactionResult`（PRD 10.4 末段）。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from pydantic import Field

from app.agent.app_store import ApprovalConflictError
from app.agent.contracts import (
    ApprovalDecision,
    ApprovalDecisionKind,
    ApprovalDisplay,
    ApprovalRequest,
    ApprovalScope,
    ApprovalStatus,
    ContractModel,
    PermissionMode,
    utc_now_iso,
)
from app.agent.models import AgentState, PendingApproval

__all__ = [
    "APPROVAL_RULE_VERSION",
    "PERMISSION_ALLOW_NOW",
    "PERMISSION_DENY",
    "PERMISSION_NEED_APPROVAL",
    "REDACTION_NOTICE",
    "TOOL_CATEGORIES",
    "ApprovalConflictError",
    "ApprovalDeniedError",
    "ApprovalInterruptUnavailable",
    "ApprovalRequired",
    "NameRedactionResult",
    "NameRedactor",
    "apply_scope_grant",
    "approval_node",
    "approval_payload",
    "build_approval_request",
    "build_approval_scope",
    "build_display",
    "decide_permission",
    "granted_categories",
    "redact_names",
    "resolve_approval",
    "scope_matches",
    "suspend",
    "tool_category",
]

#: 批准规则版本（PRD 10.2 授权范围四项之一）。规则变化后旧的「本轮同类」授权立即
#: 失效（:func:`scope_matches` 直接比较字符串）。
APPROVAL_RULE_VERSION: Final[str] = "v1.2.0-1"

#: 卡片上的脱敏提示（PRD 10.4）。
REDACTION_NOTICE: Final[str] = "联网参数已脱敏：非公开姓名替换为稳定称谓后才发出请求。"

#: 工具类别（PRD 10.2 授权范围第二项）：同类别内的操作才共享「本轮同类」授权。
TOOL_CATEGORIES: Final[dict[str, str]] = {
    "web_search": "web",
    "read_web_page": "web",
    "search_knowledge": "knowledge",
    "read_document": "knowledge",
    "lookup_catalog": "knowledge",
    "get_workspace_overview": "knowledge",
    "search_history": "history",
    "read_history": "history",
    "verify_image": "vision",
    "ask_user": "interaction",
}

#: 判定取值：直接执行 / 需要批准 / 拒绝。
PERMISSION_ALLOW_NOW: Final[str] = "allow_now"
PERMISSION_NEED_APPROVAL: Final[str] = "need_approval"
PERMISSION_DENY: Final[str] = "deny"


class ApprovalRequired(Exception):
    """某个需要批准的工具操作被拦下（研究节点据此挂起并交批准节点处理）。"""

    def __init__(self, pending: PendingApproval) -> None:
        super().__init__(f"工具 {pending.tool_name} 需要用户批准")
        self.pending = pending


class ApprovalDeniedError(Exception):
    """用户拒绝了该操作（研究节点不再重试同一 ``args_hash``）。"""

    def __init__(self, approval_id: str) -> None:
        super().__init__(f"批准 {approval_id} 已被拒绝")
        self.approval_id = approval_id


class ApprovalInterruptUnavailable(RuntimeError):
    """当前驱动没有 LangGraph ``interrupt()``（离线驱动据此按「挂起」处理）。"""


def tool_category(tool_name: str) -> str:
    """工具 → 授权类别；未知工具归入自身名字（保守：不与别的工具共享授权）。"""

    return TOOL_CATEGORIES.get(str(tool_name), str(tool_name))


def decide_permission(
    *,
    mode: PermissionMode | str,
    requires_approval: bool,
    granted_categories: Sequence[str] = (),
    category: str = "",
    denied: bool = False,
) -> str:
    """按 PRD 10.1 判定本次工具操作如何处理。

    - ``full``：已配置工具直接执行；
    - ``auto``：审核节点确认后自动批准（运行层用「符合当前问题」这一条判定，等价放行）；
    - ``ask``：需要批准的工具在**每次执行前**停顿，只有命中本轮授权范围时免停顿。

    ``denied`` 表示该操作已被用户明确拒绝：任何模式下都不再执行。
    """

    if denied:
        return PERMISSION_DENY
    if not requires_approval:
        return PERMISSION_ALLOW_NOW
    resolved = mode if isinstance(mode, PermissionMode) else PermissionMode(str(mode))
    if resolved in {PermissionMode.FULL, PermissionMode.AUTO}:
        return PERMISSION_ALLOW_NOW
    if category and category in set(granted_categories):
        return PERMISSION_ALLOW_NOW
    return PERMISSION_NEED_APPROVAL


def granted_categories(grants: Iterable[Mapping[str, Any]]) -> list[str]:
    """从 state 的授权范围 dump 列表取出工具类别。"""

    return [str(item.get("tool_category") or "") for item in grants]


def scope_matches(
    scope: ApprovalScope,
    *,
    run_id: str,
    category: str,
    question_scope: str,
    rule_version: str = APPROVAL_RULE_VERSION,
) -> bool:
    """「本轮同类操作」是否覆盖当前调用（PRD 10.2：四项全等）。"""

    return (
        scope.run_id == run_id
        and scope.tool_category == category
        and scope.question_scope == question_scope
        and scope.rule_version == rule_version
    )


def build_approval_scope(
    *,
    run_id: str,
    category: str,
    question_scope: str,
    rule_version: str = APPROVAL_RULE_VERSION,
) -> ApprovalScope:
    """构造授权范围（PRD 10.2）。"""

    return ApprovalScope(
        run_id=run_id,
        tool_category=category,
        question_scope=question_scope,
        rule_version=rule_version,
    )


def apply_scope_grant(
    grants: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    category: str,
    question_scope: str,
    rule_version: str = APPROVAL_RULE_VERSION,
) -> list[dict[str, Any]]:
    """把一次「允许本轮同类操作」并入授权列表：保留仍然有效的旧项，按四项去重。"""

    active = [dict(item) for item in grants]
    active.append(
        build_approval_scope(
            run_id=run_id,
            category=category,
            question_scope=question_scope,
            rule_version=rule_version,
        ).model_dump(mode="json")
    )
    deduped: dict[str, dict[str, Any]] = {}
    for item in active:
        key = "|".join(
            (
                str(item.get("run_id") or ""),
                str(item.get("tool_category") or ""),
                str(item.get("question_scope") or ""),
                str(item.get("rule_version") or ""),
            )
        )
        deduped[key] = item
    return list(deduped.values())


def build_display(
    *,
    purpose: str,
    tool_name: str,
    query_or_urls: Sequence[str] = (),
    date_range: str = "",
    task_question: str = "",
    parameters: Mapping[str, Any] | None = None,
    redacted: bool = False,
) -> ApprovalDisplay:
    """构造批准卡片展示内容（PRD 10.2：目的、工具、查询或 URL、日期范围、子任务、脱敏提示）。"""

    return ApprovalDisplay(
        purpose=purpose,
        tool_name=tool_name,
        query_or_urls=[str(item) for item in query_or_urls],
        date_range=date_range,
        task_question=task_question,
        redaction_notice=REDACTION_NOTICE if redacted else "",
        parameters=dict(parameters or {}),
    )


def build_approval_request(
    *,
    run_id: str,
    tool_name: str,
    operation_id: str,
    args_hash: str,
    display: ApprovalDisplay,
    context_epoch: int = 1,
    task_id: str | None = None,
    interrupt_id: str = "",
    rule_version: str = APPROVAL_RULE_VERSION,
) -> ApprovalRequest:
    """构造待保存的批准请求（``approval_id`` 由 ``operation_id`` 稳定派生）。

    稳定派生是必需的：中断节点在恢复时会重跑（PRD 10.3），若每次生成新 ID，同一
    操作会写出第二张卡片。``app_store.request_approval`` 再按 ``approval_id`` /
    ``operation_id`` 做幂等兜底。
    """

    return ApprovalRequest(
        approval_id="ap-" + str(operation_id)[:32],
        run_id=run_id,
        task_id=task_id,
        interrupt_id=interrupt_id,
        operation_id=operation_id,
        args_hash=args_hash,
        context_epoch=int(context_epoch),
        rule_version=rule_version,
        display=display,
        status=ApprovalStatus.PENDING,
        created_at=utc_now_iso(),
    )


def approval_payload(request: ApprovalRequest) -> dict[str, Any]:
    """``approval.requested`` 的 payload（PRD 13.3：完整 ``ApprovalRequest`` 对象）。"""

    from app.agent.events import ApprovalRequestedPayload

    return ApprovalRequestedPayload(approval=request).model_dump(mode="json")


# =====================================================================
# 人名脱敏（PRD 10.4）
# =====================================================================


class NameRedactionResult(ContractModel):
    """一次脱敏的结果：替换映射 + 原因（可复核，PRD 10.4）。"""

    text: str
    replacements: dict[str, str] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.replacements)


def redact_names(
    text: str,
    *,
    private_names: Iterable[str],
    replacements: Mapping[str, str] | None = None,
    prefix: str = "联系人",
) -> NameRedactionResult:
    """把非公开姓名替换为稳定称谓（逐字匹配，先长后短避免子串吞噬）。"""

    mapping = dict(replacements or {})
    original = str(text or "")
    result = original
    names = _ordered_names(private_names)
    for index, name in enumerate(names, start=1):
        if name not in result:
            continue
        alias = mapping.get(name)
        if alias is None:
            alias = f"{prefix}{index}"
            mapping[name] = alias
        result = result.replace(name, alias)
    return NameRedactionResult(text=result, replacements=mapping)


def _ordered_names(private_names: Iterable[str]) -> list[str]:
    """姓名排序：长的优先（避免「张三丰」被「张三」吞掉）；等长按名字定序。

    等长必须**确定**，否则同一次输入在不同进程里会得到不同的称谓编号，
    「稳定称谓」就无从复核（PRD 10.4）。
    """

    unique = {str(name).strip() for name in private_names if str(name).strip()}
    return sorted(unique, key=lambda item: (-len(item), item))


_PERCENT_RE: Final[re.Pattern[str]] = re.compile(r"%[0-9A-Fa-f]{2}")


class NameRedactor:
    """出网参数脱敏器：注册到 :attr:`app.agent.tools.ToolContext.redactor`。

    契约与 ``ToolRegistry.redact_arguments`` 一致：``(tool_name, arguments) -> arguments``。

    - 递归处理字符串标量（dict / list 也进）；
    - ``url`` 类参数若含非公开姓名：能安全替换就保留地址，否则退化为脱敏搜索目标并
      记录未执行原因（PRD 10.4 末段）；
    - 同一姓名在本实例内始终得到同一称谓（稳定称谓）。
    """

    def __init__(
        self,
        private_names: Iterable[str] = (),
        *,
        prefix: str = "联系人",
    ) -> None:
        self._names = _ordered_names(private_names)
        self._prefix = prefix
        self._mapping: dict[str, str] = {}
        self.reasons: list[str] = []

    # ---- 映射管理 ----

    @property
    def replacements(self) -> dict[str, str]:
        return dict(self._mapping)

    def contains_private_name(self, value: Any) -> bool:
        text = str(value or "")
        return any(name in text for name in self._names)

    def _alias_for(self, name: str) -> str:
        alias = self._mapping.get(name)
        if alias is None:
            alias = f"{self._prefix}{len(self._mapping) + 1}"
            self._mapping[name] = alias
        return alias

    def redact_text(self, text: str) -> str:
        result = str(text or "")
        for name in self._names:
            if name in result:
                result = result.replace(name, self._alias_for(name))
        return result

    # ---- 出网入口 ----

    def __call__(self, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return self.redact_arguments(tool, arguments)

    def redact_arguments(self, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return {
            str(key): self._redact_value(str(tool), str(key), value)
            for key, value in dict(arguments).items()
        }

    def _redact_value(self, tool: str, key: str, value: Any) -> Any:
        if isinstance(value, str):
            if value.startswith(("http://", "https://")):
                return self._redact_url(tool, value)
            return self.redact_text(value)
        if isinstance(value, Mapping):
            return {
                str(inner): self._redact_value(tool, str(inner), item)
                for inner, item in value.items()
            }
        if isinstance(value, list):
            return [self._redact_value(tool, key, item) for item in value]
        return value

    def _redact_url(self, tool: str, url: str) -> str:
        """URL 脱敏：能保持有效地址就保持，不能就退化为脱敏搜索目标（不伪造地址）。

        PRD 10.4 要求记录「原始查询与实际发出的联网查询的对应关系」，因此凡是
        **没有**逐字保留地址的情况都要留下原因，便于事后复核。
        """

        if not self._names:
            return url
        direct = self.redact_text(url)
        if direct != url and _is_valid_url(direct):
            # 明文姓名被替换后仍是可打开的地址：保持地址形态，无需解释。
            return direct
        encoded = self._redact_percent_encoded(url)
        if encoded is not None and _is_valid_url(encoded):
            self.reasons.append(
                f"{tool}: URL 含百分号编码的非公开姓名，按解码后替换并重新编码"
                "（原始地址未发出）"
            )
            return encoded
        if direct != url:
            self.reasons.append(
                f"{tool}: URL 含非公开姓名且无法保持有效地址，改用脱敏搜索目标（原始地址未发出）"
            )
            return direct
        return url

    def _redact_percent_encoded(self, url: str) -> str | None:
        if not _PERCENT_RE.search(url):
            return None
        try:
            decoded = unquote(url)
        except Exception:  # noqa: BLE001 - 解码失败即放弃该策略
            return None
        if not self.contains_private_name(decoded):
            return None
        replacement = self.redact_text(decoded)
        if replacement == decoded:  # pragma: no cover - 解码后必然含姓名
            return None
        return _requote(replacement, url)


def _is_valid_url(value: str) -> bool:
    parts = urlsplit(value)
    return parts.scheme in {"http", "https"} and bool(parts.netloc)


def _requote(decoded: str, original: str) -> str | None:
    """把替换后的解码文本重新编码回 URL，保留原始结构（结构变化则返回 None）。"""

    before = urlsplit(original)
    after = urlsplit(decoded)
    if before.scheme != after.scheme or before.netloc != after.netloc:
        return None
    path = quote(after.path, safe="/%@:+~")
    query = quote(after.query, safe="=&%@:+~?")
    return urlunsplit((after.scheme, after.netloc, path, query, after.fragment))


# =====================================================================
# 批准节点（PRD 10.3）
# =====================================================================


def _interrupt(payload: Mapping[str, Any]) -> Any:
    """调用 LangGraph ``interrupt()``；不可用时抛 :class:`ApprovalInterruptUnavailable`。

    离线驱动（无 LangGraph）不走到这里：它在进入批准阶段前就按「未裁决即挂起」处理，
    恢复时再用已裁决的请求调 :func:`approval_node`。
    """

    try:
        from langgraph.types import interrupt
    except ImportError as exc:  # pragma: no cover - langgraph 是 V1.2.0 依赖
        raise ApprovalInterruptUnavailable("当前环境没有 LangGraph interrupt()") from exc
    return interrupt(dict(payload))


def suspend(payload: Mapping[str, Any]) -> Any:
    """公开挂起入口：批准节点与澄清节点共用同一个 ``interrupt()`` 包装。"""

    return _interrupt(payload)


async def approval_node(state: AgentState, *, services: Any) -> dict[str, Any]:
    """批准节点：只读已保存请求 + ``interrupt()``，不做模型调用、不执行工具（PRD 10.3）。

    - 请求已裁决（或运行层已注入裁决）：把决定并入 ``approval_grants`` 并放行回研究阶段；
    - 仍未裁决：``interrupt(卡片)`` 挂起图，恢复值经 ``Command(resume=...)`` 回来。
    """

    dump = dict(state.get("pending_approval") or {})
    if not dump:
        return {"phase": "research"}
    pending = PendingApproval.model_validate(dump)
    request = await services.load_approval(pending.approval_id)
    if request is None:
        return {
            "phase": "research",
            "pending_approval": {},
            "outcomes": _outcome(state, "approval", "批准请求已不存在，按无需批准继续"),
        }

    resume_value: Any = state.get("approval_decision")
    if resume_value is None and request.status is ApprovalStatus.PENDING:
        resume_value = _interrupt(_card_payload(request))
    decision = _coerce_decision(resume_value)
    grants = list(state.get("approval_grants") or [])
    if decision is not None and decision.kind is not ApprovalDecisionKind.DENY:
        grants = apply_scope_grant(
            grants,
            run_id=request.run_id,
            category=tool_category(pending.tool_name),
            question_scope=_question_scope(state),
            rule_version=request.rule_version or APPROVAL_RULE_VERSION,
        )
    kind = decision.kind.value if decision is not None else "unknown"
    return {
        "phase": "research",
        "pending_approval": {},
        "approval_decision": None,
        "approval_grants": grants,
        "outcomes": _outcome(state, "approval", f"批准 {request.approval_id} → {kind}"),
    }


def _card_payload(request: ApprovalRequest) -> dict[str, Any]:
    """交给 ``interrupt()`` 的挂起值：卡片本身（恢复值由 API 侧传入决定）。"""

    return {
        "approval_id": request.approval_id,
        "run_id": request.run_id,
        "tool_name": request.display.tool_name,
        "operation_id": request.operation_id,
        "args_hash": request.args_hash,
        "display": request.display.model_dump(mode="json"),
        "scope_hint": {
            "rule_version": request.rule_version,
            "tool_category": tool_category(request.display.tool_name),
        },
    }


def _coerce_decision(value: Any) -> ApprovalDecision | None:
    """把恢复值收敛为 :class:`ApprovalDecision`（接受对象 / 映射 / 状态字符串）。"""

    if value is None:
        return None
    if isinstance(value, ApprovalDecision):
        return value
    if isinstance(value, ApprovalStatus):
        kind = (
            ApprovalDecisionKind.DENY
            if value is ApprovalStatus.DENIED
            else ApprovalDecisionKind.ALLOW_ONCE
        )
        return ApprovalDecision(kind=kind, decided_at=utc_now_iso())
    if isinstance(value, Mapping):
        if "kind" in value:
            return ApprovalDecision.model_validate(dict(value))
        if "status" in value:
            return _coerce_decision(ApprovalStatus(str(value["status"])))
    return None


def _question_scope(state: AgentState) -> str:
    return str(state.get("question") or "")


def _outcome(state: AgentState, node: str, summary: str) -> list[dict[str, Any]]:
    from app.agent.models import NodeOutcome

    outcomes = list(state.get("outcomes") or [])
    outcomes.append(NodeOutcome(node=node, summary=summary).model_dump(mode="json"))
    return outcomes


async def resolve_approval(
    store: Any,
    *,
    approval_id: str,
    decision: ApprovalDecision,
    run_id: str,
    question_scope: str,
    tool_name: str,
    args_hash: str | None = None,
    context_epoch: int | None = None,
) -> ApprovalRequest:
    """提交用户决定（幂等/冲突语义在 ``app_store.resolve_approval``，这里补授权范围构造）。"""

    scope = None
    if decision.kind is ApprovalDecisionKind.ALLOW_SCOPE:
        scope = build_approval_scope(
            run_id=run_id,
            category=tool_category(tool_name),
            question_scope=question_scope,
        )
    return await store.resolve_approval(
        approval_id,
        decision,
        args_hash=args_hash,
        context_epoch=context_epoch,
        scope=scope,
    )
