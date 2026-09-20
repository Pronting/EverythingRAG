"""RunManager：run 生命周期、子任务调度、事件写入与恢复（PRD 12.2、12.4）。Owner: T09。

职责：
- FastAPI lifespan 创建 RunManager / app.db 连接 / 每工作区 ``AsyncSqliteSaver``。
- 一个会话同时只允许一个活动主任务（并发 409）；多工作区各自执行。
- 子任务独立 ``thread_id``（主图 ``run:<run_id>:main``，子图
  ``run:<run_id>:task:<task_id>``）；某子任务等批准时其他继续；父任务等子任务时停止
  模型调用。
- **先保存后推送**：事件先写 ``run_events``（``event_seq`` 单调）再推给订阅者；
  SSE 用标准 ``id`` + JSON ``data``，``Last-Event-ID`` 走 ``after=`` 补读。
- 取消、重启恢复、``resumable`` 与 ``stop_reason`` 的落库语义。
- 最终答案事务由 :meth:`app_store.AppStore.commit_answer` 单事务完成（PRD 13.1 末段）。

状态取值与迁移规则只从 :mod:`app.agent.contracts` 取，不在此重复定义。

关键取舍（写进架构文档 §6 D36）：

1. **服务装配走注入缝**：``RunManager`` 不 import ``app.api.*``（那是上层），工具上下文
   需要的每区句柄（retriever / search provider / image reader）由构造参数
   ``tool_services`` 提供；默认只有工作区上下文 + app.db，因此 agent 层不反向依赖 API 层。
2. **批准在工具执行前拦截**：:class:`ApprovalGate` 包住 ``ToolRegistry``，在任何检索或
   出网**之前**判定权限（PRD 10.1）。它每次执行都**重读 run 的权限模式** —— 运行期间
   改权限因此在下一次工具执行前生效。
3. **状态以内存为准、检查点为辅**：无检查点时 state 放 ``RunManager._states``；有检查点时
   由 ``AsyncSqliteSaver`` 承载。两条路径的收尾映射（state → run 状态）是同一段代码
   :meth:`RunManager._settle`，避免两种恢复语义分叉。
4. **迟到的结果不能更新正式答案**：取消后 ``_cancelled`` 集合里的 run 拒绝一切
   状态迁移与答案提交（PRD 12.4）。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from app.agent import events as agent_events
from app.agent import graph as agent_graph
from app.agent.app_store import AppStore
from app.agent.approval import (
    PERMISSION_ALLOW_NOW,
    PERMISSION_DENY,
    ApprovalDeniedError,
    ApprovalRequired,
    NameRedactor,
    build_approval_request,
    build_display,
    decide_permission,
    granted_categories,
    tool_category,
)
from app.agent.contracts import (
    CANCELLABLE_RUN_STATUSES,
    DEFAULT_CONTEXT_TOP_K,
    DEFAULT_USER_INPUT_MAX_CHARS,
    SUBSCRIPTION_CLOSING_EVENTS,
    AgentError,
    AgentKind,
    ApprovalRequest,
    ApprovalStatus,
    EventType,
    InputRequest,
    InputRequestStatus,
    PermissionMode,
    ResearchTask,
    RouteDecision,
    RunStatus,
    TaskStatus,
    can_transition_run,
    main_thread_id,
    utc_now_iso,
)
from app.agent.models import (
    AgentState,
    GraphServices,
    PendingApproval,
    new_agent_state,
)

__all__ = [
    "GENERAL_MODEL_TIMEOUT_SECONDS",
    "MODEL_VERIFIER_CHECKS",
    "STOP_REASON_CONVERSATION_MOVED",
    "ApprovalGate",
    "ConversationBusyError",
    "InputTooLongError",
    "RunManager",
    "RunManagerError",
    "RunNotFoundError",
    "RunNotRunnableError",
    "default_services",
    "general_model_for_chat",
    "get_run_manager",
    "model_verifier_for_chat",
    "register_run_manager",
    "sse_frame",
    "synthesis_model_for_chat",
]

logger = logging.getLogger(__name__)

# =====================================================================
# General Agent 的对话模型装配（B 项：此前**从未接线**）
# =====================================================================
#
# 缺陷（t29 在真实应用上复现）：``default_services.research`` 调
# ``research_task(...)`` 时没有传 ``general_model``，而 ``researcher_for(GENERAL)`` 与
# ``GeneralResearcher(model=None)`` 都默认 None ⇒ general 意图要么走
# ``GeneralResearcher(model=None)`` 的结构化失败、要么在别处抛 AttributeError；
# 「配了对话 provider 也照样挂」——因为这条线根本没接。
#
# ``GeneralModel = Callable[[ResearchRequest], str]`` 是**同步**签名（research.py 拥有），
# 而 provider 的 ``stream_chat`` 是 async，因此本模块提供一个**同步桥**：在独立 worker
# 线程里用**它自己的事件循环**创建 provider 并跑一次生成，再取回文本。
# 权衡（如实记录）：这一次模型调用期间主事件循环被阻塞 —— 单用户本地服务可接受；
# 更彻底的做法是让 ``GeneralModel`` 接受 awaitable（research.py 两行改动），
# 已作为后续建议上报，不在本任务范围内。

#: 通用解释模型单次调用的上限（超时即按服务错误报出，而不是无限挂住 run）。
GENERAL_MODEL_TIMEOUT_SECONDS: Final[float] = 120.0

#: 同步桥用的 worker：每次调用在自己的线程里 ``asyncio.run``，避免与主循环共享 provider
#: 客户端（httpx 客户端与首次使用的事件循环绑定，跨循环复用会报错）。
_GENERAL_EXECUTOR: Final[ThreadPoolExecutor] = ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="general-model"
)


def _chat_config_from_settings() -> Any:
    from app.core.settings_store import get_settings_store

    return get_settings_store().load().chat


def _chat_provider_available(chat_config: Any) -> bool:
    """探测对话 provider 是否已配置（**零联网**：只做构造期校验，不发起请求）。"""

    from app.generation.providers import create_chat_model_from_config

    try:
        create_chat_model_from_config(chat_config)
    except Exception:  # noqa: BLE001 - 任何构造期失败都按「未配置」处理（fail-closed）
        return False
    return True


async def _generate_general_text(chat_config: Any, prompt: str) -> str:
    """在**本线程自己的事件循环**里构造 provider 并收集一次生成结果。"""

    return await _chat_text(chat_config, prompt)


async def _chat_text(chat_config: Any, prompt: str) -> str:
    """直接 ``await`` provider 收集一次生成（异步节点用；同步桥再包一层线程）。"""

    from app.generation.providers import create_chat_model_from_config

    model = create_chat_model_from_config(chat_config)
    parts: list[str] = []
    async for chunk in model.stream_chat([{"role": "user", "content": prompt}]):
        if str(getattr(chunk, "kind", "content")) == "content":
            parts.append(str(getattr(chunk, "text", "")))
    return "".join(parts)


def _run_general_blocking(chat_config: Any, prompt: str) -> str:
    # 延迟 import：``app.agent.tools`` 反过来会用到运行层类型，模块级 import 会成环。
    from app.agent.tools import ToolErrorCode, ToolUnavailableError
    from app.generation.providers import ChatProviderError

    future = _GENERAL_EXECUTOR.submit(asyncio.run, _generate_general_text(chat_config, prompt))
    try:
        return future.result(timeout=GENERAL_MODEL_TIMEOUT_SECONDS)
    except FutureTimeoutError as exc:
        raise ToolUnavailableError(
            f"通用解释模型在 {GENERAL_MODEL_TIMEOUT_SECONDS:.0f}s 内没有返回",
            code=ToolErrorCode.SERVICE_ERROR,
        ) from exc
    except ChatProviderError as exc:
        # 提供方自身的失败（网络/鉴权/协议）：转成**带类型**的服务错误（PRD 12.5），
        # 而不是把 provider 的异常类型原样透给用户。其余异常一律照实上抛（不掩盖缺陷）。
        raise ToolUnavailableError(
            str(exc) or "通用解释模型调用失败", code=ToolErrorCode.SERVICE_ERROR
        ) from exc


def _general_prompt(request: Any) -> str:
    """General Agent 的提示词：只作答 + 不生成引用编号（PRD 4.2 / 8.1）。"""

    task = getattr(request, "task", None)
    question = str(
        getattr(task, "question", "") or getattr(request, "original_question", "") or ""
    )
    requirement_texts = getattr(request, "requirement_texts", None) or {}
    requirement_ids = [str(item) for item in (getattr(task, "requirement_ids", None) or [])]
    points = [
        str(requirement_texts.get(requirement_id) or "").strip()
        for requirement_id in requirement_ids
    ]
    lines = [
        "你是通用解释助手，用中文直接回答用户的问题。",
        "如果用户是在问方案或设计，请给出可执行的建议；如果是在问原理，请解释清楚。",
        (
            "只使用你自己的通用知识，不要编造文档、链接或来源；"
            "不要输出任何形如 [S1] / [W1] 的引用编号。"
        ),
        "",
        f"用户问题：{question}",
    ]
    points = [point for point in points if point]
    if points:
        lines.append("需要覆盖的要点：")
        lines.extend(f"- {point}" for point in points)
    return "\n".join(lines)


def general_model_for_chat(chat_config: Any = None) -> Any | None:
    """构造 General Agent 的模型适配器；**未配置返回 None**。

    **这是 chat 模型的唯一注入点**：

    - 生产：不传参 → 读 ``settings_store.load().chat``；
    - 测试（验收方要用的接法）：``monkeypatch.setattr(
      "app.generation.providers.create_chat_model_from_config", fake)`` —— 本函数与
      :func:`_generate_general_text` 都**在调用时**从模块取该工厂，因此真实
      ``default_services`` 装配路径原样走通（provider 实例是在 worker 线程里新建的，
      假模型不需要任何出网）；
    - 也可以直接 monkeypatch 本函数（返回一个 ``Callable[[ResearchRequest], str]``）。

    返回 ``None`` 表示「未配置」：``GeneralResearcher(model=None)`` 会给出**结构化**失败
    （``error_code=not_configured`` + 用户可读说明 + 失败节点 + 可继续），
    这正是 PRD 12.5 要的失败方式；**绝不是** AttributeError。
    """

    config = _chat_config_from_settings() if chat_config is None else chat_config
    if not _chat_provider_available(config):
        return None

    def call(request: Any) -> str:
        return _run_general_blocking(config, _general_prompt(request))

    return call


# ---------------------------------------------------------------------------
# ③ 模型核对器（PRD 8.2 模型侧）与 ④ 候选答案由模型撰写（PRD 8.1 / 8.3）
# ---------------------------------------------------------------------------

#: 允许模型核对自己判定的检查项（PRD 8.2）。字段/格式/引用身份等**程序判定**项
#: 不接受模型结论 —— 否则模型可以替程序宣布「引用有效」。
MODEL_VERIFIER_CHECKS: Final[tuple[str, ...]] = (
    "factual_support",
    "version_and_scope",
    "source_conflict",
)


def _extract_json_object(raw: str) -> Mapping[str, Any] | None:
    """从模型输出里取出第一个可解析的 JSON 对象（模型常带解释文字或代码块）。"""

    text = str(raw or "")
    start = text.find("{")
    while start >= 0:
        end = text.rfind("}")
        while end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except ValueError:
                end = text.rfind("}", start, end)
                continue
            if isinstance(parsed, Mapping):
                return parsed
            break
        start = text.find("{", start + 1)
    return None


class _ChatModelVerifier:
    """用对话模型判断**语义支持**（PRD 8.2 的模型侧那一项）。

    ``review`` 是 async（与 :class:`app.agent.verification.ModelVerifier` 协议一致），
    因此这里直接 ``await`` provider，不需要走同步桥。

    **失败口径（刻意 fail-open）**：模型不可用或输出不可解析时返回**空 issues**并附一条
    ``notes`` —— 程序侧八项检查仍在，答案不会因为「模型没能复核」而被判死；反过来把解析
    失败当成发现问题，会让核对结论随机失败（PRD 8.2 的模型判断是补充，不是唯一闸门）。
    """

    def __init__(self, chat_config: Any) -> None:
        self._config = chat_config

    async def review(self, request: Any) -> Any:
        from app.agent.verification import ModelVerification

        try:
            raw = await _chat_text(self._config, _verifier_prompt(request))
        except Exception as exc:  # noqa: BLE001 - 见类 docstring 的 fail-open 口径
            logger.warning("模型核对不可用：error=%s", type(exc).__name__)
            return ModelVerification(notes=["模型核对不可用，本轮仅按程序检查判定"])
        return _parse_model_verification(raw)


def _parse_model_verification(raw: str) -> Any:
    from app.agent.contracts import VerificationCheck, VerificationSeverity
    from app.agent.verification import ModelVerification, ModelVerificationIssue

    payload = _extract_json_object(raw)
    if payload is None:
        return ModelVerification(notes=["模型核对输出不是可解析的 JSON，按未发现问题处理"])
    issues: list[Any] = []
    for item in payload.get("issues") or []:
        if not isinstance(item, Mapping):
            continue
        check = str(item.get("check") or "").strip()
        if check not in MODEL_VERIFIER_CHECKS:
            continue  # 模型不得替程序判定「引用有效/数字格式」等程序项
        issues.append(
            ModelVerificationIssue(
                check=VerificationCheck(check),
                statement=str(item.get("statement") or ""),
                evidence_ids=[str(value) for value in (item.get("evidence_ids") or [])],
                requirement_ids=[str(value) for value in (item.get("requirement_ids") or [])],
                required_action=str(item.get("required_action") or ""),
                severity=VerificationSeverity(str(item.get("severity") or "medium")),
            )
        )
    questions = [
        str(question).strip()
        for question in (payload.get("ambiguity_questions") or [])
        if str(question).strip()
    ]
    return ModelVerification(
        issues=issues,
        ambiguity_questions=questions,
        notes=[str(note) for note in (payload.get("notes") or [])],
    )


def _verifier_prompt(request: Any) -> str:
    """模型核对提示词：只判语义支持/版本范围/来源冲突，且**必须给 JSON**。"""

    lines = [
        (
            "你是答案核对助手。逐条判断下列结论是否真由给定证据支持（语义层面），"
            "以及是否存在版本/范围不一致或来源互相冲突。"
        ),
        "只输出一个 JSON 对象，形如：",
        (
            '{"issues":[{"check":"factual_support|version_and_scope|source_conflict",'
            '"statement":"结论原文","evidence_ids":["ev-…"],"requirement_ids":["R1"],'
            '"required_action":"怎么改","severity":"low|medium|high"}],'
            '"ambiguity_questions":["需要用户澄清的问题"],"notes":[]}'
        ),
        '没有问题就输出 {"issues":[]}。不要输出其它文字。',
        "",
        f"问题：{getattr(request, 'question', '')}",
        f"候选答案：{getattr(request, 'content', '')}",
        "结论与证据：",
    ]
    for conclusion in getattr(request, "conclusions", None) or []:
        lines.append(
            f"- {conclusion.statement}（依据：{', '.join(conclusion.evidence_ids) or '无'}）"
        )
    for evidence in getattr(request, "evidence", None) or []:
        lines.append(f"[{evidence.evidence_id}] {str(evidence.content)[:200]}")
    return "\n".join(lines)


def synthesis_model_for_chat(chat_config: Any = None) -> Callable[[Any], str] | None:
    """构造「由模型撰写候选答案」的同步可调用（PRD 8.1 / 8.3）；未配置返回 None。

    返回 ``None`` 时 ``synthesize_answer`` 走确定性 ``render_candidate``（兜底）。
    模型输出**必须带 ``{{ev-…}}`` 占位符**（编号由 :func:`assign_citation_ids` 分配）；
    若模型忽略该要求或调用失败，则**回落到确定性渲染** —— 答案必须带来源（PRD 8.3），
    宁可用结构化渲染，也不发布一篇没有引用的正文。
    """

    config = _chat_config_from_settings() if chat_config is None else chat_config
    if not _chat_provider_available(config):
        return None

    def call(request: Any) -> str:
        from app.agent.contracts import citation_prefix
        from app.agent.verification import render_candidate

        def fallback() -> str:
            """两处兜底共用同一确定性渲染（答案必须带来源，PRD 8.3）。"""

            return render_candidate(request.conclusions, request.requirements)
        citable = [
            item
            for item in request.evidence
            if citation_prefix(item.source_type) is not None
        ]
        try:
            text = _run_general_blocking(config, _synthesis_prompt(request, citable)).strip()
        except Exception as exc:  # noqa: BLE001 - 提供方失败 → 兜底渲染（不判死 run）
            logger.warning("模型撰写候选答案失败：error=%s", type(exc).__name__)
            return fallback()
        if not text:
            return fallback()
        if citable and not any(
            f"{{{{{item.evidence_id}}}}}" in text for item in citable
        ):
            logger.warning("模型正文缺少来源占位符，回落确定性渲染（PRD 8.3）")
            return fallback()
        return text

    return call


def _synthesis_prompt(request: Any, citable: Sequence[Any]) -> str:
    lines = [
        "你是知识助理。根据下列**已核实结论**与**证据**，用中文写一段连贯的答案正文。",
        "硬性要求：",
        (
            "1. 每条来自文档/网页/记忆/历史的陈述后面必须紧跟对应的证据占位符，"
            "形如 {{ev-xxxx}}（只能使用给出的占位符，不要自己编 id，也不要写成 [S1]）；"
        ),
        "2. 通用解释与建议不要加任何占位符或编号；",
        "3. 不要编造证据之外的事实；不要输出 JSON 或解释你自己的写作过程。",
        "",
        f"问题：{getattr(request, 'question', '')}",
        "已核实结论与占位符：",
    ]
    for conclusion in getattr(request, "conclusions", None) or []:
        markers = "".join(f"{{{{{eid}}}}}" for eid in conclusion.evidence_ids)
        lines.append(f"- {conclusion.statement}{markers}")
    lines.append("可引用的证据占位符（仅这些可引用）：")
    for item in citable:
        lines.append(f"{{{{{item.evidence_id}}}}} ← {str(item.content)[:200]}")
    return "\n".join(lines)


def model_verifier_for_chat(chat_config: Any = None) -> Any | None:
    """构造模型核对器；未配置返回 ``None``（此时只有程序侧八项检查）。"""

    config = _chat_config_from_settings() if chat_config is None else chat_config
    if not _chat_provider_available(config):
        return None
    return _ChatModelVerifier(config)


#: 因**会话移动**终止的 run 的 ``stop_reason``（PRD 5.4 原文取值）。
#: 放在运行层：这个取值不只是个字符串 —— 它决定 run **不可恢复**
#: （``contracts.can_transition_run`` 的 ``cancelled → running`` 要求 ``resumable``），
#: 因此不变量必须由状态机一侧兜住，而不是只靠调用方记得传 ``resumable=False``。
STOP_REASON_CONVERSATION_MOVED: Final[str] = "conversation_moved"

#: 取消后仍收到结果时使用的标记（PRD 12.4：迟到结果不能更新正式答案）。
_LATE_RESULT_GRACE: Final[int] = 0


class RunManagerError(RuntimeError):
    """RunManager 语义错误的基类。"""


class RunNotFoundError(RunManagerError):
    """run 不存在（API 层 404）。"""


class ConversationBusyError(RunManagerError):
    """同一会话已有活动主任务（PRD 12.2，API 层 409）。"""


class RunNotRunnableError(RunManagerError):
    """run 当前状态不允许该操作（批准/澄清提交、取消、恢复；API 层 409）。"""


class InputTooLongError(RunManagerError):
    """输入超过 PRD 13.2 的长度上限（API 层 422）。"""


# =====================================================================
# 权限门（工具执行前拦截，PRD 10.1 / 10.2）
# =====================================================================


@dataclass
class ApprovalGate:
    """包住 ``ToolRegistry``：需要批准的工具在**执行前**停顿（PRD 10.1）。

    - 每次执行前重读 run 的权限模式（运行期间改权限立即生效）；
    - 命中「本轮同类操作」授权或已有同 ``operation_id`` 的批准 → 放行；
    - 已被拒绝的同一操作 → :class:`ApprovalDeniedError`（不再重试）；
    - 否则写 ``approvals`` + 发 ``approval.requested`` + 抛 :class:`ApprovalRequired`
      （研究节点据此挂起，父图停止模型调用）。
    """

    manager: Any
    registry: Any
    run: Any
    state: AgentState
    question_scope: str = ""
    redactor: NameRedactor | None = None

    def __getattr__(self, item: str) -> Any:
        return getattr(self.registry, item)

    async def execute(
        self,
        name: str,
        arguments: str | Mapping[str, Any],
        context: Any,
        **kwargs: Any,
    ) -> Any:
        """执行一次工具调用；需要批准时先过权限门，**出网参数一定是脱敏后的值**。

        注意一处刻意的差异：审批记录绑定的是**原始参数**的 ``args_hash``（用户看到的
        是自己问的话），而真正执行时下发给工具的是脱敏后的参数（PRD 10.4）。两者都
        可复核：卡片里有展示参数，工具侧收到的是称谓化后的查询。
        """

        spec = self.registry.get(name)
        effective: str | Mapping[str, Any] = arguments
        if bool(getattr(spec, "requires_approval", False)):
            effective = await self._ensure(name, arguments, context, spec)
        params = self.registry.parse_arguments(spec, effective)
        safe = self._redact(str(spec.name), params.model_dump(mode="json"))
        return await self.registry.execute(name, safe, context, **kwargs)

    # ---- 内部 ----

    async def _ensure(self, name: str, arguments: Any, context: Any, spec: Any) -> Any:
        from app.agent.tools import arguments_hash, stable_operation_id

        params = self.registry.parse_arguments(spec, arguments)
        payload = params.model_dump(mode="json")
        args_hash = arguments_hash(payload)
        operation_id = stable_operation_id(
            context.run_id, context.task_id, str(spec.name), args_hash
        )
        category = tool_category(str(spec.name))

        current = await self.manager.store.get_run(self.run.id)
        mode = str(current.permission_mode if current is not None else self.run.permission_mode)
        # 运行期间改权限要在**下一次工具执行前**生效（PRD 10.1）：用户改的是会话级
        # 权限开关，因此这里以会话当前值为准，run 上的值只作为兜底。
        conversation = await self.manager.store.get_conversation(self.run.conversation_id)
        if conversation is not None:
            mode = str(conversation.permission_mode or mode)

        existing = await self._find_approval(operation_id)
        if existing is not None and existing.status is ApprovalStatus.DENIED:
            raise ApprovalDeniedError(existing.approval_id)
        if existing is not None and existing.status is ApprovalStatus.APPROVED:
            # 「本轮同类操作」的授权范围随批准一起生效（PRD 10.2）：登记到运行层，
            # 后续同类调用即使不在同一次节点执行里也能免停顿。
            if existing.scope is not None:
                self.manager.register_grant(self.run.id, existing.scope.model_dump(mode="json"))
            return self._effective_arguments(existing, params, payload)

        decision = decide_permission(
            mode=mode,
            requires_approval=True,
            category=category,
            granted_categories=self.manager.granted_for(
                self.run.id, self.state.get("approval_grants") or []
            ),
        )
        if decision == PERMISSION_DENY:
            raise ApprovalDeniedError(operation_id)
        if decision == PERMISSION_ALLOW_NOW:
            return arguments

        redacted = self._redact(str(spec.name), payload)
        request = build_approval_request(
            run_id=self.run.id,
            tool_name=str(spec.name),
            operation_id=operation_id,
            args_hash=args_hash,
            display=build_display(
                purpose=f"执行 {spec.name} 以取得外部资料",
                tool_name=str(spec.name),
                query_or_urls=_query_or_urls(redacted),
                date_range=_date_range_text(redacted),
                task_question=str(context.task_id or ""),
                parameters=_display_safe(redacted),
                redacted=self.redactor is not None,
            ),
            context_epoch=int(self.run.context_epoch or 1),
            task_id=context.task_id,
        )
        async with self.manager.db():
            stored = await self.manager.store.request_approval(request)
        await self.manager.publish_event(
            self.run.id,
            EventType.APPROVAL_REQUESTED,
            {"approval": stored.model_dump(mode="json")},
            task_id=context.task_id,
            object_id=stored.approval_id,
        )
        raise ApprovalRequired(
            PendingApproval(
                approval_id=stored.approval_id,
                tool_name=str(spec.name),
                operation_id=stored.operation_id,
                args_hash=stored.args_hash,
                args_summary=_args_summary(redacted),
                task_id=context.task_id,
                category=category,
                question_scope=self.question_scope,
            )
        )

    async def _find_approval(self, operation_id: str) -> ApprovalRequest | None:
        async with self.manager.db():
            for item in await self.manager.store.list_approvals(self.run.id):
                if item.operation_id == operation_id:
                    return item
        return None

    def _effective_arguments(self, request: ApprovalRequest, params: Any, payload: Any) -> Any:
        """批准后执行：用户编辑过参数则用编辑后的参数（PRD 10.2）。"""

        decision = request.decision
        if decision is not None and decision.edited_parameters:
            return dict(decision.edited_parameters)
        return payload

    def _redact(self, tool: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self.redactor is None:
            return dict(payload)
        return dict(self.redactor(tool, payload))


def _query_or_urls(redacted: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("query", "url", "urls", "queries"):
        raw = redacted.get(key)
        if isinstance(raw, str) and raw:
            values.append(raw)
        elif isinstance(raw, list):
            values.extend(str(item) for item in raw if item)
    return values


def _date_range_text(redacted: Mapping[str, Any]) -> str:
    start = redacted.get("start_date") or redacted.get("date_from") or ""
    end = redacted.get("end_date") or redacted.get("date_to") or ""
    if start or end:
        return f"{start or '…'} ~ {end or '…'}"
    return ""


#: 批准卡片参数里**不进界面**的键（可能承载绝对路径、密钥或认证头，PRD 12.5）。
_CARD_HIDDEN_KEYS: Final[frozenset[str]] = frozenset(
    {"api_key", "authorization", "headers", "token", "source_path", "abs_path", "root_dir"}
)


def _display_safe(redacted: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in redacted.items()
        if str(key).casefold() not in _CARD_HIDDEN_KEYS
    }


def _args_summary(redacted: Mapping[str, Any]) -> str:
    parts = [f"{key}={value}" for key, value in _display_safe(redacted).items() if value != ""]
    return ", ".join(parts)[:300]


# =====================================================================
# SSE 帧
# =====================================================================


def sse_frame(event: agent_events.StoredEvent) -> str:
    """SSE 帧：标准 ``id`` + JSON ``data``（PRD 13.3）。"""

    return f"id: {event.event_seq}\ndata: {agent_events.dumps_event(event)}\n\n"


def sse_error_frame(*, error_code: str, message: str) -> str:
    """协议错误帧（不在 20 个存储事件里）：界面据此显示协议错误而不是静默丢帧。"""

    body = {"type": "error", "error_code": error_code, "message": message}
    import json

    return f"data: {json.dumps(body, ensure_ascii=False, separators=(',', ':'))}\n\n"


# =====================================================================
# RunManager
# =====================================================================


@dataclass
class _Subscription:
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)


class RunManager:
    """run 生命周期的唯一入口（PRD 12.2）。"""
    def __init__(
        self,
        *,
        store: AppStore,
        context_factory: Any = None,
        registry: Any = None,
        tool_services: Callable[[Any], Mapping[str, Any]] | None = None,
        services_factory: Callable[[RunManager, Any, AgentState], GraphServices] | None = None,
        use_langgraph: bool = True,
        private_names: Sequence[str] = (),
        research_concurrency: int = 3,
        image_ingest: Callable[[Any, Sequence[str]], Awaitable[list[dict[str, Any]]]]
        | None = None,
    ) -> None:
        self.store = store
        self.context_factory = context_factory
        self.registry = registry
        self.tool_services = tool_services
        self._services_factory = services_factory or default_services
        self.use_langgraph = use_langgraph
        self.private_names = tuple(private_names)
        self.research_concurrency = max(1, int(research_concurrency))
        #: 图片引用落库实现（PRD 13.2）：由 ``app/api/run_services.py`` 注入本区
        #: 图片库 / 下载通道 / 识图管线；None = 只记录「未落库」而不假装支持。
        self.image_ingest = image_ingest
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._states: dict[str, AgentState] = {}
        self._subscriptions: dict[str, list[_Subscription]] = {}
        self._cancelled: set[str] = set()
        #: 本轮已生效的「允许本轮同类操作」授权（按 run 记）——批准可能发生在
        #: 节点之外（如恢复后工具直接经权限门执行），因此授权不能只存 state。
        self._grants: dict[str, list[dict[str, Any]]] = {}
        #: 本 run 的「已筛选来源」相关性记录（按 ``source_ref`` 索引）。由检索工具经
        #: :func:`_rerank_outcome_sink` 写入，合成阶段交给 ``assign_citation_ids``，
        #: 成为 ``answer.meta.citations`` 的追加字段（集成 t5）。只在内存里按 run 存活，
        #: 与 ``_states`` / ``_grants`` 同一生命周期。
        self._rerank_sources: dict[str, dict[str, dict[str, Any]]] = {}
        #: app.db 只有一个 aiosqlite 连接：后台驱动与请求处理并发时，两个
        #: ``BEGIN IMMEDIATE`` 会互相踩（实测 ``cannot start a transaction within a
        #: transaction``）。因此**每一次** store 调用都经这把锁串行化，粒度是单次调用
        #: （不是整次驱动），API 读请求不会被长任务饿死。
        self._db_lock = asyncio.Lock()
        # 刻意**不**按工作区缓存编译产物：节点闭包绑定的是**当前 run**（emit / publish /
        # 工具上下文），跨 run 复用会把第二个 run 的事件写进第一个 run（D36）。因此主图与
        # 研究子图都在 :meth:`_invoke` 里按本次 run 现编译；检查点仍按工作区缓存
        # （``checkpoints.db`` 每区一个文件）。曾经存在的 ``_graphs`` / ``_research_graphs``
        # 两个"按区缓存"字典**从未被写入**，只会误导读代码的人以为子图没接线 —— 已删除。
        self._checkpointers: dict[str, Any] = {}
        self._checkpoint_stack: contextlib.AsyncExitStack | None = None
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    # =================================================================
    # 创建与查询
    # =================================================================

    async def create_run(
        self,
        *,
        workspace_id: str,
        conversation_id: str,
        question: str,
        client_request_id: str | None = None,
        permission_mode: PermissionMode | None = None,
        reasoning_level: Any = None,
        images: Sequence[str] = (),
        origin_run_id: str | None = None,
    ) -> Any:
        """创建 run（PRD 13.2 建 run 路径）：保存用户消息 + 创建任务，同一次请求内完成。

        - 输入长度超限 → :class:`InputTooLongError`（API 层 422）；
        - 同一会话已有活动主任务 → :class:`ConversationBusyError`（API 层 409）；
        - ``client_request_id`` 会话内唯一：重复请求幂等返回**既有** run；
        - ``images``（PRD 13.2「图片引用」）：交给注入的 ``image_ingest`` 落**本区**图片库，
          逐条结果（含 ``status`` / ``reason``）写进用户消息 ``source_snapshot["images"]`` ——
          **绝不 accept-and-drop**：拿不到字节或未配置视觉时也留下如实的原因；
        - ``origin_run_id``（PRD 5.4）：会话移动后在**目标工作区**继续当前问题时填**被移动
          终止的那个 run**，形成可追溯链。新 run 用全新的 thread/checkpoint，状态从零开始，
          因此原区未发布的研究结果不会进入新执行。
        """

        text = str(question or "")
        if not text.strip():
            raise RunManagerError("问题不能为空")
        if len(text) > DEFAULT_USER_INPUT_MAX_CHARS:
            raise InputTooLongError(
                f"输入长度 {len(text)} 超过上限 {DEFAULT_USER_INPUT_MAX_CHARS}（PRD 13.2）"
            )
        async with self.db():
            conversation = await self.store.get_conversation(conversation_id)
        if conversation is None:
            raise RunManagerError("会话不存在")
        if conversation.workspace_id != workspace_id:
            raise RunManagerError("会话不属于该工作区")
        if client_request_id is not None:
            async with self.db():
                existing = await self.store.find_run_by_client_request(
                    conversation_id, client_request_id
                )
            if existing is not None:
                return existing
        async with self.db():
            active = await self.store.list_active_runs(conversation_id)
        if active:
            raise ConversationBusyError(
                f"会话 {conversation_id} 已有活动任务 {active[0].id}（PRD 12.2）"
            )
        image_refs = await self._resolve_images(workspace_id, images)
        async with self.db():
            message = await self.store.append_message(
                conversation_id=conversation_id,
                role="user",
                content=text,
                source_snapshot={"images": image_refs} if image_refs else None,
            )
            created = await self.store.create_run(
                conversation_id=conversation_id,
                workspace_id=workspace_id,
                client_request_id=client_request_id,
                input_message_id=message.id,
                permission_mode=permission_mode,
                reasoning_level=reasoning_level,
                origin_run_id=origin_run_id,
            )
        # 新 run 从零开始：清掉可能残留的筛选来源记录（run id 唯一，这里只是防御）。
        self._rerank_sources.pop(str(getattr(created, "id", "") or ""), None)
        return created

    async def _resolve_images(
        self, workspace_id: str, images: Sequence[str]
    ) -> list[dict[str, Any]]:
        """把本次 run 的图片引用交给注入的落库实现；未注入时**如实标记未落库**。"""

        refs = [str(item) for item in images if str(item or "").strip()]
        if not refs:
            return []
        if self.image_ingest is None:
            return [
                {
                    "image_id": "",
                    "block_id": "",
                    "content_hash": "",
                    "kind": "unknown",
                    "status": "unavailable",
                    "reason": "运行层未接入图片落库（本轮只记录引用，不宣称已生效）",
                    "src_ref": "",
                    "recorded_at": utc_now_iso(),
                }
                for _ in refs
            ]
        context = self._context(workspace_id)
        try:
            produced = await self.image_ingest(context, refs)
        except Exception as exc:  # noqa: BLE001 - 图片落库失败不阻断建 run，但要如实记录
            logger.warning("图片引用落库失败：error=%s", type(exc).__name__)
            return [
                {
                    "image_id": "",
                    "block_id": "",
                    "content_hash": "",
                    "kind": "unknown",
                    "status": "unavailable",
                    "reason": f"图片落库失败：{type(exc).__name__}",
                    "src_ref": "",
                    "recorded_at": utc_now_iso(),
                }
                for _ in refs
            ]
        return [dict(item) for item in (produced or [])]

    async def image_refs(self, run_id: str) -> list[dict[str, Any]]:
        """读回某次 run 用户消息上的图片引用处理结果（PRD 13.2「图片引用」可核对）。"""

        run = await self.require_run(run_id)
        if not run.input_message_id:
            return []
        async with self.db():
            message = await self.store.get_message(run.input_message_id)
        if message is None:
            return []
        snapshot = getattr(message, "source_snapshot", None) or {}
        refs = snapshot.get("images") if isinstance(snapshot, Mapping) else None
        return [dict(item) for item in (refs or [])]

    async def require_run(self, run_id: str) -> Any:
        async with self.db():
            run = await self.store.get_run(run_id)
        if run is None:
            raise RunNotFoundError(f"run 不存在：{run_id}")
        return run

    def is_running(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        return task is not None and not task.done()

    # ---- app.db 串行化（见 __init__ 的 _db_lock 说明） ----

    @contextlib.asynccontextmanager
    async def db(self) -> AsyncIterator[None]:
        """串行化一段 app.db 访问（后台驱动与请求处理共用同一连接）。"""

        async with self._db_lock:
            yield

    # ---- 供 API 层使用的受保护读取（避免请求与驱动并发开事务） ----

    async def get_approval(self, approval_id: str) -> Any:
        async with self.db():
            return await self.store.get_approval(approval_id)

    async def get_input_request(self, input_request_id: str) -> Any:
        async with self.db():
            return await self.store.get_input_request(input_request_id)

    async def get_conversation(self, conversation_id: str) -> Any:
        async with self.db():
            return await self.store.get_conversation(conversation_id)

    async def list_runs(self, conversation_id: str, *, limit: int | None = None) -> list[Any]:
        async with self.db():
            return await self.store.list_runs(conversation_id, limit=limit)

    async def list_tasks(self, run_id: str) -> list[Any]:
        async with self.db():
            return await self.store.list_tasks(run_id)

    async def list_pending_approvals(self, run_id: str) -> list[Any]:
        async with self.db():
            return await self.store.list_approvals(run_id, status=ApprovalStatus.PENDING)

    async def list_pending_inputs(self, run_id: str) -> list[Any]:
        async with self.db():
            return await self.store.list_input_requests(run_id, status=InputRequestStatus.PENDING)

    async def last_event_seq(self, run_id: str) -> int:
        async with self.db():
            return await self.store.last_event_seq(run_id)

    async def get_answer(self, answer_id: str) -> Any:
        async with self.db():
            return await self.store.get_answer(answer_id)

    # ---- 「本轮同类操作」授权（PRD 10.2） ----

    def register_grant(self, run_id: str, scope: Mapping[str, Any]) -> None:
        """登记一次授权范围（去重，按 run 保留）。"""

        from app.agent.approval import apply_scope_grant

        current = self._grants.get(run_id, [])
        self._grants[run_id] = apply_scope_grant(
            [*current, dict(scope)],
            run_id=str(scope.get("run_id") or run_id),
            category=str(scope.get("tool_category") or ""),
            question_scope=str(scope.get("question_scope") or ""),
            rule_version=str(scope.get("rule_version") or ""),
        )

    def granted_for(self, run_id: str, state_grants: Sequence[Mapping[str, Any]] = ()) -> list[str]:
        """把 state 里的授权与运行层登记的授权合并成工具类别清单。"""

        return granted_categories([*list(state_grants), *self._grants.get(run_id, [])])

    def grants_for_state(self, run_id: str) -> list[dict[str, Any]]:
        return [dict(item) for item in self._grants.get(run_id, [])]

    # =================================================================
    # 已筛选来源（集成 t5：引用卡片的「已筛选资料」数据通路）
    # =================================================================

    def register_rerank_sources(self, run_id: str, records: Sequence[Mapping[str, Any]]) -> None:
        """登记本 run 入选来源的相关性记录（按 ``source_ref`` 去重，**先到先得**）。

        同一个 source_ref 可能被多轮检索各自评分（分数会随查询变），引用卡片只能展示一个；
        这里保留**首次入选**的记录：它对应「第一条把它带进本轮证据的检索」，可复现且不随
        后续轮次抖动。空记录（``source_ref`` 为空）直接忽略，绝不写入没有身份的条目。
        """

        bucket = self._rerank_sources.setdefault(run_id, {})
        for item in records:
            source_ref = str((item or {}).get("source_ref") or "")
            if not source_ref or source_ref in bucket:
                continue
            bucket[source_ref] = dict(item)

    def rerank_sources(self, run_id: str) -> dict[str, dict[str, Any]]:
        """本 run 已登记的相关性记录（``{source_ref: 记录}``；未登记时为空 dict）。"""

        return {key: dict(value) for key, value in self._rerank_sources.get(run_id, {}).items()}

    # =================================================================
    # 驱动
    # =================================================================

    async def start(self, run_id: str, *, resume: bool = False) -> None:
        """后台启动执行（不阻塞请求线程）。"""

        await self.require_run(run_id)
        if self.is_running(run_id):
            return
        self._cancelled.discard(run_id)
        task = asyncio.create_task(self._drive_guarded(run_id, resume=resume))
        self._tasks[run_id] = task

    async def run_to_completion(self, run_id: str, *, timeout: float | None = None) -> Any:
        """前台执行到收敛（测试与内部调用用）。"""

        await self.start(run_id)
        return await self.wait(run_id, timeout=timeout)

    async def wait(self, run_id: str, *, timeout: float | None = None) -> Any:
        task = self._tasks.get(run_id)
        if task is not None:
            with contextlib.suppress(asyncio.TimeoutError):
                if timeout is None:
                    await asyncio.shield(task)
                else:
                    await asyncio.wait_for(asyncio.shield(task), timeout)
        return await self.require_run(run_id)

    async def _drive_guarded(self, run_id: str, *, resume: bool) -> None:
        try:
            await self._drive(run_id, resume=resume)
        except asyncio.CancelledError:  # 取消路径自己写状态，这里只吞掉
            raise
        except Exception as exc:  # noqa: BLE001 - 任何异常都必须落到 run.failed
            logger.warning("run %s 执行失败：error=%s", run_id, type(exc).__name__)
            await self._fail(run_id, exc, node="runtime")
        finally:
            self._tasks.pop(run_id, None)

    async def _drive(self, run_id: str, *, resume: bool, resume_value: Any = None) -> None:
        run = await self.require_run(run_id)
        if run_id in self._cancelled:
            return
        await self._transition(run_id, RunStatus.RUNNING)
        state = self._states.get(run_id)
        if state is None or not resume:
            state = await self._initial_state(run)
        else:
            state = _reset_for_resume(state)
        services = self._services_factory(self, run, state)
        graph_state = await self._invoke(services, state, run, resume=resume, resume_value=resume_value)
        self._states[run_id] = self._with_grants(run_id, graph_state)
        await self._settle(run_id, self._states[run_id])

    def _with_grants(self, run_id: str, state: AgentState) -> AgentState:
        """把运行层登记的授权范围并回 state（批准可能发生在节点之外）。"""

        merged = dict(state)
        existing = list(merged.get("approval_grants") or [])
        merged["approval_grants"] = [
            *existing,
            *[
                item
                for item in self.grants_for_state(run_id)
                if item not in existing
            ],
        ]
        return merged  # type: ignore[return-value]

    async def _invoke(
        self,
        services: GraphServices,
        state: AgentState,
        run: Any,
        *,
        resume: bool,
        resume_value: Any,
    ) -> AgentState:
        """按配置走 LangGraph 或离线驱动，两条路径的收尾语义一致。

        图**每次调用按当前 run 现编译**：节点闭包里的 ``services`` 绑定本次 run
        （emit / publish / 工具上下文都需要真实 run），因此不能按工作区缓存编译产物
        后跨 run 复用。检查点仍按工作区缓存（``checkpoints.db`` 是每区一个文件）。
        """

        if not self.use_langgraph:
            return await agent_graph.run_pipeline(
                services, state, concurrency=self.research_concurrency
            )
        checkpointer = await self._checkpointer_for(run.workspace_id)
        research_graph = agent_graph.build_research_graph(services, checkpointer=checkpointer)
        if research_graph is not None:
            services.research_graph = research_graph
        graph = agent_graph.build_main_graph(services, checkpointer=checkpointer)
        if graph is None:
            return await agent_graph.run_pipeline(
                services, state, concurrency=self.research_concurrency
            )
        config = {
            "configurable": {"thread_id": main_thread_id(run.id)},
            "recursion_limit": services.recursion_limit,
        }
        try:
            if resume and resume_value is not None:
                from langgraph.types import Command

                produced = await graph.ainvoke(Command(resume=resume_value), config=config)
            else:
                produced = await graph.ainvoke(state, config=config)
        except BaseException as exc:  # GraphInterrupt 走挂起路径，其余上抛
            if type(exc).__name__ not in {"GraphInterrupt", "GraphBubbleUp"}:
                raise
            snapshot = await graph.aget_state(config)
            return dict(snapshot.values or {})
        return dict(produced or {})

    # ---- 检查点（每工作区独立 checkpoints.db，PRD 12.2） ----

    async def _checkpointer_for(self, workspace_id: str) -> Any:
        """每工作区 ``AsyncSqliteSaver``（PRD 12.2 / 12.4：重启后继续依赖检查点）。

        **必须 async**：``AsyncSqliteSaver.from_conn_string()`` 返回的是**异步**上下文管理器
        （``langgraph-checkpoint-sqlite`` 3.x），早先用同步 ``AsyncExitStack.enter_context``
        包装会抛 ``TypeError``，而那里只记了一条 warning ⇒ **检查点从未真正建立**：
        主图与研究子图都在无检查点的情况下编译，「每任务独立状态 + 重启继续」实际不成立。
        """

        if workspace_id in self._checkpointers:
            return self._checkpointers[workspace_id]
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError:  # pragma: no cover - 依赖缺失即退离线驱动
            return None
        context = self._context(workspace_id)
        if context is None:
            return None
        path = Path(context.checkpoints_db)
        path.parent.mkdir(parents=True, exist_ok=True)
        stack = self._checkpoint_stack
        if stack is None:  # pragma: no cover - 正常路径由 aclose 之外的首次调用建立
            return None
        try:
            saver = await stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(str(path))
            )
        except Exception as exc:  # noqa: BLE001 - 检查点不可用不应让 run 无法执行
            logger.warning("检查点不可用：error=%s", type(exc).__name__)
            return None
        self._checkpointers[workspace_id] = saver
        return saver

    def enable_checkpoints(self) -> None:
        """开启每工作区检查点（lifespan 调用；失败即退离线驱动）。"""

        if self._checkpoint_stack is None:
            self._checkpoint_stack = contextlib.AsyncExitStack()

    # =================================================================
    # 收尾：state → run 状态
    # =================================================================

    async def _initial_state(self, run: Any) -> AgentState:
        conversation = await self.store.get_conversation(run.conversation_id)
        history: list[dict[str, str]] = []
        if conversation is not None:
            messages = await self.store.list_messages(run.conversation_id, limit=20)
            history = [
                {"role": item.role, "content": item.content}
                for item in messages
                if item.id != run.input_message_id
            ]
        context = self._context(run.workspace_id)
        question = ""
        if run.input_message_id:
            message = await self.store.get_message(run.input_message_id)
            question = message.content if message is not None else ""
        return new_agent_state(
            run_id=run.id,
            workspace_id=run.workspace_id,
            conversation_id=run.conversation_id,
            question=question,
            history=history,
            knowledge_revision=int(getattr(context, "knowledge_revision", 0) or 0),
            resume_from="resume" if _is_resumable(run.status) else "",
        )

    async def _settle(self, run_id: str, state: AgentState) -> None:
        """把 state 映射成 run 的终态/暂停态（两条驱动共用）。"""

        if run_id in self._cancelled:
            return
        if state.get("error"):
            await self._fail_from_state(run_id, state)
            return
        pending = dict(state.get("pending_approval") or {})
        if pending:
            async with self.db():
                request = await self.store.get_approval(str(pending.get("approval_id") or ""))
            if request is None or request.status is ApprovalStatus.PENDING:
                await self._transition(run_id, RunStatus.WAITING_APPROVAL)
                return
        if str(state.get("phase") or "") == "clarify":
            clarification = dict(state.get("clarification") or {})
            if not clarification or str(clarification.get("status")) == "pending":
                await self._transition(run_id, RunStatus.WAITING_INPUT)
                return
        if state.get("answer_id"):
            # 答案事务已由 publish 完成（commit_answer 内已写终态与 answer.meta/delta*/
            # run.completed 事件）。这里**不能再发 run.status**：run.completed 必须是
            # 该订阅的收尾帧，否则前端会先关订阅再收到一条孤儿事件（PRD 13.3）。
            return
        await self._fail(run_id, RuntimeError("运行结束时没有产出答案"), node="runtime")

    async def _fail_from_state(self, run_id: str, state: AgentState) -> None:
        error = AgentError.model_validate(state.get("error") or {})
        await self.publish_event(
            run_id, EventType.RUN_FAILED, {"error": error.model_dump(mode="json")}
        )
        await self._transition(
            run_id,
            RunStatus.FAILED,
            resumable=error.resumable,
            error_code=error.error_code,
            via_running=True,
        )

    async def _fail(self, run_id: str, exc: BaseException, *, node: str) -> None:
        if run_id in self._cancelled:
            return
        error = AgentError(
            error_code=str(getattr(exc, "code", "") or type(exc).__name__),
            message=f"{node} 阶段失败：{type(exc).__name__}",
            node=node,
            resumable=True,
            required_action="POST /api/workspaces/{workspace_id}/runs/{run_id}/resume 继续",
            created_at=utc_now_iso(),
        )
        await self.publish_event(
            run_id, EventType.RUN_FAILED, {"error": error.model_dump(mode="json")}
        )
        await self._transition(
            run_id, RunStatus.FAILED, resumable=True, error_code=error.error_code, via_running=True
        )

    async def _transition(
        self,
        run_id: str,
        target: RunStatus,
        *,
        resumable: bool | None = None,
        error_code: str | None = None,
        stop_reason: str | None = None,
        via_running: bool = False,
    ) -> Any:
        """按 PRD 12.3 的迁移表迁移；非法迁移直接拒绝（绝不静默改写状态）。

        ``via_running`` 用于失败收敛：``waiting_approval`` 等状态在迁移图里不能直接到
        ``failed``，先回 ``running`` 再落 ``failed`` —— 两步都必须是合法迁移。
        """

        async with self.db():
            run = await self.store.get_run(run_id)
        if run is None:
            raise RunNotFoundError(f"run 不存在：{run_id}")
        if run.status is target:
            return run
        if not can_transition_run(run.status, target, resumable=bool(run.resumable)):
            if via_running and can_transition_run(
                run.status, RunStatus.RUNNING, resumable=bool(run.resumable)
            ):
                await self.store.update_run_status(run_id, RunStatus.RUNNING)
            else:
                raise RunNotRunnableError(
                    f"run {run_id} 不能从 {run.status.value} 迁移到 {target.value}（PRD 12.3）"
                )
        async with self.db():
            return await self.store.update_run_status(
                run_id,
                target,
            resumable=resumable,
            error_code=error_code,
            stop_reason=stop_reason,
        )

    # =================================================================
    # 取消 / 恢复
    # =================================================================

    async def cancel(
        self, run_id: str, *, reason: str = "user_stopped", resumable: bool = True
    ) -> Any:
        """取消：终止主任务与活动子任务，**等模型请求退出后**才记完成（PRD 12.4）。

        ``resumable=False`` 用于「会话移动终止」（PRD 5.4）：该 run 不再提供继续入口
        —— ``contracts.can_transition_run`` 的 ``cancelled → running`` 要求
        ``resumable=true``，因此置 false 即从状态机层面关闭恢复与批准/澄清提交。
        **安全网**：``reason`` 为 :data:`STOP_REASON_CONVERSATION_MOVED` 时一律
        ``resumable=False`` —— 调用方漏传也不会出现「标记为会话移动却还能继续」的状态。
        """

        if reason == STOP_REASON_CONVERSATION_MOVED:
            resumable = False
        run = await self.require_run(run_id)
        if run.status not in CANCELLABLE_RUN_STATUSES:
            # 已结束的 run（completed / cancelled / failed …）取消是幂等空操作：
            # 终态不允许被改写（PRD 12.3）。
            return run
        await self._transition(run_id, RunStatus.CANCELLING)
        self._cancelled.add(run_id)
        task = self._tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        async with self.db():
            await self.store.supersede_pending_approvals(run_id, reason=reason)
            await self.store.supersede_pending_inputs(run_id)
        await self.publish_event(
            run_id,
            EventType.RUN_CANCELLED,
            {"stop_reason": reason, "resumable": resumable},
        )
        return await self._transition(
            run_id, RunStatus.CANCELLED, resumable=resumable, stop_reason=reason
        )

    async def stop_conversation_runs(
        self, conversation_id: str, *, reason: str = "conversation_moved"
    ) -> list[Any]:
        """停止某会话的全部活动 run 并**等待活动请求退出**（PRD 5.4 移动的前置动作）。

        顺序契约（移动必须先满足）：停止主任务及其子任务 → 等活动请求退出 → 才允许
        改会话归属。``cancel`` 会 ``await`` 掉驱动任务，因此本方法返回即代表「已退出」；
        返回前再用 :meth:`ensure_conversation_idle` 复核一次，避免把顺序保证留给调用方。
        """

        async with self.db():
            active = await self.store.list_active_runs(conversation_id)
        stopped: list[Any] = []
        for run in active:
            stopped.append(await self.cancel(run.id, reason=reason, resumable=False))
        await self.ensure_conversation_idle(conversation_id)
        return stopped

    async def ensure_conversation_idle(self, conversation_id: str) -> None:
        """确认会话没有活动 run、也没有仍在跑的驱动任务（移动改归属前的最后一道闸）。"""

        async with self.db():
            active = await self.store.list_active_runs(conversation_id)
        running = [run.id for run in active if self.is_running(run.id)]
        if active or running:
            raise RunManagerError(
                "会话仍有活动任务，不能移动："
                f"active={[run.id for run in active]} running={running}（PRD 5.4）"
            )

    async def resume(self, run_id: str) -> Any:
        """用户「继续」：从记录位置重新驱动（PRD 12.3 / 12.4）。"""

        run = await self.require_run(run_id)
        if run.status not in {
            RunStatus.FAILED,
            RunStatus.INTERRUPTED,
            RunStatus.CANCELLED,
        }:
            raise RunNotRunnableError(f"run {run_id} 当前状态 {run.status.value} 不可继续")
        if run.status is RunStatus.CANCELLED and not run.resumable:
            raise RunNotRunnableError("该 run 因会话移动终止，不可继续（PRD 5.4）")
        async with self.db():
            active = await self.store.list_active_runs(run.conversation_id)
        if any(item.id != run_id for item in active):
            raise ConversationBusyError("会话已有其他活动任务，不能继续该 run")
        self._cancelled.discard(run_id)
        await self.start(run_id, resume=True)
        return await self.require_run(run_id)

    # =================================================================
    # 批准 / 澄清
    # =================================================================

    async def submit_approval(
        self,
        approval_id: str,
        *,
        decision_kind: str,
        edited_parameters: Mapping[str, Any] | None = None,
        note: str = "",
    ) -> ApprovalRequest:
        """提交批准决定（PRD 10.2）：幂等重复、冲突 409 由 app_store 保证。"""

        from app.agent.approval import resolve_approval
        from app.agent.contracts import ApprovalDecision, ApprovalDecisionKind

        async with self.db():
            request = await self.store.get_approval(approval_id)
        if request is None:
            raise RunNotFoundError(f"批准不存在：{approval_id}")
        run = await self.require_run(request.run_id)
        decision = ApprovalDecision(
            kind=ApprovalDecisionKind(decision_kind),
            edited_parameters=None
            if edited_parameters is None
            else dict(edited_parameters),
            decided_at=utc_now_iso(),
            note=note,
        )
        if run.status is RunStatus.CANCELLED and not run.resumable:
            # 因会话移动终止的 run：批准卡片已 superseded，运行时也直接拒绝
            # （PRD 5.4「拒绝提交批准」），不把判定只留给路由。
            raise RunNotRunnableError("该 run 因会话移动终止，不能提交批准（PRD 5.4）")
        question_scope = await self._question_scope(run)
        resolved = await resolve_approval(
            self.store,
            approval_id=approval_id,
            decision=decision,
            run_id=run.id,
            question_scope=question_scope,
            tool_name=request.display.tool_name,
            context_epoch=None,
        )
        await self.publish_event(
            run.id,
            EventType.APPROVAL_RESOLVED,
            {
                "approval_id": approval_id,
                "status": resolved.status.value,
                "decision_kind": decision.kind.value,
            },
            object_id=approval_id,
            version=1,
        )
        if run.status is RunStatus.WAITING_APPROVAL:
            state = self._states.get(run.id)
            if state is not None:
                state = dict(state)  # type: ignore[assignment]
                state["approval_decision"] = decision.model_dump(mode="json")
                self._states[run.id] = state
            await self.start(run.id, resume=True)
        return resolved

    async def submit_input(
        self, input_request_id: str, *, answer: str, question_version: int | None = None
    ) -> InputRequest:
        """提交澄清回答（PRD 13.2）：版本不符/重复不同回答 → 409。"""

        async with self.db():
            request = await self.store.get_input_request(input_request_id)
        if request is None:
            raise RunNotFoundError(f"澄清请求不存在：{input_request_id}")
        run = await self.require_run(request.run_id)
        if run.status is RunStatus.CANCELLED and not run.resumable:
            raise RunNotRunnableError("该 run 因会话移动终止，不能提交澄清（PRD 5.4）")
        resolved = await self.store.resolve_input(
            input_request_id, answer=answer, question_version=question_version
        )
        await self.publish_event(
            run.id,
            EventType.INPUT_RESOLVED,
            {"input_request_id": input_request_id, "answer": answer},
            object_id=input_request_id,
            version=1,
        )
        if run.status is RunStatus.WAITING_INPUT:
            state = self._states.get(run.id)
            if state is not None:
                state = dict(state)  # type: ignore[assignment]
                state["clarification"] = resolved.model_dump(mode="json")
                self._states[run.id] = state
            await self.start(run.id, resume=True)
        return resolved

    async def _question_scope(self, run: Any) -> str:
        if run.input_message_id:
            message = await self.store.get_message(run.input_message_id)
            if message is not None:
                return message.content
        return ""

    # =================================================================
    # 事件与订阅
    # =================================================================

    async def publish_event(
        self,
        run_id: str,
        event_type: EventType | str,
        payload: Mapping[str, Any],
        *,
        task_id: str | None = None,
        execution_id: str = "",
        object_id: str | None = None,
        version: int | None = None,
    ) -> agent_events.StoredEvent:
        """**先保存后推送**（PRD 13.3）：写库成功再唤醒订阅者。"""

        async with self.db():
            event = await self.store.append_event(
                run_id=run_id,
                type=event_type,
                payload=payload,
                execution_id=execution_id,
                task_id=task_id,
                object_id=object_id,
                version=version,
            )
        for subscription in list(self._subscriptions.get(run_id, ())):
            subscription.queue.put_nowait(event)
        return event

    async def read_events(self, run_id: str, *, after_seq: int = 0) -> list[Any]:
        """按 ``event_seq`` 补读（``Last-Event-ID``）；不合法事件按协议错误暴露。"""

        async with self.db():
            rows = await self.store.read_events(run_id, after_seq=after_seq)
        events: list[Any] = []
        for row in rows:
            events.append(agent_events.parse_event(row))
        return events

    async def stream_events(
        self, run_id: str, *, after_seq: int = 0, timeout: float | None = None
    ) -> AsyncIterator[str]:
        """SSE 帧流：先补读历史事件，再实时推送，收到结束事件即关闭订阅（PRD 13.3）。"""

        subscription = _Subscription()
        self._subscriptions.setdefault(run_id, []).append(subscription)
        last_seq = int(after_seq)
        try:
            try:
                for event in await self.read_events(run_id, after_seq=last_seq):
                    last_seq = max(last_seq, event.event_seq)
                    yield sse_frame(event)
                    if event.type in SUBSCRIPTION_CLOSING_EVENTS:
                        return
            except agent_events.EventProtocolError as exc:
                yield sse_error_frame(
                    error_code="event_protocol_error", message=str(exc) or "事件协议错误"
                )
                return
            run = await self.store.get_run(run_id)
            if run is not None and run.status in {
                RunStatus.COMPLETED,
                RunStatus.COMPLETED_WITH_GAPS,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }:
                return
            while True:
                try:
                    event = (
                        await asyncio.wait_for(subscription.queue.get(), timeout)
                        if timeout is not None
                        else await subscription.queue.get()
                    )
                except TimeoutError:
                    return
                if event.event_seq <= last_seq:
                    continue
                last_seq = event.event_seq
                yield sse_frame(event)
                if event.type in SUBSCRIPTION_CLOSING_EVENTS:
                    return
        finally:
            listeners = self._subscriptions.get(run_id, [])
            with contextlib.suppress(ValueError):
                listeners.remove(subscription)
            if not listeners:
                self._subscriptions.pop(run_id, None)

    def subscriber_count(self, run_id: str) -> int:
        return len(self._subscriptions.get(run_id, ()))

    # =================================================================
    # 内部装配
    # =================================================================

    def _context(self, workspace_id: str) -> Any:
        if self.context_factory is None:
            return None
        try:
            return self.context_factory.resolve(workspace_id)
        except Exception as exc:  # noqa: BLE001 - 上下文解析失败按无句柄处理
            logger.warning("工作区上下文解析失败：error=%s", type(exc).__name__)
            return None

    def _services_factory_default(self, run: Any, state: AgentState) -> GraphServices:
        return self._services_factory(self, run, state)

    async def aclose(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()
        if self._checkpoint_stack is not None:
            with contextlib.suppress(Exception):
                await self._checkpoint_stack.aclose()
            self._checkpoint_stack = None
        # 已筛选来源记录只服务于当前进程内的 run；关闭时不留下任何残留。
        self._rerank_sources.clear()


@dataclass(frozen=True)
class _WorkspaceStub:
    """只带 ``workspace_id`` 的 run 替身（仅为编译期构建 services 用）。"""

    workspace_id: str


def _is_resumable(status: RunStatus) -> bool:
    return status in {
        RunStatus.FAILED,
        RunStatus.INTERRUPTED,
        RunStatus.CANCELLED,
    }


def _reset_for_resume(state: AgentState) -> AgentState:
    """「继续」前的 state 复位：清错误、保留已完成结果与授权（PRD 12.4）。

    保留 ``task_results`` / ``completed_task_ids`` 是有意的：成功子任务不重复执行；
    清掉 ``error`` / ``failed_task_ids`` 是为了让研究节点重新评估失败任务，而不是
    一进去就被上一次的错误直接判定为失败。
    """

    resumed = dict(state)
    resumed["error"] = {}
    resumed["failed_task_ids"] = []
    resumed["approval_decision"] = {}
    if str(resumed.get("phase") or "") in {"done", "approval", "clarify"}:
        resumed["phase"] = "research"
    return resumed  # type: ignore[return-value]


# =====================================================================
# 默认服务装配（真实链路）
# =====================================================================


def default_services(manager: RunManager, run: Any, state: AgentState) -> GraphServices:
    """把上游模块的真实入口接进图（routing / planner / research / verification / 图）。

    这里刻意**只做接线**：所有语义都在各自模块里，运行层不重复实现规则。
    """

    from app.agent import planner as agent_planner
    from app.agent import research as agent_research
    from app.agent import routing as agent_routing
    from app.agent import verification as agent_verification
    from app.agent.tools import build_default_registry

    registry = manager.registry or build_default_registry()
    workspace_id = str(run.workspace_id)
    context = manager._context(workspace_id)

    async def emit(
        event_type: EventType, payload: Mapping[str, Any], *, task_id: str | None = None
    ) -> None:
        await manager.publish_event(run.id, event_type, payload, task_id=task_id)

    async def memory_context(current: Mapping[str, Any]) -> str:
        del current
        return _agent_memory_snapshot(manager, run)

    async def route(current: Mapping[str, Any]) -> RouteDecision:
        catalog = _catalog_for(manager, context)
        routing_input = agent_routing.RoutingInput(
            question=str(current.get("question") or ""),
            workspace_id=workspace_id,
            knowledge_revision=int(current.get("knowledge_revision") or 0),
            memory_snapshot=str(current.get("memory_context") or ""),
            recent_history=[
                agent_routing.RoutingHistoryTurn(role=turn["role"], content=turn["content"])
                for turn in (current.get("history") or [])
                if turn.get("role") in {"user", "assistant"}
            ],
            file_count=_vector_count(manager, context, "count_files"),
            chunk_count=_vector_count(manager, context, "count"),
        )
        agent = agent_routing.IntentAgent(catalog=catalog)
        return agent_routing.route_question(routing_input, agent=agent, catalog=catalog)

    async def plan(current: Mapping[str, Any]) -> Any:
        decision = RouteDecision.model_validate(current.get("decision") or {})
        planner_agent = agent_planner.PlannerAgent(
            workspace_id=workspace_id,
            knowledge_revision=int(current.get("knowledge_revision") or 0),
        )
        return planner_agent.plan(str(current.get("question") or ""), decision)

    async def research(
        current: Mapping[str, Any], task_dump: Mapping[str, Any], request_dump: Mapping[str, Any]
    ) -> Any:
        del request_dump
        task = ResearchTask.model_validate(task_dump)
        decision_raw = current.get("decision")
        decision = (
            RouteDecision.model_validate(decision_raw) if decision_raw else None
        )
        request = agent_research.ResearchRequest(
            task=task,
            original_question=str(current.get("question") or ""),
            resolved_question=str(current.get("question") or ""),
            requirement_texts=dict(current.get("requirement_texts") or {}),
            decision=decision,
            date_range=_date_range(decision),
        )
        tool_context = await tool_context_for(manager, run, current, task.task_id, task.agent_kind)
        gate = ApprovalGate(
            manager=manager,
            registry=registry,
            run=run,
            state=current,
            question_scope=str(current.get("question") or ""),
            redactor=tool_context.redactor,
        )
        semaphore = manager._semaphores.setdefault(
            workspace_id, asyncio.Semaphore(manager.research_concurrency)
        )
        # B 项（t31）：General Agent 的模型适配器在此**按当前设置**装配；
        # 未配置 → None → ``GeneralResearcher`` 给出 not_configured 的结构化失败。
        # 只在 general 任务上构造，RAG/WEB 任务不为此付出任何构造成本。
        general_model = (
            general_model_for_chat() if task.agent_kind is AgentKind.GENERAL else None
        )
        async with semaphore:
            return await agent_research.research_task(
                request,
                tool_context,
                registry=gate,
                context_top_k=DEFAULT_CONTEXT_TOP_K,
                general_model=general_model,
            )

    def synthesize(
        current: Mapping[str, Any],
        results: Sequence[Mapping[str, Any]],
        requirements: Mapping[str, str],
    ) -> Any:
        parsed = [agent_research.ResearchResult.model_validate(item) for item in results]
        candidate = agent_verification.synthesize_answer(
            str(current.get("question") or ""),
            parsed,
            requirements=dict(requirements),
            workspace_id=workspace_id,
            run_id=run.id,
            # ④ PRD 8.1/8.3：候选答案由模型撰写；未配置（None）时 synthesize_answer
            # 自己回落确定性 render_candidate（模型拒绝带占位符时同样回落，见该工厂）。
            model=synthesis_model_for_chat(),
        )
        # PRD 8.3：引用编号在**最终候选答案形成时**分配（把 {{ev-…}} 占位符换成 [S1]/[W1]…）。
        #
        # PRD 8.1 末句：General Agent 的解释与建议（``general_knowledge`` / ``proposal``）
        # **不得生成文档引用编号**。``render_candidate`` 会给每条结论都放占位符，因此这里
        # 先把**不可引用来源**的占位符去掉，再交给 ``assign_citation_ids`` —— 否则它会按
        # 「该来源不得编号」直接抛 ``CitationError``，把一次本可正常发布的 general 运行判死
        # （t31 实测：配好 provider 后 general 问题在 synthesize 阶段失败）。
        from app.agent.contracts import citation_prefix

        content = candidate.content
        for item in candidate.evidence:
            if citation_prefix(item.source_type) is None:
                content = content.replace(f"{{{{{item.evidence_id}}}}}", "")
        assignment = agent_verification.assign_citation_ids(
            content,
            candidate.evidence,
            existing=candidate.citations,
            # 集成 t5：本 run 入选来源的相关性记录（检索工具经 rerank_outcome 接缝登记），
            # 按 evidence.source_ref 透传到引用条目的追加字段 —— 无记录时保持默认值，
            # 绝不补造排序分数（PRD §8.3）。
            relevance_by_source=manager.rerank_sources(run.id),
        )
        if assignment.citations or assignment.content != candidate.content:
            candidate = candidate.model_copy(
                update={"content": assignment.content, "citations": assignment.citations}
            )
        return candidate

    async def verify(current: Mapping[str, Any]) -> Any:
        candidate = agent_verification.CandidateAnswer.model_validate(
            current.get("candidate") or {}
        )
        return await agent_verification.verify_answer(
            candidate,
            requirements=dict(current.get("requirement_texts") or {}),
            # ③ PRD 8.2：模型判断语义支持（未配置时为 None ⇒ 只有程序侧八项检查）
            model_verifier=model_verifier_for_chat(),
        )

    async def publish(current: Mapping[str, Any]) -> Any:
        candidate = agent_verification.CandidateAnswer.model_validate(
            current.get("candidate") or {}
        )
        verification = agent_verification.VerificationResult.model_validate(
            current.get("verification") or {}
        )
        requirements = dict(current.get("requirement_texts") or {})
        try:
            artifact = agent_verification.publish_answer(
                candidate, verification, requirements=requirements
            )
        except agent_verification.PublicationError:
            return None
        await manager.store.commit_answer(
            artifact=artifact,
            deltas=[artifact.content],
            gaps=[issue.required_action for issue in verification.issues if issue.required_action],
        )
        return artifact

    async def save_task(
        current: Mapping[str, Any],
        task: ResearchTask,
        status: TaskStatus,
        thread_id: str,
        *,
        result_ref: str | None = None,
        error: BaseException | None = None,
    ) -> None:
        del error
        async with manager.db():
            await manager.store.create_task(
                run_id=run.id,
                task_id=task.task_id,
            agent_kind=task.agent_kind,
            question=task.question,
            parent_task_id=task.parent_task_id,
            requirement_ids=task.requirement_ids,
            depends_on=task.depends_on,
            expected_evidence=task.expected_evidence,
            thread_id=thread_id,
        )
        await manager.store.update_task_status(
            run.id, task.task_id, status, result_ref=result_ref, thread_id=thread_id
        )

    async def load_approval(approval_id: str) -> Any:
        return await manager.store.get_approval(approval_id)

    async def request_input(
        current: Mapping[str, Any], question: str, options: Sequence[str], reason: str
    ) -> InputRequest:
        request = InputRequest(
            input_request_id=f"in-{run.id[:24]}-{len(current.get('outcomes') or [])}",
            run_id=run.id,
            task_id=None,
            question=question,
            options=list(options),
            reason=reason,
        )
        stored = await manager.store.request_input(request)
        await emit(
            EventType.INPUT_REQUESTED, {"request": stored.model_dump(mode="json")}
        )
        return stored

    return GraphServices(
        emit=emit,
        memory_context=memory_context,
        route=route,
        plan=plan,
        research=research,
        research_graph=None,  # 由 :meth:`_invoke` 按当前 run 现编译后写入（见那里的说明）
        synthesize=synthesize,
        verify=verify,
        publish=publish,
        save_task=save_task,
        load_approval=load_approval,
        request_input=request_input,
        tool_context=tool_context_for,
    )


async def tool_context_for(
    manager: RunManager,
    run: Any,
    state: Mapping[str, Any],
    task_id: str | None,
    agent_kind: AgentKind | None = None,
) -> Any:
    """组装一次工具执行的上下文本（PRD 5.2 / 10.4）。"""

    from app.agent.tools import ToolContext

    workspace_id = str(run.workspace_id)
    context = manager._context(workspace_id)
    extras: dict[str, Any] = {}
    if manager.tool_services is not None and context is not None:
        extras = _tool_context_kwargs(dict(manager.tool_services(context)))
    return ToolContext(
        workspace=context,
        app_store=manager.store,
        run_id=run.id,
        task_id=task_id,
        agent_kind=agent_kind,
        redactor=NameRedactor(manager.private_names),
        # 筛选进度 → v2 事件（`tool.started` / `tool.completed`，`tool_name=rerank`）。
        # 契约里这两个类型本就存在（PRD 13.3 的 20 类），此前无人发布；这里按事件对象的
        # 既有字段发布，不新增事件类型、不改 payload 模型（冻结契约只读）。
        rerank_progress=_rerank_progress_sink(manager, run),
        # 筛选**结果**（纯追加接缝）：真实计数进同一个 `tool.completed`，入选来源的相关性
        # 记录留给 `answer.meta.citations`（运行层按 source_ref 转交引用编号处）。
        rerank_outcome=_rerank_outcome_sink(manager, run),
        **extras,
    )


def _rerank_outcome_sink(
    manager: RunManager, run: Any
) -> Callable[[Mapping[str, Any]], Awaitable[None]]:
    """把工具层上报的筛选**结果**转成 v2 事件 + 记下来源的相关性记录。

    * ``tool.completed`` 填**真实** ``result_count``（入选来源数，0 ⇒ 筛选后 0 条）与
      ``source_count``（候选数）—— 前端据此区分 ``filtered`` / ``empty``，不再把 0 当成
      「未统计」；
    * 来源按 ``source_ref`` 记到本 run（:meth:`RunManager.register_rerank_sources`），在合成
      阶段交给 :func:`app.agent.verification.assign_citation_ids`，成为
      ``answer.meta.citations`` 的追加字段（PRD §8.3 / RAG-relevance `:304`）。

    **``status="disabled"`` 是特例（t9）**：未启用筛选时**不发布任何事件**，只登记来源 ——
    前端的「已筛选 / 空 / 未启用」三态判定**依赖事件流**（无 rerank 事件 ⇒ ``disabled``），
    因此这里绝不能补发一个 ``tool.completed``，否则会把未筛选的一轮误报成「筛选过」；
    登记来源是为了让引用仍能带上**真实向量相似度**（PRD RAG-relevance `:52`）。

    只用**已声明事件的现有字段**：不新增事件类型、不改 payload 模型。
    """

    from app.retrieval.rerank_pipeline import RERANK_PROGRESS_TOOL_NAME, STATUS_DISABLED

    run_id = str(run.id)

    async def report(payload: Mapping[str, Any]) -> None:
        sources = [
            dict(item) for item in (payload.get("sources") or []) if isinstance(item, Mapping)
        ]
        manager.register_rerank_sources(run_id, sources)
        if str(payload.get("status") or "") == STATUS_DISABLED:
            # 未启用筛选：只登记来源（让引用带上真实向量分数），**不发布筛选事件**。
            return
        await manager.publish_event(
            run_id,
            EventType.TOOL_COMPLETED,
            {
                "tool_name": RERANK_PROGRESS_TOOL_NAME,
                "status": "succeeded",
                "result_count": max(0, int(payload.get("result_count") or 0)),
                "source_count": max(0, int(payload.get("source_count") or 0)),
            },
        )

    return report


def _rerank_progress_sink(
    manager: RunManager, run: Any
) -> Callable[[str, str], Awaitable[None]]:
    """把工具的筛选进度上报转成 v2 事件（``running`` / ``done`` / ``error``）。

    只做映射，不判断筛选语义：状态由工具层按 ``app.retrieval.rerank_pipeline`` 的
    四类状态给出。发布异常**不吞** —— 事件是用户可见进度，写不进去就是运行层故障。
    """

    from app.retrieval.rerank_pipeline import RERANK_PROGRESS_TOOL_NAME

    run_id = str(run.id)

    async def report(state: str, error_code: str = "") -> None:
        if state == "running":
            await manager.publish_event(
                run_id,
                EventType.TOOL_STARTED,
                {"tool_name": RERANK_PROGRESS_TOOL_NAME},
            )
            return
        await manager.publish_event(
            run_id,
            EventType.TOOL_COMPLETED,
            {
                "tool_name": RERANK_PROGRESS_TOOL_NAME,
                "status": "succeeded" if state == "done" else "failed",
                "error_code": "" if state == "done" else str(error_code or ""),
            },
        )

    return report


def _tool_context_kwargs(extras: Mapping[str, Any]) -> dict[str, Any]:
    """只保留 ``ToolContext`` 的字段；``_catalog`` / ``_memory`` 是运行层内部键。

    装配层会在同一个 dict 里同时给出「工具要用的句柄」与「运行层自己要用的目录 /
    记忆」，这里做一次显式过滤，避免把内部键塞进 dataclass 构造而报错。
    """

    from dataclasses import fields

    from app.agent.tools import ToolContext

    allowed = {item.name for item in fields(ToolContext)}
    return {key: value for key, value in extras.items() if key in allowed}


def _date_range(decision: RouteDecision | None) -> tuple[str, str] | None:
    if decision is None:
        return None
    try:
        from app.agent.research import resolve_date_range

        return resolve_date_range(
            decision.freshness, now=datetime.now(UTC), timezone="Asia/Shanghai"
        )
    except Exception:  # noqa: BLE001 - 时效范围只是优化，失败不影响研究
        return None


def _vector_count(manager: RunManager, context: Any, method: str) -> int:
    """向量库统计（路由 R9 需要「知识库是否为空」）。

    句柄来自注入的 ``tool_services``；缺失时按 0 处理（路由会说明资料情况），
    绝不为了拿计数去新建真实向量库。
    """

    if manager.tool_services is None or context is None:
        return 0
    try:
        extras = dict(manager.tool_services(context))
    except Exception:  # noqa: BLE001
        return 0
    store = extras.get("vector_store")
    counter = getattr(store, method, None)
    if counter is None:
        return 0
    try:
        return int(counter() or 0)
    except Exception:  # noqa: BLE001
        return 0


def _catalog_for(manager: RunManager, context: Any) -> Any:
    if manager.tool_services is None or context is None:
        return None
    try:
        return dict(manager.tool_services(context)).get("catalog")
    except Exception:  # noqa: BLE001
        return None


def _agent_memory_snapshot(manager: RunManager, run: Any) -> str:
    """工作区记忆固定快照（PRD 9.3 输入顺序第 3 位）。

    记忆开关以 ``WorkspaceMemory.load().enabled`` 为准，**不用**
    ``WorkspaceContextView.memory_enabled``（T01 未接线，恒为 True）。
    """

    if manager.tool_services is None or run is None:
        return ""
    context = manager._context(str(run.workspace_id))
    if context is None:
        return ""
    try:
        extras = dict(manager.tool_services(context))
    except Exception:  # noqa: BLE001
        return ""
    memory = extras.get("memory")
    if memory is None:
        return ""
    try:
        loaded = memory.load()
        if not getattr(loaded, "enabled", False):
            return ""
        snapshot = getattr(memory, "agent_context", None)
        if callable(snapshot):
            return str(snapshot(run.conversation_id) or "")
    except Exception as exc:  # noqa: BLE001 - 快照失败不阻塞问答
        logger.warning("记忆快照构建失败：error=%s", type(exc).__name__)
    return ""


# =====================================================================
# 全局单例（lifespan 注册，路由读取）
# =====================================================================

_run_manager: RunManager | None = None


def register_run_manager(manager: RunManager | None) -> None:
    """注册 / 复位全局 RunManager（lifespan 用；传 None 表示关闭）。"""

    global _run_manager
    _run_manager = manager


def get_run_manager() -> RunManager | None:
    """返回全局 RunManager；未注册时为 None（路由据此 503）。"""

    return _run_manager
