"""Planner Agent 测试（PRD 6.4）。Owner: T07。

覆盖：要求提取与保留原文、每项要求映射到任务（A09）、来源拆分、依赖顺序与并行、
依赖环 / 缺失任务 ID / 无来源完成声明被拒、重复任务复用、并发预算（PRD 18.1）、
「不设总时长与总轮数上限」。全程纯离线（无模型、无 Chroma、无出网）。
"""

from __future__ import annotations

import asyncio

import pytest

from app.agent.contracts import (
    AgentKind,
    Freshness,
    IntentKind,
    ResearchTask,
    RouteDecision,
    TaskStatus,
    validate_task_graph,
)
from app.agent.planner import (
    LONG_INPUT_MIN_CHARS,
    TASK_SOURCES,
    ClarificationRecord,
    ConcurrencyBudget,
    Plan,
    PlannerAgent,
    PlanSettings,
    PlanValidationError,
    TaskResultCache,
    TaskSource,
    _apply_retrieval_queries,
    _build_tasks,
    build_plan,
    build_web_subtask,
    dependency_order,
    extract_requirements,
    is_long_input,
    ready_tasks,
    requirement_coverage,
    result_key,
    validate_plan,
    validate_requirement_quotes,
)

#: 复杂长问题（A09）：5 项独立要求 + 1 项要求结合通用实践，长度超过长输入阈值。
LONG_QUESTION = (
    "1. 我们内部风控方案的 P99 延迟阈值到底是多少毫秒，请给出原文位置和对应文档版本？\n"
    "2. 这个阈值在 1.2.0 版本的实现里有没有被调整过，调整前后分别取什么值？\n"
    "3. 请结合业界最佳实践给出改进建议，说明每一步的依据、适用范围和可能的副作用？\n"
    "4. 历史讨论里关于该阈值有没有形成过结论性的判断，当时的依据和结论是什么？\n"
    "5. 最新版本的行业标准对这类阈值的取值是怎么规定的，与我们的现状差距在哪里？\n"
)

#: 单句同时包含「内部资料 + 结合通用实践」的要求（PRD 6.3 R6 / A03）。
COMBINE_QUESTION = "结合业界最佳实践，我们的内部风控方案应该怎么设计？"


def _decision(question: str = LONG_QUESTION, **overrides: object) -> RouteDecision:
    base: dict[str, object] = {
        "question": question,
        "resolved_question": question,
        "intent": IntentKind.RESEARCH,
        "needs_knowledge": True,
        "needs_web": False,
        "needs_general": False,
        "needs_history": False,
    }
    base.update(overrides)
    return RouteDecision(**base)


def _task(
    task_id: str,
    *,
    requirement_ids: list[str] | None = None,
    depends_on: list[str] | None = None,
    status: TaskStatus = TaskStatus.QUEUED,
    expected_evidence: str = "本工作区原文",
    agent_kind: AgentKind = AgentKind.RAG,
    result_ref: str | None = None,
) -> ResearchTask:
    return ResearchTask(
        task_id=task_id,
        question=f"问题 {task_id}",
        requirement_ids=["R1"] if requirement_ids is None else requirement_ids,
        agent_kind=agent_kind,
        depends_on=[] if depends_on is None else depends_on,
        expected_evidence=expected_evidence,
        status=status,
        result_ref=result_ref,
    )


def _plan(tasks: list[ResearchTask], requirements=("R1",), **overrides: object) -> Plan:
    base: dict[str, object] = {
        "question": "测试问题",
        "workspace_id": "ws-a",
        "knowledge_revision": 3,
        "requirements": [
            {"requirement_id": rid, "text": f"要求 {rid}", "quote": f"要求 {rid}"}
            for rid in requirements
        ],
        "tasks": tasks,
    }
    base.update(overrides)
    return Plan(**base)


# =====================================================================
# 1. 用户要求提取（PRD 6.4：提取全部要求、保留原文）
# =====================================================================


def test_extract_requirements_keeps_quotes_verbatim_and_orders_them() -> None:
    question = "1. 查一下报销制度的时限？\n2. 差旅标准怎么规定的？\n"

    requirements = extract_requirements(question)

    assert [item.requirement_id for item in requirements] == ["R1", "R2"]
    assert requirements[0].text == "查一下报销制度的时限"
    assert requirements[1].quote.strip() in question
    for item in requirements:
        assert item.quote in question  # quote 是原问题的连续子串
    validate_requirement_quotes(requirements, question)


def test_extract_requirements_rejects_empty_question() -> None:
    with pytest.raises(PlanValidationError):
        extract_requirements("   \n  ")


def test_validate_requirement_quotes_rejects_rewritten_text() -> None:
    requirements = extract_requirements("查一下报销制度")
    rewritten = requirements[0].model_copy(update={"quote": "被改写的原文"})

    with pytest.raises(PlanValidationError, match="原文"):
        validate_requirement_quotes([rewritten], "查一下报销制度")


def test_long_input_detection_uses_prd_threshold() -> None:
    assert is_long_input("短问题") is False
    assert is_long_input("测" * LONG_INPUT_MIN_CHARS) is True


def test_web_subtask_from_rag_gap_links_parent_and_dependency() -> None:
    """PRD 7.2 第 5 步：RAG 缺口 → Planner 落成 Web 子任务（父任务 + 依赖 + 要求）。"""

    parent = _task("t-r1-knowledge")

    subtask = build_web_subtask(parent, "最新版本的行业标准怎么说")

    assert subtask.task_id == "t-r1-knowledge-web1"
    assert subtask.parent_task_id == parent.task_id
    assert subtask.depends_on == [parent.task_id]
    assert subtask.agent_kind is AgentKind.WEB
    assert subtask.requirement_ids == ["R1"]
    assert "网页原文" in subtask.expected_evidence

    plan = _plan([parent, subtask])
    validate_plan(plan)
    assert dependency_order(plan.tasks) == [["t-r1-knowledge"], ["t-r1-knowledge-web1"]]
    assert [task.task_id for task in ready_tasks(plan.tasks, completed=[])] == [
        "t-r1-knowledge"
    ]
    assert [task.task_id for task in ready_tasks(plan.tasks, completed=["t-r1-knowledge"])] == [
        "t-r1-knowledge",
        "t-r1-knowledge-web1",
    ]


def test_web_subtask_requires_parent_requirements() -> None:
    parent = _task("t-r1-knowledge", requirement_ids=[])

    with pytest.raises(PlanValidationError, match="没有关联用户要求"):
        build_web_subtask(parent, "问题")


def test_web_subtask_inherits_requirement_ids_when_given() -> None:
    parent = _task("t-r1-knowledge", requirement_ids=["R1", "R2"])

    subtask = build_web_subtask(parent, "问题", requirement_ids=["R2"], index=2)

    assert subtask.task_id == "t-r1-knowledge-web2"
    assert subtask.requirement_ids == ["R2"]


# =====================================================================
# 2. 计划生成与来源拆分（PRD 6.4 / A09）
# =====================================================================


def test_long_input_plan_covers_every_requirement() -> None:
    """A09：复杂长问题的每项要求都有任务或澄清记录。"""

    decision = _decision(
        needs_web=True,
        needs_general=True,
        needs_history=True,
        freshness=Freshness.CURRENT,
    )

    plan = build_plan(LONG_QUESTION, decision, workspace_id="ws-a", knowledge_revision=7)

    coverage = plan.coverage()
    assert plan.long_input is True
    assert len(plan.requirements) == 5
    assert coverage.uncovered == []
    assert coverage.ratio == 1.0
    for requirement in plan.requirements:
        assert any(
            requirement.requirement_id in task.requirement_ids for task in plan.tasks
        ) or requirement.requirement_id in coverage.clarified
    assert plan.clarification is None
    # 任务字段完整（PRD 6.4 全 9 字段）
    for task in plan.tasks:
        assert task.task_id and task.question and task.expected_evidence
        assert task.agent_kind in {AgentKind.RAG, AgentKind.WEB, AgentKind.GENERAL}
    validate_plan(plan)


def test_tasks_are_split_by_evidence_source() -> None:
    decision = _decision(
        "我们内部风控方案怎么设计？请结合业界最佳实践",
        needs_general=True,
    )

    plan = build_plan(decision.question, decision)

    kinds = {task.agent_kind for task in plan.tasks}
    assert kinds == {AgentKind.RAG, AgentKind.GENERAL}
    knowledge = next(task for task in plan.tasks if task.agent_kind is AgentKind.RAG)
    general = next(task for task in plan.tasks if task.agent_kind is AgentKind.GENERAL)
    assert "原文" in knowledge.expected_evidence
    assert "general_knowledge" in general.expected_evidence


def test_web_and_history_sources_create_dedicated_tasks() -> None:
    decision = _decision(
        "最新版本的标准怎么说？另外回忆一下以前的讨论",
        needs_knowledge=False,
        needs_web=True,
        needs_history=True,
        needs_general=False,
        freshness=Freshness.CURRENT,
    )

    plan = build_plan(decision.question, decision)

    kinds = sorted(task.agent_kind.value for task in plan.tasks)
    assert kinds == ["rag", "rag", "web", "web"]
    web = next(task for task in plan.tasks if task.agent_kind is AgentKind.WEB)
    history = next(
        task for task in plan.tasks if "历史" in task.expected_evidence
    )
    assert "网页原文" in web.expected_evidence
    assert history.agent_kind is AgentKind.RAG


def test_product_intent_uses_product_task() -> None:
    question = "这个项目支持哪些模型？"
    decision = _decision(question, intent=IntentKind.PRODUCT, needs_knowledge=False, needs_general=False)

    plan = build_plan(question, decision)

    assert [task.agent_kind for task in plan.tasks] == [AgentKind.GENERAL]
    assert "产品说明" in plan.tasks[0].expected_evidence


def test_without_decision_text_markers_decide_sources() -> None:
    internal = build_plan("我们的内部实现文档里怎么写的？")
    general = build_plan("为什么这样设计比较好？")

    assert [task.agent_kind for task in internal.tasks] == [AgentKind.RAG]
    assert [task.agent_kind for task in general.tasks] == [AgentKind.GENERAL]


# =====================================================================
# 3. 依赖顺序与并行（PRD 6.4）
# =====================================================================


def test_combine_requirement_orders_general_after_knowledge() -> None:
    """A03：通用解释与内部方案综合时，通用任务在同一要求之后启动。"""

    decision = _decision(COMBINE_QUESTION, needs_general=True)

    plan = build_plan(COMBINE_QUESTION, decision)

    knowledge = next(task for task in plan.tasks if task.agent_kind is AgentKind.RAG)
    general = next(task for task in plan.tasks if task.agent_kind is AgentKind.GENERAL)
    assert general.depends_on == [knowledge.task_id]
    levels = dependency_order(plan.tasks)
    assert knowledge.task_id in levels[0]
    assert general.task_id in levels[1]


def test_dependency_order_layers_independent_tasks_together() -> None:
    tasks = [
        _task("t-a"),
        _task("t-b"),
        _task("t-c", depends_on=["t-a", "t-b"]),
        _task("t-d", depends_on=["t-c"]),
    ]

    assert dependency_order(tasks) == [["t-a", "t-b"], ["t-c"], ["t-d"]]


def test_ready_tasks_start_independent_and_wait_dependencies() -> None:
    tasks = [
        _task("t-a"),
        _task("t-b"),
        _task("t-c", depends_on=["t-a"]),
        _task("t-d", depends_on=["t-a", "t-b"], status=TaskStatus.RUNNING),
    ]

    first = ready_tasks(tasks, completed=[])
    second = ready_tasks(tasks, completed=["t-a"])

    # 相互独立的任务可并行：t-a / t-b 同时就绪，t-c / t-d 依赖未完成时不可启动。
    assert [task.task_id for task in first] == ["t-a", "t-b"]
    # 前置 t-a 完成后 t-c 才就绪；已开始（RUNNING）的 t-d 不再出现在就绪列表。
    assert [task.task_id for task in second] == ["t-a", "t-b", "t-c"]


def test_dependency_order_rejects_cycle() -> None:
    tasks = [_task("t-a", depends_on=["t-b"]), _task("t-b", depends_on=["t-a"])]

    with pytest.raises(PlanValidationError, match="环"):
        dependency_order(tasks)


# =====================================================================
# 4. 校验：依赖环 / 缺失任务 ID / 无来源的完成声明（PRD 6.4 末段）
# =====================================================================


def test_validate_plan_rejects_dependency_cycle() -> None:
    tasks = [_task("t-a", depends_on=["t-b"]), _task("t-b", depends_on=["t-a"])]

    with pytest.raises(PlanValidationError, match="环"):
        validate_plan(_plan(tasks))


def test_validate_plan_rejects_missing_task_id() -> None:
    with pytest.raises(PlanValidationError, match="缺失任务 ID"):
        validate_plan(_plan([_task("   ")]))


def test_validate_plan_rejects_missing_dependency_id() -> None:
    tasks = [_task("t-a", depends_on=["t-missing"])]

    with pytest.raises(PlanValidationError, match="依赖的任务不存在"):
        validate_plan(_plan(tasks))


def test_validate_plan_rejects_unknown_requirement_reference() -> None:
    tasks = [_task("t-a", requirement_ids=["R9"])]

    with pytest.raises(PlanValidationError, match="不存在的要求"):
        validate_plan(_plan(tasks))


def test_validate_plan_rejects_requirement_without_task_or_clarification() -> None:
    tasks = [_task("t-a", requirement_ids=["R1"])]

    with pytest.raises(PlanValidationError, match="没有任务或澄清记录"):
        validate_plan(_plan(tasks, requirements=("R1", "R2")))


def test_validate_plan_accepts_requirement_covered_by_clarification() -> None:
    plan = _plan(
        [_task("t-a", requirement_ids=["R1"])],
        requirements=("R1", "R2"),
        clarification=ClarificationRecord(question="问什么", requirement_ids=["R2"]),
    )

    validate_plan(plan)

    assert plan.coverage().uncovered == []


def test_validate_plan_rejects_completion_claim_without_sources() -> None:
    task = _task("t-a", status=TaskStatus.COMPLETED, expected_evidence="")

    with pytest.raises(PlanValidationError, match="没有任何来源依据"):
        validate_plan(_plan([task]))


def test_validate_plan_accepts_completion_with_evidence_ids() -> None:
    task = _task("t-a", status=TaskStatus.COMPLETED, expected_evidence="")

    validate_plan(_plan([task]), evidence_ids={"t-a": ["ev-1"]})


def test_validate_plan_accepts_completion_with_declared_expected_evidence() -> None:
    task = _task("t-a", status=TaskStatus.COMPLETED_WITH_GAPS)

    validate_plan(_plan([task]))


def test_validate_plan_rejects_task_without_requirement() -> None:
    task = _task("t-a", requirement_ids=[])

    with pytest.raises(PlanValidationError, match="没有关联任何用户要求"):
        validate_plan(_plan([task]))


# =====================================================================
# 5. 模型提出的任务（Pydantic + 程序检查）
# =====================================================================


def test_model_proposed_tasks_are_kept_and_rule_duplicates_skipped() -> None:
    question = "查一下报销制度的时限"
    decision = _decision(question)
    proposed = [
        ResearchTask(
            task_id="m-1",
            question="查一下报销制度的时限",
            requirement_ids=["R1"],
            agent_kind=AgentKind.RAG,
            expected_evidence="本工作区原文",
        )
    ]

    plan = build_plan(question, decision, proposed_tasks=proposed)

    assert plan.task_by_id("m-1") is not None
    assert not [task for task in plan.tasks if task.task_id.startswith("t-r1-")]


def test_model_proposed_cycle_is_rejected() -> None:
    question = "查一下报销制度的时限"
    proposed = [
        _task("m-1", depends_on=["m-2"]),
        _task("m-2", depends_on=["m-1"]),
    ]

    with pytest.raises(PlanValidationError, match="模型计划不合法"):
        build_plan(question, _decision(question), proposed_tasks=proposed)


def test_model_proposed_completion_without_source_is_rejected() -> None:
    question = "查一下报销制度的时限"
    proposed = [
        ResearchTask(
            task_id="m-1",
            question=question,
            requirement_ids=["R1"],
            status=TaskStatus.COMPLETED,
            expected_evidence="",
        )
    ]

    with pytest.raises(PlanValidationError, match="没有任何来源依据"):
        build_plan(question, _decision(question), proposed_tasks=proposed)


def test_model_proposed_tasks_are_normalized_to_queued() -> None:
    question = "查一下报销制度的时限"
    proposed = [
        ResearchTask(
            task_id="m-1",
            question=question,
            requirement_ids=["R1"],
            status=TaskStatus.COMPLETED,
            expected_evidence="本工作区原文",
        )
    ]

    plan = build_plan(question, _decision(question), proposed_tasks=proposed)

    task = plan.task_by_id("m-1")
    assert task is not None
    assert task.status is TaskStatus.QUEUED
    assert task.result_ref is None


def test_planner_repairs_requirements_the_model_missed() -> None:
    question = "1. 我们内部方案 A 是什么？\n2. 我们内部方案 B 是什么？"
    decision = _decision(question)
    proposed = [
        ResearchTask(
            task_id="m-1",
            question="我们内部方案 A 是什么",
            requirement_ids=["R1"],
            agent_kind=AgentKind.RAG,
            expected_evidence="本工作区原文",
        )
    ]

    plan = build_plan(question, decision, proposed_tasks=proposed)

    assert plan.coverage().uncovered == []
    assert any(task.task_id != "m-1" for task in plan.tasks)
    assert any("跳过" in note for note in plan.notes)


# =====================================================================
# 6. 澄清路由（PRD 6.2 clarification / A09）
# =====================================================================


def test_clarification_route_records_requirements() -> None:
    question = "这个方案呢？"
    decision = _decision(
        question,
        intent=IntentKind.CLARIFICATION,
        needs_knowledge=False,
        needs_general=False,
        unresolved_terms=["方案"],
    )

    plan = build_plan(question, decision)

    assert plan.tasks == []
    assert plan.clarification is not None
    assert plan.clarification.requirement_ids == ["R1"]
    assert plan.clarification.unresolved_terms == ["方案"]
    assert plan.coverage().uncovered == []


# =====================================================================
# 7. 重复任务与已有结果复用（PRD 6.4）
# =====================================================================


def test_duplicate_requirement_targets_reuse_one_task() -> None:
    question = "查一下报销制度的时限\n查一下报销制度的时限"
    decision = _decision(question)

    plan = build_plan(question, decision)

    assert len(plan.tasks) == 1
    assert plan.tasks[0].requirement_ids == ["R1", "R2"]
    assert any("重复查询目标复用任务" in note for note in plan.notes)


def test_existing_result_is_reused_for_same_workspace_and_revision() -> None:
    question = "查一下报销制度的时限"
    key = result_key("ws-a", 3, question)

    plan = build_plan(
        question,
        _decision(question),
        workspace_id="ws-a",
        knowledge_revision=3,
        existing_results={key: "operations/op-1.json"},
    )

    task = plan.tasks[0]
    assert task.status is TaskStatus.COMPLETED
    assert task.result_ref == "operations/op-1.json"
    assert ready_tasks(plan.tasks, completed=[]) == []
    assert any("复用已有结果" in note for note in plan.notes)


def test_existing_result_is_not_reused_across_knowledge_revision() -> None:
    question = "查一下报销制度的时限"
    key = result_key("ws-a", 3, question)

    plan = build_plan(
        question,
        _decision(question),
        workspace_id="ws-a",
        knowledge_revision=4,
        existing_results={key: "operations/op-1.json"},
    )

    assert plan.tasks[0].status is TaskStatus.QUEUED
    assert plan.tasks[0].result_ref is None


# =====================================================================
# 7b. 整理后的检索查询（用户实测缺陷 C02/C03）
# =====================================================================


def test_retrieval_queries_replace_per_clause_tasks_and_keep_coverage() -> None:
    question = "有一个经典面试题：如果让你设计一个 Agent，你会考虑哪些因素？有哪些维度值得考虑？"
    decision = _decision(question)

    plain = build_plan(question, decision)
    assert [task.question for task in plain.tasks] == [
        "有一个经典面试题：如果让你设计一个 Agent，你会考虑哪些因素",
        "有哪些维度值得考虑",
    ]

    merged = build_plan(
        question,
        decision,
        retrieval_queries=["设计一个 Agent 需要考虑哪些因素和设计维度？"],
    )

    assert [task.task_id for task in merged.tasks] == ["t-q1-knowledge"]
    assert merged.tasks[0].question == "设计一个 Agent 需要考虑哪些因素和设计维度？"
    # 两项要求仍被覆盖（每条要求映射到至少一个任务，PRD 6.4）
    assert merged.tasks[0].requirement_ids == ["R1", "R2"]
    assert merged.coverage().uncovered == []
    assert any("整理后的 1 条查询" in note for note in merged.notes)


def test_retrieval_queries_keep_combine_dependency() -> None:
    question = "结合业界最佳实践，我们的内部风控方案应该怎么设计？"
    decision = _decision(question, needs_general=True)

    merged = build_plan(question, decision, retrieval_queries=["内部风控方案的设计要点"])

    knowledge = [task for task in merged.tasks if task.agent_kind is AgentKind.RAG]
    general = [task for task in merged.tasks if task.agent_kind is AgentKind.GENERAL]
    assert knowledge and general
    assert general[0].depends_on == [knowledge[0].task_id]
    validate_plan(merged)


def test_retrieval_queries_redirect_dependencies_of_every_replaced_task() -> None:
    """F01：多要求场景下，**后续旧任务被合并成空列表**时依赖也必须重定向。

    报告 T01—T05 的真实问题：R1/R2 各建一条知识任务与一条 general 任务，其中
    ``t-r2-general`` 依赖 ``t-r2-knowledge``（结合类要求）。把两条知识任务合并成一条
    查询任务后，旧实现只给第一项旧任务登记新目标，``t-r2-general`` 继续依赖已被删除的
    ``t-r2-knowledge`` ⇒ ``validate_task_graph`` 直接拒绝整个计划（浏览器四个等级都失败）。
    """

    question = (
        "为什么用了 AI 以后个人开发快了，人却更累？"
        "结合知识库中的实践，Harness、AI Native 建设和 Spec 能做哪些基础工作来减少这种负担？"
    )
    decision = _decision(question, needs_general=True)

    tasks, notes, sources = _build_tasks(extract_requirements(question), decision)
    assert "t-r2-general" in {task.task_id for task in tasks}
    assert build_plan(question, decision).tasks  # 合并前：依赖存在，计划可用

    merged = _apply_retrieval_queries(tasks, sources, [question], notes)
    ids = {task.task_id for task in merged}
    general = next(task for task in merged if task.task_id == "t-r2-general")
    assert general.depends_on, "结合类要求的 general 任务必须仍然依赖知识检索任务"
    assert set(general.depends_on) <= ids, "依赖必须指向仍然存在的任务"
    assert all(dep.endswith("-knowledge") for dep in general.depends_on)
    validate_task_graph(merged)
    # 端到端：走真实 build_plan 入口也不得抛 PlanValidationError。
    plan = build_plan(question, decision, retrieval_queries=[question])
    validate_plan(plan)


def test_retrieval_queries_drop_dependency_that_no_longer_exists() -> None:
    """查询条数被裁剪到上限时，不得留下指向已裁剪任务的不存在依赖。"""

    question = "结合知识库中的实践，Harness 与 Spec 能做哪些基础工作？另外 AI Native 建设呢？"
    decision = _decision(question, needs_general=True)
    queries = [f"查询 {index}" for index in range(1, 12)]

    plan = build_plan(question, decision, retrieval_queries=queries)

    ids = {task.task_id for task in plan.tasks}
    for task in plan.tasks:
        assert set(task.depends_on) <= ids
    validate_plan(plan)


def test_retrieval_queries_without_knowledge_tasks_change_nothing() -> None:
    decision = _decision("什么是 Agent？", needs_knowledge=False, needs_general=True)

    plan = build_plan("什么是 Agent？", decision, retrieval_queries=["Agent 概念"])

    assert [task.task_id for task in plan.tasks] == ["t-r1-general"]


def test_result_key_normalizes_punctuation_but_not_workspace() -> None:
    assert result_key("ws-a", 3, "查一下报销制度的时限！") == result_key(
        "ws-a", 3, "查一下报销制度的时限"
    )
    assert result_key("ws-a", 3, "问题") != result_key("ws-b", 3, "问题")
    assert result_key("ws-a", 3, "问题") != result_key("ws-a", 4, "问题")


def test_task_result_cache_records_and_looks_up() -> None:
    cache = TaskResultCache()
    task = _task("t-a")

    key = cache.record(task, "operations/op-2.json", workspace_id="ws-a", knowledge_revision=3)

    assert key == result_key("ws-a", 3, task.question)
    assert (
        cache.lookup(task, workspace_id="ws-a", knowledge_revision=3)
        == "operations/op-2.json"
    )
    assert cache.lookup(task, workspace_id="ws-a", knowledge_revision=4) is None
    assert cache.as_mapping() == {key: "operations/op-2.json"}


# =====================================================================
# 8. 并发预算（PRD 6.4 / 18.1）
# =====================================================================


def test_plan_settings_defaults_match_prd_18_1() -> None:
    settings = PlanSettings()

    assert settings.research_agent_concurrency == 3
    assert settings.model_request_concurrency == 4
    assert PlanSettings(research_agent_concurrency=1, model_request_concurrency=2)


def test_plan_settings_have_no_total_duration_or_round_limit() -> None:
    """PRD 7.4 / 18.1：本版不设置任务总时长与业务搜索轮数上限。"""

    assert set(PlanSettings.model_fields) == {
        "research_agent_concurrency",
        "model_request_concurrency",
    }
    budget = ConcurrencyBudget()
    assert not hasattr(budget, "max_total_seconds")
    assert not hasattr(budget, "max_rounds")


def test_concurrency_budget_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError):
        ConcurrencyBudget(research_agents=0)


async def test_concurrency_budget_limits_research_agents() -> None:
    budget = ConcurrencyBudget(research_agents=2, model_requests=4)
    seen: list[int] = []

    async def worker() -> None:
        async with budget.research_slot():
            seen.append(budget.stats.active_research_agents)
            await asyncio.sleep(0.005)

    await asyncio.gather(*(worker() for _ in range(6)))

    assert budget.stats.peak_research_agents == 2
    assert max(seen) == 2


async def test_concurrency_budget_limits_model_requests() -> None:
    budget = ConcurrencyBudget(research_agents=3, model_requests=1)
    seen: list[int] = []

    async def worker() -> None:
        async with budget.model_slot():
            seen.append(budget.stats.active_model_requests)
            await asyncio.sleep(0.005)

    await asyncio.gather(*(worker() for _ in range(4)))

    assert budget.stats.peak_model_requests == 1
    assert max(seen) == 1
    assert budget.research_agents == 3


def test_concurrency_budget_from_settings_is_configurable() -> None:
    budget = ConcurrencyBudget.from_settings(
        PlanSettings(research_agent_concurrency=5, model_request_concurrency=6)
    )

    assert (budget.research_agents, budget.model_requests) == (5, 6)


# =====================================================================
# 9. PlannerAgent 门面
# =====================================================================


def test_planner_agent_plans_validates_and_orders() -> None:
    question = "我们内部风控方案怎么设计？请结合业界最佳实践给出建议"
    decision = _decision(question, needs_general=True)
    agent = PlannerAgent(workspace_id="ws-a", knowledge_revision=9)

    plan = agent.plan(question, decision)
    agent.validate(plan)
    levels = agent.levels(plan)

    assert plan.workspace_id == "ws-a"
    assert plan.knowledge_revision == 9
    assert len(levels) == 2
    assert agent.ready(plan, completed=[]) == [task for task in plan.tasks if not task.depends_on]
    assert agent.budget.research_agents == 3


def test_planner_agent_records_results_for_reuse() -> None:
    question = "查一下报销制度的时限"
    agent = PlannerAgent(workspace_id="ws-a", knowledge_revision=3)
    plan = agent.plan(question, _decision(question))
    task = plan.tasks[0]

    agent.record_result(task, "operations/op-9.json")
    second = agent.plan(question, _decision(question))

    assert second.tasks[0].result_ref == "operations/op-9.json"
    assert second.tasks[0].status is TaskStatus.COMPLETED


def test_planner_agent_validate_surfaces_plan_validation_error() -> None:
    agent = PlannerAgent(workspace_id="ws-a")
    plan = build_plan("查一下报销制度", _decision("查一下报销制度"))
    plan.tasks[0].status = TaskStatus.COMPLETED
    plan.tasks[0].expected_evidence = ""

    with pytest.raises(PlanValidationError):
        agent.validate(plan)


# =====================================================================
# 10. 账目辅助
# =====================================================================


def test_requirement_coverage_accounts_ratio_and_by_task() -> None:
    plan = _plan(
        [_task("t-a", requirement_ids=["R1"]), _task("t-b", requirement_ids=["R1", "R2"])],
        requirements=("R1", "R2", "R3"),
    )

    coverage = requirement_coverage(plan)

    assert coverage.covered == ["R1", "R2"]
    assert coverage.uncovered == ["R3"]
    assert coverage.by_task == {"t-a": ["R1"], "t-b": ["R1", "R2"]}
    assert coverage.ratio == pytest.approx(2 / 3)
    assert coverage.complete is False


def test_task_source_table_maps_to_agent_kinds() -> None:
    assert TASK_SOURCES[TaskSource.WEB][0] is AgentKind.WEB
    assert TASK_SOURCES[TaskSource.GENERAL][0] is AgentKind.GENERAL
    assert TASK_SOURCES[TaskSource.KNOWLEDGE][0] is AgentKind.RAG
    assert TASK_SOURCES[TaskSource.HISTORY][0] is AgentKind.RAG
    assert TASK_SOURCES[TaskSource.PRODUCT][0] is AgentKind.GENERAL
