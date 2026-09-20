#!/usr/bin/env python
"""路由评测脚本（PRD 16.3）。Owner: T06（routing）。

用法（在 ``backend/`` 目录下执行）::

    python scripts/eval_routing.py --dry-run          # 评测集规模与分布统计（不发模型请求）
    python scripts/eval_routing.py                    # 规则层离线评测 + 指标报告
    python scripts/eval_routing.py --report out.json  # 落盘逐条证据（RouteDecision / 范围检查 / 线索）
    python scripts/eval_routing.py --catalog cat.json # 用真实工作区导出的目录替代线索代理

评测集放在 ``tests/data/routing_eval_cases.json``（T06 归属），本脚本只读它。

**能力边界（不要把它当成端到端评测）**：

- 默认离线模式只覆盖**规则层**：线索来自评测集里真实语料证据串构成的代理目录
  （``catalog_hint_origin=cases_evidence``），不访问任何工作区，也不调用模型。
- ``--catalog`` 模式用真实工作区导出的目录（``KnowledgeCatalog.model_dump_json()``）
  替代代理，此时结果反映真实目录下的路由表现。
- 跨工作区误用（M7）与「执行过的工具」「进入模型的证据」需要真实运行：
  由 T_verify（t11）在真实工作区执行并写入报告。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Final

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.agent.contracts import IntentKind
from app.agent.routing import (
    ROUTING_RULES_VERSION,
    IntentAgent,
    RouteResult,
    RoutingHistoryTurn,
    RoutingInput,
)
from app.generation.knowledge_catalog import CatalogMatch, KnowledgeCatalog
from app.generation.prompt_policy import QueryMode, classify_query

DEFAULT_CASES = BACKEND_ROOT / "tests" / "data" / "routing_eval_cases.json"
SOURCE_CASES = BACKEND_ROOT.parent / "benchmark" / "cases.json"

CATEGORIES: Final[tuple[str, ...]] = (
    "internal",
    "general",
    "freshness",
    "product",
    "multi_source",
    "multi_turn",
)

#: PRD 16.3 的配额下限。
QUOTAS: Final[dict[str, int]] = {
    "internal": 30,
    "general": 30,
    "freshness": 20,
    "product": 10,
    "multi_source": 15,
    "multi_turn": 15,
}

ALLOWED_SOURCES: Final[frozenset[str]] = frozenset(
    {"knowledge", "web", "general", "product", "catalog", "history"}
)
ALLOWED_SPLITS: Final[frozenset[str]] = frozenset({"dev", "holdout"})
PLANNED_TOOLS: Final[dict[str, tuple[str, ...]]] = {
    "knowledge": ("search_knowledge", "read_document"),
    "web": ("web_search", "read_web_page"),
    "general": (),
    "product": (),
    "catalog": ("get_workspace_overview", "lookup_catalog"),
    "history": ("search_history", "read_history"),
}
EVAL_NOW = "2026-09-19T10:00:00+08:00"
EVAL_TIMEZONE = "Asia/Shanghai"


class DatasetError(ValueError):
    """评测集结构或配额不满足 PRD 16.3 的要求。"""


# ---------------------------------------------------------------------
# 载入与校验
# ---------------------------------------------------------------------


def load_dataset(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DatasetError(f"评测集不存在: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"评测集不是合法 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetError("评测集顶层必须是对象")
    return payload


def validate_dataset(payload: dict[str, Any]) -> list[str]:
    """逐条校验评测集；返回问题列表（空表示通过）。"""

    problems: list[str] = []
    if payload.get("schema_version") != 1:
        problems.append(f"schema_version 必须为 1，实际 {payload.get('schema_version')!r}")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise DatasetError("评测集缺少 cases 数组")
    if len(cases) < sum(QUOTAS.values()):
        problems.append(f"条目总数 {len(cases)} 少于配额总和 {sum(QUOTAS.values())}")

    seen_ids: set[str] = set()
    counts: Counter[str] = Counter()
    family_splits: dict[str, set[str]] = {}
    real_lookup = source_case_digests()
    intent_values = {item.value for item in IntentKind}

    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            problems.append(f"cases[{index}] 不是对象")
            continue
        case_id = str(case.get("id") or "")
        if not case_id:
            problems.append(f"cases[{index}] 缺少 id")
        elif case_id in seen_ids:
            problems.append(f"id 重复: {case_id}")
        seen_ids.add(case_id)

        category = str(case.get("category") or "")
        if category not in CATEGORIES:
            problems.append(f"{case_id}: 未知 category={category!r}")
        else:
            counts[category] += 1

        question = str(case.get("question") or "").strip()
        if not question:
            problems.append(f"{case_id}: question 为空")

        source = str(case.get("source") or "")
        if source not in {"real", "constructed"}:
            problems.append(f"{case_id}: source 必须是 real 或 constructed，实际 {source!r}")
        if source == "real":
            reused = str(case.get("reused_case_id") or "")
            if not reused:
                problems.append(f"{case_id}: real 条目必须带 reused_case_id")
            elif real_lookup and reused in real_lookup:
                if real_lookup[reused] != query_digest(question):
                    problems.append(f"{case_id}: 与 benchmark/cases.json 的 {reused} 文本不一致")
            elif real_lookup:
                problems.append(
                    f"{case_id}: reused_case_id={reused} 在 benchmark/cases.json 中不存在"
                )
        if source == "constructed" and not str(case.get("constructed_reason") or "").strip():
            problems.append(f"{case_id}: constructed 条目必须带 constructed_reason")

        split = str(case.get("split") or "")
        if split not in ALLOWED_SPLITS:
            problems.append(f"{case_id}: split 必须是 dev 或 holdout，实际 {split!r}")
        family = str(case.get("family") or case_id)
        family_splits.setdefault(family, set()).add(split)

        required = case.get("required_sources")
        if not isinstance(required, list) or not required:
            problems.append(f"{case_id}: required_sources 必须是非空数组")
        else:
            unknown = [item for item in required if item not in ALLOWED_SOURCES]
            if unknown:
                problems.append(f"{case_id}: required_sources 含未知来源 {unknown}")
        intents = case.get("acceptable_intents")
        if not isinstance(intents, list) or not intents:
            problems.append(f"{case_id}: acceptable_intents 必须是非空数组")
        else:
            unknown_intents = [item for item in intents if item not in intent_values]
            if unknown_intents:
                problems.append(f"{case_id}: acceptable_intents 含未知取值 {unknown_intents}")
        if not isinstance(case.get("forbidden_misstatements"), list):
            problems.append(f"{case_id}: forbidden_misstatements 必须是数组")
        if str(case.get("catalog_hint_origin") or "") not in {"cases_evidence", "none"}:
            problems.append(f"{case_id}: catalog_hint_origin 取值非法")
        if category == "multi_turn" and not case.get("turns"):
            problems.append(f"{case_id}: multi_turn 条目必须带 turns")

    for family, splits in family_splits.items():
        if len(splits) > 1:
            problems.append(f"family {family} 同时出现在 {sorted(splits)}：改写必须归同一集合")

    for category, minimum in QUOTAS.items():
        if counts[category] < minimum:
            problems.append(f"{category} 只有 {counts[category]} 条，低于配额 {minimum}")

    declared = payload.get("counts") or {}
    if declared.get("total") not in (None, len(cases)):
        problems.append(f"counts.total={declared.get('total')} 与实际 {len(cases)} 不一致")
    return problems


def query_digest(question: str) -> str:
    """问题原文的 SHA-256（与 ``benchmark/cases_loader._sha256`` 同一口径）。

    ``benchmark/cases.json`` 在集成后是**脱敏结构**（main 的 ``chore(privacy)`` 提交，
    PRD §7.2）：私文问题原文与证据串不入仓库，只保留 ``query_sha256``。因此「真实条目与
    benchmark 用例逐字一致」这条来源纪律改为**指纹比对**——SHA-256 相等 ⇔ 文本逐字相等
    （碰撞不可行），证据强度不因脱敏降低，且不依赖本机私文文件是否存在。
    """

    return hashlib.sha256(str(question).encode("utf-8", errors="replace")).hexdigest()


def source_case_digests() -> dict[str, str]:
    """``benchmark/cases.json`` 的问题指纹表：``{case_id: sha256(query)}``。

    两种副本都收敛成同一形状：

    * 脱敏结构（带 ``redaction`` 段）：直接读 ``query_sha256``；
    * 未脱敏的旧副本：内联 ``query`` 现算指纹。

    于是调用方只有一种比较方式（见 :func:`query_digest`），不再有两套文本口径。
    """

    if not SOURCE_CASES.exists():
        return {}
    try:
        payload = json.loads(SOURCE_CASES.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    digests: dict[str, str] = {}
    for case in payload.get("cases", []):
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("id") or "")
        if not case_id:
            continue
        recorded = case.get("query_sha256")
        if isinstance(recorded, str) and recorded.strip():
            digests[case_id] = recorded.strip()
            continue
        query = case.get("query")
        if query is not None:
            digests[case_id] = query_digest(query)
    return digests


# ---------------------------------------------------------------------
# 路由输入构造
# ---------------------------------------------------------------------


def _hint_matches(case: dict[str, Any]) -> list[CatalogMatch]:
    """离线线索代理：真实条目的 evidence 串当作文档级线索（分数高于弱命中阈值）。"""

    matches: list[CatalogMatch] = []
    for index, hint in enumerate(case.get("catalog_hints") or [], start=1):
        text = str(hint or "").strip()
        if not text:
            continue
        matches.append(
            CatalogMatch(
                kind="heading",
                document_id=f"hint-{index}",
                title="线索代理",
                heading_path=text,
                snippet=text,
                score=400,
                matched_on=text,
            )
        )
    return matches


def build_input(
    case: dict[str, Any], *, index: int, catalog: KnowledgeCatalog | None
) -> RoutingInput:
    history = [
        RoutingHistoryTurn(role=str(turn.get("role")), content=str(turn.get("content")))
        for turn in (case.get("turns") or [])
    ]
    if catalog is not None:
        doc_chunks = sum(document.chunk_count for document in catalog.documents)
        return RoutingInput(
            question=str(case["question"]),
            now=EVAL_NOW,
            timezone=EVAL_TIMEZONE,
            recent_history=history,
            file_count=len(catalog.documents),
            chunk_count=doc_chunks,
            workspace_id=catalog.workspace_id or "eval-workspace",
            knowledge_revision=catalog.knowledge_revision,
        )
    hints = _hint_matches(case)
    return RoutingInput(
        question=str(case["question"]),
        now=EVAL_NOW,
        timezone=EVAL_TIMEZONE,
        recent_history=history,
        catalog_hits=hints,
        catalog_clues=[match.matched_on for match in hints],
        file_count=1 if hints else 0,
        chunk_count=len(hints),
        workspace_id="offline-proxy",
        knowledge_revision=index,
    )


def planned_tools(decision: Any) -> list[str]:
    """本路线会用到哪些工具（离线模式**未执行**，仅用于核对来源选择）。"""

    tools: list[str] = []
    if decision.needs_knowledge:
        tools.extend(PLANNED_TOOLS["knowledge"])
    if decision.needs_web:
        tools.extend(PLANNED_TOOLS["web"])
    if decision.needs_history:
        tools.extend(PLANNED_TOOLS["history"])
    if decision.intent in (IntentKind.CATALOG, IntentKind.PRODUCT):
        tools.extend(PLANNED_TOOLS["catalog"])
    return list(dict.fromkeys(tools))


# ---------------------------------------------------------------------
# 判定与指标（PRD 16.3 表）
# ---------------------------------------------------------------------


def _sources_satisfied(decision: Any, required: list[str]) -> bool:
    for source in required:
        if source == "knowledge" and not decision.needs_knowledge:
            return False
        if source == "web" and not decision.needs_web:
            return False
        if source == "general" and not decision.needs_general:
            return False
        if source == "history" and not decision.needs_history:
            return False
        if source == "product" and decision.intent is not IntentKind.PRODUCT:
            return False
        if source == "catalog" and decision.intent not in (
            IntentKind.CATALOG,
            IntentKind.PRODUCT,
        ):
            return False
    return True


def evaluate_case(case: dict[str, Any], result: RouteResult) -> tuple[bool, list[str]]:
    decision = result.decision
    reasons: list[str] = []
    category = str(case["category"])
    required = list(case.get("required_sources") or [])
    acceptable = list(case.get("acceptable_intents") or [])

    if decision.intent.value not in acceptable:
        reasons.append(f"intent={decision.intent.value} 不在可接受路径 {acceptable}")
    if not _sources_satisfied(decision, required):
        reasons.append(
            f"来源要求未满足：{required}（实际 K/W/G/H = "
            f"{decision.needs_knowledge}/{decision.needs_web}/"
            f"{decision.needs_general}/{decision.needs_history}）"
        )
    freshness_required = str(case.get("freshness_required") or "none")
    if freshness_required != "none" and decision.freshness.value != freshness_required:
        reasons.append(f"时效要求 {freshness_required}，实际 {decision.freshness.value}")
    if category in {"general", "product"} and decision.needs_knowledge:
        reasons.append("普通问题被注入本区知识检索")
    if category == "freshness" and not decision.needs_web:
        reasons.append("时效问题未选择联网")
    return not reasons, reasons


def _meets(requirement: str, rate: float) -> bool:
    if requirement == ">=95%":
        return rate >= 0.95
    return rate >= 1.0


def metric(
    metric_id: str,
    name: str,
    requirement: str,
    failures: list[dict[str, Any]],
    total: int,
    *,
    note: str = "",
) -> dict[str, Any]:
    numerator = max(0, total - len(failures))
    rate = (numerator / total) if total else None
    return {
        "id": metric_id,
        "name": name,
        "requirement": requirement,
        "numerator": numerator,
        "denominator": total,
        "rate": rate,
        "passed": (rate is not None and _meets(requirement, rate)),
        "note": note,
        "failed_case_ids": [str(case["id"]) for case in failures[:20]],
    }


def run(
    payload: dict[str, Any],
    *,
    catalog: KnowledgeCatalog | None,
    split: str,
    limit: int | None,
) -> dict[str, Any]:
    cases = [case for case in payload["cases"] if split == "all" or case["split"] == split]
    if limit is not None:
        cases = cases[:limit]
    agent = IntentAgent(catalog=catalog)
    results: dict[str, RouteResult] = {}
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    modes: Counter[str] = Counter()

    for index, case in enumerate(cases, start=1):
        routing_input = build_input(case, index=index, catalog=catalog)
        result = agent.decide(routing_input)
        results[str(case["id"])] = result
        modes[classify_query(str(case["question"])).value] += 1
        accepted, reasons = evaluate_case(case, result)
        if not accepted:
            failures.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "split": case["split"],
                    "source": case["source"],
                    "question": case["question"],
                    "reasons": reasons,
                    "decision": result.decision.model_dump(mode="json"),
                    "rules": result.scope.rules,
                }
            )
        records.append(
            {
                "id": case["id"],
                "category": case["category"],
                "split": case["split"],
                "source": case["source"],
                "question": case["question"],
                "accepted": accepted,
                "reasons": reasons,
                "routing_rules_version": ROUTING_RULES_VERSION,
                # PRD 16.3：至少记录实际 RouteDecision、范围检查、执行过的工具、进入模型的证据。
                "decision": result.decision.model_dump(mode="json"),
                "scope_check": {
                    "scope_evidence_ids": result.scope.scope_evidence_ids,
                    "resolved_terms": result.scope.resolved_terms,
                    "unresolved_terms": result.scope.unresolved_terms,
                    "definition_unknown_terms": result.scope.definition_unknown_terms,
                    "catalog_empty": result.scope.catalog_empty,
                    "notes": result.scope.notes,
                },
                "planned_tools_not_executed": planned_tools(result.decision),
                "prompt_evidence": {
                    "model_used": result.diagnostics.model_used,
                    "catalog_clues": len(case.get("catalog_hints") or []),
                    "history_turns": len(case.get("turns") or []),
                    "memory_snapshot_chars": 0,
                    "note": "离线模式不组装提示、不发模型请求；真实运行由 T_verify 执行",
                },
            }
        )

    internal = [case for case in cases if case["category"] == "internal"]
    internal_holdout = [case for case in internal if case["split"] == "holdout"]
    ordinary = [case for case in cases if case["category"] in {"general", "freshness", "product"}]
    freshness = [case for case in cases if case["category"] == "freshness"]
    product = [case for case in cases if case["category"] == "product"]
    multi = [case for case in cases if case["category"] in {"multi_source", "multi_turn"}]

    def missed(candidate: list[dict[str, Any]], predicate: Any) -> list[dict[str, Any]]:
        return [case for case in candidate if not predicate(case, results[str(case["id"])])]

    metrics = [
        metric(
            "M1",
            "用户确认的关键内部问题全部进入知识路径",
            "100%",
            missed(internal, lambda case, result: result.decision.needs_knowledge),
            len(internal),
        ),
        metric(
            "M2",
            "内部问题来源召回率（保留集）",
            "100%",
            missed(internal_holdout, lambda case, result: result.decision.needs_knowledge),
            len(internal_holdout),
        ),
        metric(
            "M3",
            "普通问题无无关知识注入比例",
            ">=95%",
            missed(ordinary, lambda case, result: not result.decision.needs_knowledge),
            len(ordinary),
        ),
        metric(
            "M4",
            "最新信息问题正确选择 Web",
            "100%",
            missed(freshness, lambda case, result: result.decision.needs_web),
            len(freshness),
        ),
        metric(
            "M5",
            "产品和统计问题选择正确来源",
            "100%",
            missed(
                product,
                lambda case, result: _sources_satisfied(
                    result.decision, list(case["required_sources"])
                ),
            ),
            len(product),
        ),
        metric(
            "M6",
            "多来源与多轮问题完成来源要求",
            ">=95%",
            missed(
                multi,
                lambda case, result: _sources_satisfied(
                    result.decision, list(case["required_sources"])
                ),
            ),
            len(multi),
        ),
    ]

    scope_checked = [record for record in records if record["scope_check"]["scope_evidence_ids"]]
    return {
        "mode": "rule-layer-offline",
        "catalog": "real-catalog" if catalog is not None else "hint-proxy",
        "split": split,
        "cases": len(cases),
        "model_calls": 0,
        "routing_rules_version": ROUTING_RULES_VERSION,
        "query_modes": {mode.value: modes.get(mode.value, 0) for mode in QueryMode},
        "metrics": metrics,
        "failures": failures,
        "scope_checks": {
            "cases_with_scope_evidence": len(scope_checked),
            "cases_with_unresolved_terms": sum(
                1 for record in records if record["scope_check"]["unresolved_terms"]
            ),
        },
        "not_covered_here": {
            "M7": "跨工作区内容误用需要真实工作区；由 T_verify 在真实数据上执行",
            "tools_executed": "离线模式不执行工具，只给 planned_tools_not_executed",
            "answer_quality": "答案质量评测（PRD 16.4）由 T_verify 执行",
        },
        "records": records,
    }


# ---------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------


def distribution(payload: dict[str, Any], *, path: Path) -> dict[str, Any]:
    cases = payload["cases"]
    by_category = Counter(str(case["category"]) for case in cases)
    by_source = Counter(str(case["source"]) for case in cases)
    by_split = Counter(str(case["split"]) for case in cases)
    hint_origins = Counter(str(case.get("catalog_hint_origin")) for case in cases)
    real_per_category = Counter(
        str(case["category"]) for case in cases if case["source"] == "real"
    )
    constructed_per_category = Counter(
        str(case["category"]) for case in cases if case["source"] == "constructed"
    )
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "total": len(cases),
        "real": by_source.get("real", 0),
        "constructed": by_source.get("constructed", 0),
        "by_category": {category: by_category.get(category, 0) for category in CATEGORIES},
        "quota": dict(QUOTAS),
        "real_by_category": {
            category: real_per_category.get(category, 0) for category in CATEGORIES
        },
        "constructed_by_category": {
            category: constructed_per_category.get(category, 0) for category in CATEGORIES
        },
        "by_split": {split: by_split.get(split, 0) for split in sorted(ALLOWED_SPLITS)},
        "catalog_hint_origin": dict(hint_origins),
        "label_origin": payload.get("label_origin", ""),
        "label_disclaimer": payload.get("label_disclaimer", ""),
    }


def print_distribution(stats: dict[str, Any]) -> None:
    print("== 路由评测集（PRD 16.3）" + "=" * 40)
    print(f"文件           : {stats['path']}")
    print(f"sha256         : {stats['sha256']}")
    print(f"条目总数       : {stats['total']}（真实 {stats['real']} / 构造 {stats['constructed']}）")
    print(
        "开发集/保留集  : dev "
        f"{stats['by_split'].get('dev', 0)} / holdout {stats['by_split'].get('holdout', 0)}"
    )
    print("分类（实际/配额，真实+构造）:")
    for category in CATEGORIES:
        print(
            f"  - {category:<12} {stats['by_category'][category]:>3} / {stats['quota'][category]:>3}"
            f"  = 真实 {stats['real_by_category'][category]} + 构造 "
            f"{stats['constructed_by_category'][category]}"
        )
    print(f"线索来源       : {stats['catalog_hint_origin']}")
    print(f"标注来源       : {stats['label_origin']}")
    print(f"标注声明       : {stats['label_disclaimer']}")


def print_report(report: dict[str, Any]) -> None:
    print("== 规则层离线评测" + "=" * 44)
    print(f"模式                    : {report['mode']}（目录来源：{report['catalog']}）")
    print(f"评测范围                : split={report['split']}，{report['cases']} 条")
    print(f"实际模型请求            : {report['model_calls']}（离线模式不发请求）")
    print(f"路由规则版本            : {report['routing_rules_version']}")
    print(f"既有查询分类分布        : {report['query_modes']}")
    print("指标（PRD 16.3）:")
    for item in report["metrics"]:
        rate = "n/a" if item["rate"] is None else f"{item['rate'] * 100:.1f}%"
        status = "n/a" if item["passed"] is None else ("PASS" if item["passed"] else "FAIL")
        print(
            f"  [{status}] {item['id']} {item['name']}: "
            f"{item['numerator']}/{item['denominator']} = {rate}（要求 {item['requirement']}）"
        )
        if item["note"]:
            print(f"          注：{item['note']}")
        if item["failed_case_ids"]:
            print(f"          未通过：{', '.join(item['failed_case_ids'])}")
    print(
        "范围检查：带目录证据 "
        f"{report['scope_checks']['cases_with_scope_evidence']} 条，"
        f"含待确认术语 {report['scope_checks']['cases_with_unresolved_terms']} 条"
    )
    print("本脚本不覆盖（属 T_verify 真实运行）:")
    for key, value in report["not_covered_here"].items():
        print(f"  - {key}: {value}")
    if report["failures"]:
        print(f"失败明细（{len(report['failures'])} 条）:")
        for failure in report["failures"][:15]:
            print(f"  - {failure['id']} [{failure['category']}] {failure['question'][:40]}")
            for reason in failure["reasons"]:
                print(f"      · {reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="路由评测（PRD 16.3）")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="评测集 JSON 路径")
    parser.add_argument("--dry-run", action="store_true", help="只输出规模与分布统计")
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（调试用）")
    parser.add_argument("--report", type=Path, default=None, help="把逐条证据写入 JSON 文件")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="真实工作区导出的知识目录 JSON（替代离线线索代理）",
    )
    args = parser.parse_args(argv)

    try:
        payload = load_dataset(args.cases)
        problems = validate_dataset(payload)
    except DatasetError as exc:
        print(f"[评测集错误] {exc}", file=sys.stderr)
        return 2

    stats = distribution(payload, path=args.cases)
    print_distribution(stats)
    if problems:
        print(f"\n[评测集校验失败] {len(problems)} 项：")
        for problem in problems:
            print(f"  - {problem}")
        return 2
    print("\n评测集校验：通过（结构、配额、来源标注、family 与 split 一致性）")
    if args.dry_run:
        return 0

    catalog: KnowledgeCatalog | None = None
    if args.catalog is not None:
        if not args.catalog.exists():
            print(f"[目录文件不存在] {args.catalog}", file=sys.stderr)
            return 2
        catalog = KnowledgeCatalog.model_validate_json(args.catalog.read_text(encoding="utf-8"))
        print(
            f"使用真实目录：{args.catalog}"
            f"（文档 {len(catalog.documents)} 个，术语 {len(catalog.terms)} 条）"
        )

    report = run(payload, catalog=catalog, split=args.split, limit=args.limit)
    print()
    print_report(report)

    if args.report is not None:
        output = dict(report)
        output["dataset"] = stats
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(output, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        print(f"\n逐条证据已写入：{args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
