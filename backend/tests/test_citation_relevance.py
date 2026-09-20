"""`CitationEntry` 的纯追加 rerank 字段 + 透传（集成 t5，PRD §8.3「已筛选资料」）。

背景：main 在旧 ``/api/chat`` 的 ``meta.sources`` 上已经展示过「排序分数 / 组合成员 range /
证据哈希」；V1.2.0 的 run 架构没有 ``meta.sources``，来源只能经 ``answer.meta.citations``
到达界面。本文件守住三件事：

1. **向后兼容**：追加字段全部带默认值 ⇒ 既有构造 / 序列化 / ``extra="forbid"`` 行为不变，
   旧记录（没有这些键）反序列化仍成功；
2. **透传不伪造**：记录命中时字段来自**真实筛选产物**，未命中时保持默认值
   （不得出现 0 分 / 0% 这类编造值）；
3. **形状与 main 一致**：``member_refs`` 就是视图成员引用 ``MemberRef.to_dict()`` 的原样投影。
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from app.agent.contracts import (
    CitationEntry,
    CitationSourceStatus,
    Evidence,
    EvidenceSourceType,
    SourceType,
)
from app.agent.verification import assign_citation_ids
from app.retrieval.evidence_view import MemberRef

# ------------------------------------------------------------------ 夹具


def _evidence(source_ref: str = "doc-1#blk-1", *, content: str = "苹果富含维生素 C。") -> Evidence:
    return Evidence(
        evidence_id=f"ev-{abs(hash(source_ref)) % 10**8:08d}",
        workspace_id="ws-1",
        run_id="run-1",
        source_type=EvidenceSourceType.KNOWLEDGE,
        source_ref=source_ref,
        content=content,
        content_hash="a" * 64,
        retrieved_at="2026-10-01T00:00:00Z",
    )


def _placeholder(evidence: Evidence) -> str:
    return "{{" + evidence.evidence_id + "}}"


def _member_ref() -> MemberRef:
    """与 ``evidence_view`` 同形的成员引用（补充成员：有半开区间）。"""

    return MemberRef(
        block_id="blk-1",
        content_hash="b" * 64,
        relation="supplement",
        start=4,
        end=12,
        block_char_offset=0,
    )


def _record(source_ref: str, **overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source_ref": source_ref,
        "relevance_checked": True,
        # 两个分数字段刻意用**不同值**：similarity = 原始向量分数，rerank_score = 重排分数。
        "similarity": 0.42,
        "rerank_score": 0.812,
        "score_type": "rerank",
        "profile_id": "profile-cal-1",
        "evidence_hash": "c" * 64,
        "member_refs": [_member_ref().to_dict()],
    }
    record.update(overrides)
    return record


# ------------------------------------------------------------------ ① 向后兼容


def test_citation_entry_construction_and_defaults_are_backward_compatible() -> None:
    """既有字段位置与默认值不变；新增字段全部有默认值。"""

    citation = CitationEntry(
        label="S1",
        source_type=SourceType.KNOWLEDGE,
        evidence_id="ev-1",
        workspace_id="ws-1",
    )

    assert citation.workspace_name == ""
    assert citation.source_ref == "" and citation.snippet == ""
    assert citation.source_status is CitationSourceStatus.ACTIVE
    # 追加字段的默认值：未筛选 ⇒ 明确「没有」而不是 0 分 / 0% / 空字符串分数
    assert citation.relevance_checked is False
    assert citation.rerank_score is None
    assert citation.score_type == ""
    assert citation.profile_id == ""
    assert citation.evidence_hash == ""
    assert citation.member_refs == []
    # 原始向量相似度同样是「没有就是 None」，不是 0（PRD RAG-relevance :52 词法命中口径）
    assert citation.similarity is None


def test_citation_entry_round_trips_old_payload_without_new_keys() -> None:
    """旧记录（不含新键）必须照旧反序列化 —— 已入库的 answer 不受追加影响。"""

    legacy = {
        "label": "W2",
        "source_type": "web",
        "evidence_id": "ev-2",
        "workspace_id": "ws-1",
        "workspace_name": "默认",
        "source_ref": "https://example.com/a",
        "snippet": "摘录",
        "source_status": "deleted",
    }
    citation = CitationEntry.model_validate(legacy)

    assert citation.label == "W2" and citation.source_status is CitationSourceStatus.DELETED
    assert citation.relevance_checked is False and citation.member_refs == []
    # 往返：投影里多出默认值键，但不丢任何旧字段
    dumped = citation.model_dump(mode="json")
    assert {key: dumped[key] for key in legacy} == legacy
    assert CitationEntry.model_validate(dumped).model_dump(mode="json") == dumped


def test_citation_entry_still_forbids_unknown_fields() -> None:
    """``extra="forbid"`` 未被追加破坏：未知字段照旧报错。"""

    with pytest.raises(ValidationError):
        CitationEntry(
            label="S1",
            source_type=SourceType.KNOWLEDGE,
            evidence_id="ev-1",
            workspace_id="ws-1",
            rerank_probability=0.9,
        )


def test_citation_entry_accepts_the_appended_fields() -> None:
    citation = CitationEntry(
        label="S1",
        source_type=SourceType.KNOWLEDGE,
        evidence_id="ev-1",
        workspace_id="ws-1",
        relevance_checked=True,
        rerank_score=0.812,
        score_type="rerank",
        profile_id="profile-cal-1",
        evidence_hash="c" * 64,
        member_refs=[_member_ref().to_dict()],
    )

    assert citation.relevance_checked is True
    assert citation.rerank_score == pytest.approx(0.812)
    assert citation.member_refs[0]["relation"] == "supplement"
    assert citation.model_dump(mode="json")["member_refs"][0]["start"] == 4
    assert "sk-" not in citation.model_dump_json()


# ------------------------------------------------------------------ ② 透传


def test_assign_citation_ids_carries_relevance_from_the_real_record() -> None:
    """有记录 ⇒ 引用带上真实值（排序分数 / 档案 / 证据哈希 / 成员引用）。"""

    evidence = _evidence()
    assignment = assign_citation_ids(
        "苹果富含维生素 C " + _placeholder(evidence),
        [evidence],
        relevance_by_source={evidence.source_ref: _record(evidence.source_ref)},
    )

    citation = assignment.citations[0]
    assert citation.label == "S1"
    assert citation.relevance_checked is True
    assert citation.rerank_score == pytest.approx(0.812)
    assert citation.score_type == "rerank"
    assert citation.profile_id == "profile-cal-1"
    assert citation.evidence_hash == "c" * 64
    assert citation.member_refs == [_member_ref().to_dict()]
    # 既有字段不受影响
    assert citation.source_ref == evidence.source_ref
    assert citation.snippet == "苹果富含维生素 C。"


def test_assign_citation_ids_without_record_keeps_defaults() -> None:
    """无记录（未启用筛选 / 该来源未被选中）⇒ 全部默认值，**不伪造**分数。"""

    evidence = _evidence()
    for kwargs in ({}, {"relevance_by_source": {}}, {"relevance_by_source": {"other#blk": {}}}):
        assignment = assign_citation_ids(
            "正文 " + _placeholder(evidence), [evidence], **kwargs
        )
        citation = assignment.citations[0]
        assert citation.relevance_checked is False, kwargs
        assert citation.rerank_score is None, kwargs
        assert citation.score_type == "" and citation.profile_id == ""
        assert citation.evidence_hash == "" and citation.member_refs == [], kwargs


def test_assign_citation_ids_ignores_unknown_record_keys() -> None:
    """记录里多出的诊断键不进契约（只取已声明的 7 个键）。"""

    evidence = _evidence()
    assignment = assign_citation_ids(
        "正文 " + _placeholder(evidence),
        [evidence],
        relevance_by_source={
            evidence.source_ref: _record(evidence.source_ref, scope_id="global", candidate_count=9)
        },
    )

    citation = assignment.citations[0]
    assert citation.relevance_checked is True
    assert "scope_id" not in citation.model_dump()
    assert "candidate_count" not in citation.model_dump()


# ------------------------------------------------------------------ ②b 原始向量相似度（t7）


def test_assign_citation_ids_carries_the_original_vector_similarity() -> None:
    """① 经筛选的引用带**真实** ``similarity``（原始向量分数），且与 ``rerank_score`` 互不混淆。"""

    evidence = _evidence()
    assignment = assign_citation_ids(
        "正文 " + _placeholder(evidence),
        [evidence],
        relevance_by_source={evidence.source_ref: _record(evidence.source_ref)},
    )

    citation = assignment.citations[0]
    assert citation.similarity == pytest.approx(0.42)  # 检索阶段的向量分数
    assert citation.rerank_score == pytest.approx(0.812)  # 筛选服务的重排分数
    assert citation.similarity != citation.rerank_score  # 用不同值断言：两条通路不得互相代入
    assert citation.score_type == "rerank"  # 分数类型只描述 rerank_score


def test_similarity_is_never_taken_from_the_rerank_score() -> None:
    """④ 记录只有 ``rerank_score``、没有 ``similarity`` ⇒ ``similarity`` 保持 ``None``。"""

    evidence = _evidence()
    record = _record(evidence.source_ref)
    record.pop("similarity")
    citation = assign_citation_ids(
        "正文 " + _placeholder(evidence),
        [evidence],
        relevance_by_source={evidence.source_ref: record},
    ).citations[0]

    assert citation.rerank_score == pytest.approx(0.812)
    assert citation.similarity is None, "不得把重排分数当成向量相似度（PRD :304）"


def test_missing_similarity_is_none_not_zero() -> None:
    """② 无向量分数 ⇒ ``None``：显式 ``None`` 与「记录里根本没有该键」两种形态都不编造 0。"""

    evidence = _evidence()
    explicit_none = _record(evidence.source_ref, similarity=None)
    missing_key = _record(evidence.source_ref)
    missing_key.pop("similarity")

    for index, record in enumerate((explicit_none, missing_key)):
        citation = assign_citation_ids(
            "正文 " + _placeholder(evidence),
            [evidence],
            relevance_by_source={evidence.source_ref: record},
        ).citations[0]
        assert citation.similarity is None, (index, record)
        assert citation.rerank_score == pytest.approx(0.812), index


def test_a_real_zero_similarity_is_preserved_not_treated_as_missing() -> None:
    """真实的 0 向量分数如实保留：``None`` 只表示「没有这个分数」，不是「分数为 0」。"""

    evidence = _evidence()
    citation = assign_citation_ids(
        "正文 " + _placeholder(evidence),
        [evidence],
        relevance_by_source={evidence.source_ref: _record(evidence.source_ref, similarity=0.0)},
    ).citations[0]

    assert citation.similarity == 0.0
    assert citation.similarity is not None


def test_similarity_round_trips_with_extra_forbid_intact() -> None:
    """③ ``extra="forbid"`` 下 ``similarity`` 可正常声明、投影与往返。"""

    citation = CitationEntry(
        label="S1",
        source_type=SourceType.KNOWLEDGE,
        evidence_id="ev-1",
        workspace_id="ws-1",
        similarity=0.379238,  # PRD 复现案例里那个真实值
    )
    dumped = citation.model_dump(mode="json")
    assert dumped["similarity"] == pytest.approx(0.379238)
    assert CitationEntry.model_validate(dumped) == citation

    with pytest.raises(ValidationError):
        CitationEntry(
            label="S1",
            source_type=SourceType.KNOWLEDGE,
            evidence_id="ev-1",
            workspace_id="ws-1",
            similarity_score=0.5,
        )


def test_assign_citation_ids_partial_record_keeps_other_defaults() -> None:
    """记录只有部分键时，其余保持默认（绝不把缺失分数补成 0）。"""

    evidence = _evidence()
    assignment = assign_citation_ids(
        "正文 " + _placeholder(evidence),
        [evidence],
        relevance_by_source={evidence.source_ref: {"source_ref": evidence.source_ref}},
    )

    citation = assignment.citations[0]
    assert citation.relevance_checked is False
    assert citation.rerank_score is None


def test_member_refs_shape_matches_main() -> None:
    """组合成员引用形状与 main 的 ``MemberRef.to_dict()`` 逐字一致。"""

    expected_keys = {"block_id", "content_hash", "relation", "start", "end", "block_char_offset"}
    reference = _member_ref().to_dict()
    assert set(reference) == expected_keys

    evidence = _evidence()
    assignment = assign_citation_ids(
        "正文 " + _placeholder(evidence),
        [evidence],
        relevance_by_source={evidence.source_ref: _record(evidence.source_ref)},
    )
    members = assignment.citations[0].member_refs
    assert len(members) == 1
    assert set(members[0]) == expected_keys
    assert members[0] == reference


def test_web_and_history_sources_get_no_rerank_fields() -> None:
    """非知识来源（网页）不带筛选字段，即使记录里恰好有同 source_ref 的条目。"""

    evidence = Evidence(
        evidence_id="ev-web-1",
        workspace_id="ws-1",
        run_id="run-1",
        source_type=EvidenceSourceType.WEB,
        source_ref="https://example.com/a",
        content="网页正文",
        content_hash="d" * 64,
        retrieved_at="2026-10-01T00:00:00Z",
    )
    assignment = assign_citation_ids(
        "正文 " + _placeholder(evidence),
        [evidence],
        relevance_by_source={evidence.source_ref: _record(evidence.source_ref)},
    )

    citation = assignment.citations[0]
    assert citation.label == "W1"
    # 记录按 source_ref 匹配到就透传（口径是「该来源被筛选过」），值仍是真实记录
    assert citation.rerank_score == pytest.approx(0.812)
    assert citation.source_type is SourceType.WEB
