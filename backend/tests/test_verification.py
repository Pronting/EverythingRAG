import unittest

from app.agent.contracts import AnswerBasis, Evidence, SourceType
from app.agent.research import MaterialGap, ResearchConclusion, ResearchProgress, ResearchResult
from app.agent.verification import (
    CitationError,
    SynthesisUnavailable,
    answer_basis_display,
    assign_citation_ids,
    collect_evidence,
    compute_answer_basis,
    publish_answer,
    synthesize_answer,
)


def evidence(identity="ev-1", source="knowledge", workspace="ws-1"):
    return Evidence(
        evidence_id=identity, workspace_id=workspace, run_id="run-1", source_type=source,
        source_ref=identity, content="请求耗时为120ms。", content_hash=identity,
        retrieved_at="2026-09-20T00:00:00Z",
    )


def research_result(identity="ev-1"):
    return ResearchResult(
        task_id=identity, agent_kind="rag", status="completed", evidence=[evidence(identity)],
        progress=ResearchProgress(task_id=identity),
        conclusions=[ResearchConclusion(statement="请求耗时为120ms", evidence_ids=[identity], requirement_ids=["R1"])],
    )


def placeholder_model(request):
    """最小候选答案模型：把结论里的证据占位符照抄一遍。

    候选答案**必须由模型撰写**（``synthesize_answer`` 在没有 model 时抛
    ``SynthesisUnavailable``，不再回落确定性拼接稿）；因此这里给测试一个真实的
    可调用实现，而不是让被测代码走已删除的兜底路径。
    """

    return "".join(
        "{{" + eid + "}}" for item in request.conclusions for eid in item.evidence_ids
    ) or "（无证据）"


class SynthesisTests(unittest.TestCase):
    def test_aggregate_and_deduplicate(self):
        result = research_result()
        candidate = synthesize_answer(
            "请求耗时", [result, result], requirements={"R1": "请求耗时"}, model=placeholder_model
        )
        self.assertEqual(len(candidate.evidence), 1)
        self.assertEqual(candidate.covered_requirements, ["R1"])
        self.assertEqual(candidate.conclusions[0].evidence_ids, ["ev-1"])
        self.assertEqual(candidate.citations, [])
        self.assertIn("{{ev-1}}", candidate.content)
        self.assertEqual(len(collect_evidence([result, result])), 1)

    def test_synthesis_without_model_fails_instead_of_falling_back(self):
        """没有可用的对话模型时明确失败：不得用确定性拼接稿冒充答案（PRD 8.1/8.3）。"""

        with self.assertRaises(SynthesisUnavailable):
            synthesize_answer("请求耗时", [research_result()], requirements={"R1": "请求耗时"})

    def test_missing_requirement_preserves_research_gap(self):
        result = research_result()
        result.gaps = [MaterialGap(requirement_ids=["R2"], missing="缺少时间范围", impact="无法确定适用日期")]
        candidate = synthesize_answer(
            "问题", [result], requirements={"R1": "耗时", "R2": "时间范围"}, model=placeholder_model
        )
        self.assertEqual(candidate.uncovered_requirements, ["R2"])
        self.assertEqual(candidate.gaps[0].missing, "缺少时间范围")

    def test_extra_gaps_from_delivery_checks_land_in_the_candidate(self):
        """外部核查（如「要求官方来源但没有取得官方页面」）必须进入终态缺口（缺陷 F19）。"""

        candidate = synthesize_answer(
            "问题",
            [research_result()],
            requirements={"R1": "耗时"},
            model=placeholder_model,
            extra_gaps=[
                MaterialGap(requirement_ids=["R1"], missing="没有取得官方页面", impact="不能声称已核实")
            ],
        )
        self.assertIn("没有取得官方页面", [gap.missing for gap in candidate.gaps])
        artifact = publish_answer(candidate, requirements={"R1": "耗时"})
        self.assertEqual(artifact.status.value, "completed_with_gaps")

    def test_basis_types(self):
        for source in ["knowledge", "web", "memory", "history", "catalog", "product"]:
            with self.subTest(source=source):
                basis, components = compute_answer_basis([evidence(source=source)])
                self.assertEqual(basis.value, source)
                self.assertEqual(components, [SourceType(source)])
        self.assertEqual(compute_answer_basis([evidence(source="general_knowledge")])[0], AnswerBasis.GENERAL)
        self.assertEqual(compute_answer_basis([])[0], AnswerBasis.INSUFFICIENT)
        self.assertEqual(compute_answer_basis([evidence(), evidence("ev-2", "web")])[0], AnswerBasis.MIXED)
        self.assertEqual(answer_basis_display("legacy"), "legacy")

    def test_citation_prefixes_and_reuse(self):
        sources = [evidence("ev-1"), evidence("ev-2", "web"), evidence("ev-3", "memory"), evidence("ev-4", "history")]
        assigned = assign_citation_ids("{{ev-1}} {{ev-2}} {{ev-3}} {{ev-4}} {{ev-1}}", sources)
        self.assertEqual(assigned.content, "[S1] [W1] [M1] [H1] [S1]")
        self.assertEqual(assigned.reused_labels, ["S1"])
        again = assign_citation_ids("{{ev-1}}", sources, existing=assigned.citations)
        self.assertEqual(again.content, "[S1]")
        self.assertEqual(again.unused_evidence_ids, ["ev-2", "ev-3", "ev-4"])
        self.assertEqual(again.citations[0].workspace_id, "ws-1")

    def test_invalid_citation_fails(self):
        with self.assertRaises(CitationError):
            assign_citation_ids("{{ev-missing}}", [evidence()])
        for source in ["general_knowledge", "proposal", "catalog", "product"]:
            with self.subTest(source=source), self.assertRaises(CitationError):
                assign_citation_ids("{{ev-1}}", [evidence(source=source)])

    def test_gap_is_dropped_when_another_task_already_answered_it(self):
        """同一项要求已由其他来源回答时，不能再报「资料不足」（用户实测缺陷 C04）。"""

        def fake_model(request):
            return "".join(
                "{{" + eid + "}}" for item in request.conclusions for eid in item.evidence_ids
            )

        rag = research_result("ev-rag")
        rag.gaps = [MaterialGap(requirement_ids=["R1"], missing="本轮共执行 2 条查询，没有命中任何候选片段", impact="")]
        general = research_result("ev-general")
        general.conclusions[0].requirement_ids = ["R1"]
        candidate = synthesize_answer(
            "问题", [rag, general], requirements={"R1": "耗时"}, model=fake_model
        )
        self.assertEqual(candidate.uncovered_requirements, [])
        self.assertEqual(candidate.gaps, [])
        # 发布时不再追加「资料不足说明」小节（has_gaps 依据 gaps + uncovered_requirements）
        artifact = publish_answer(candidate, requirements={"R1": "耗时"})
        self.assertNotIn("资料不足说明", artifact.content)
        self.assertEqual(artifact.status.value, "completed")

    def test_relevance_fields_preserved(self):
        assigned = assign_citation_ids("{{ev-1}}", [evidence()], relevance_by_source={
            "ev-1": {"relevance_checked": True, "rerank_score": 0.8, "similarity": 0.6, "member_refs": []},
        })
        self.assertEqual(assigned.citations[0].rerank_score, 0.8)
        self.assertEqual(assigned.citations[0].similarity, 0.6)


if __name__ == "__main__":
    unittest.main()
