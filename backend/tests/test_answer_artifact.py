import hashlib
import unittest

from app.agent.contracts import AgentError, EvidenceSourceType, RunStatus
from app.agent.research import MaterialGap
from app.agent.verification import (
    AnswerVersionError,
    CandidateAnswer,
    CitationError,
    answer_version,
    assign_citation_ids,
    build_failure_outcome,
    publish_answer,
    synthesize_answer,
)
from tests.test_verification import placeholder_model, research_result


class AnswerArtifactTests(unittest.TestCase):
    def candidate(self):
        candidate = synthesize_answer(
            "耗时", [research_result()], requirements={"R1": "耗时"}, workspace_id="ws-1", run_id="run-1",
            # 候选答案必须由模型撰写（没有 model 时明确失败，不再回落确定性拼接稿）。
            model=placeholder_model,
        )
        assigned = assign_citation_ids(candidate.content, candidate.evidence)
        return candidate.model_copy(update={"content": assigned.content, "citations": assigned.citations})

    def test_direct_publication_preserves_content_and_citations(self):
        candidate = self.candidate()
        artifact = publish_answer(candidate, requirements={"R1": "耗时"})
        self.assertEqual(artifact.status, RunStatus.COMPLETED)
        self.assertEqual(artifact.content, candidate.content)
        self.assertEqual(artifact.citations, candidate.citations)
        self.assertIsNone(artifact.verification)
        self.assertEqual(artifact.content_hash, hashlib.sha256(artifact.content.encode()).hexdigest())

    def test_identity_and_version(self):
        candidate = self.candidate()
        first = publish_answer(candidate, requirements={})
        self.assertEqual(first.answer_id, publish_answer(candidate, requirements={}).answer_id)
        self.assertEqual(publish_answer(candidate, requirements={}, answer_id="custom").answer_id, "custom")
        candidate.content += "补充说明。"
        second = publish_answer(candidate, requirements={})
        self.assertNotEqual(answer_version(first), answer_version(second))
        with self.assertRaises(AnswerVersionError):
            answer_version(first.model_copy(update={"content_hash": ""}))

    def test_research_gaps_are_published_once(self):
        candidate = self.candidate()
        candidate.gaps = [MaterialGap(requirement_ids=["R2"], missing="日期", impact="日期未知")]
        candidate.uncovered_requirements = ["R2"]
        first = publish_answer(candidate, requirements={"R2": "适用日期"})
        self.assertEqual(first.status, RunStatus.COMPLETED_WITH_GAPS)
        self.assertIn("日期未知", first.content)
        candidate.content = first.content
        second = publish_answer(candidate, requirements={"R2": "适用日期"})
        self.assertEqual(second.content.count("### 资料不足说明"), 1)

    def test_invalid_reference_fails_at_publication(self):
        candidate = self.candidate()
        candidate.content += "[S9]"
        with self.assertRaises(CitationError):
            publish_answer(candidate, requirements={})
        candidate = self.candidate()
        candidate.evidence[0].workspace_id = "other"
        with self.assertRaises(CitationError):
            publish_answer(candidate, requirements={})
        candidate = self.candidate()
        candidate.evidence[0].source_type = EvidenceSourceType.WEB
        with self.assertRaises(CitationError):
            publish_answer(candidate, requirements={})

    def test_general_and_empty_content(self):
        candidate = CandidateAnswer(question="你好", content="你好。", run_id="run-1", answer_basis="general")
        self.assertEqual(publish_answer(candidate, requirements={}).citations, [])
        candidate.content = ""
        with self.assertRaises(ValueError):
            publish_answer(candidate, requirements={})

    def test_failure_keeps_research(self):
        error = AgentError(error_code="service_error", message="请求失败")
        failure = build_failure_outcome([research_result()], error, run_id="run-1")
        self.assertTrue(failure.has_research_records())
        self.assertEqual(failure.completed_task_ids, ["ev-1"])
        self.assertIn("resume", failure.resume_action)


if __name__ == "__main__":
    unittest.main()
