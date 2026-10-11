import asyncio
import tempfile
import unittest
from pathlib import Path

from app.agent.app_store import AppStore, ConflictError
from app.agent.contracts import (
    Evidence,
    PermissionMode,
    ReasoningLevel,
    RunStatus,
)
from app.agent.graph import build_main_graph
from app.agent.models import GraphServices
from app.agent.planner import extract_requirements
from app.agent.research import detect_numeric_conflicts
from app.agent.routing import RoutingInput, evaluate_route_rules
from app.agent.runtime import (
    RunManager,
    _prepare_chat_messages,
    _synthesis_prompt,
    default_services,
)
from app.agent.verification import CandidateAnswer
from app.generation.providers import OpenAICompatChatModel


class ChatPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = Path(__file__).resolve().parents[2] / ".tmp_probe"
        self.temporary = tempfile.TemporaryDirectory(dir=directory)
        self.path = Path(self.temporary.name) / "app.db"
        self.store = await AppStore.open(self.path)
        await self.store.create_workspace("workspace", "workspace")
        await self.store.create_conversation("conversation", "workspace")

    async def asyncTearDown(self):
        await self.store.close()
        self.temporary.cleanup()

    async def test_settings_persist(self):
        run = await self.store.create_run(conversation_id="conversation", workspace_id="workspace")
        await self.store.update_run_settings(run.id, PermissionMode.AUTO, ReasoningLevel.LOW)
        updated = await self.store.get_run(run.id)
        self.assertEqual(updated.permission_mode, PermissionMode.AUTO)
        self.assertEqual(updated.reasoning_level, ReasoningLevel.LOW)

    async def test_edit_removes_messages_and_prevents_old_resume(self):
        message = await self.store.append_message(
            conversation_id="conversation", role="user", content="请解释光的散射。"
        )
        run = await self.store.create_run(
            conversation_id="conversation", workspace_id="workspace",
            input_message_id=message.id, status=RunStatus.CANCELLED,
        )
        await self.store.truncate_messages("conversation", 0)
        self.assertEqual(await self.store.list_messages("conversation"), [])
        updated = await self.store.get_run(run.id)
        self.assertIsNone(updated.input_message_id)
        self.assertFalse(updated.resumable)
        self.assertEqual((await self.store.get_conversation("conversation")).context_epoch, 2)

    async def test_edit_rejects_active_task(self):
        await self.store.create_run(conversation_id="conversation", workspace_id="workspace")
        with self.assertRaises(ConflictError):
            await self.store.truncate_messages("conversation", 0)

    async def test_subscription_receives_live_event(self):
        run = await self.store.create_run(conversation_id="conversation", workspace_id="workspace")
        manager = RunManager(store=self.store)
        stream = manager.stream_events(run.id)
        waiting = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        await manager.publish_event(run.id, "run.status", {"status": "running"})
        frame = await asyncio.wait_for(waiting, 2)
        self.assertIn('"run.status"', frame)
        await stream.aclose()
        await manager.aclose()

    async def test_published_answer_notifies_existing_subscription(self):
        run = await self.store.create_run(
            conversation_id="conversation", workspace_id="workspace", status=RunStatus.RUNNING,
        )
        manager = RunManager(store=self.store)
        stream = manager.stream_events(run.id)
        waiting = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        services = default_services(manager, run, {})
        candidate = CandidateAnswer(
            run_id=run.id, workspace_id="workspace", question="你好", content="你好。",
        )
        await services.publish({
            "candidate": candidate.model_dump(mode="json"),
        })
        self.assertIn('"answer.meta"', await asyncio.wait_for(waiting, 2))
        self.assertIn('"answer.delta"', await asyncio.wait_for(anext(stream), 2))
        self.assertIn('"run.completed"', await asyncio.wait_for(anext(stream), 2))
        await stream.aclose()
        await manager.aclose()


class ChatRoutingTests(unittest.TestCase):
    def test_compiled_graph_publishes_after_synthesis(self):
        graph = build_main_graph(GraphServices()).get_graph()
        self.assertEqual(
            [(edge.source, edge.target) for edge in graph.edges if edge.source == "synthesis"],
            [("synthesis", "publish")],
        )
        self.assertNotIn("verification", graph.nodes)
        self.assertNotIn("revise", graph.nodes)

    def test_synthesis_includes_complete_evidence(self):
        evidence = self.evidence("a", "原文内容" * 60 + "关键依据")
        candidate = CandidateAnswer(run_id="run", workspace_id="workspace", question="问题", content="答案")
        prompt = _synthesis_prompt(candidate, [evidence])
        self.assertIn("关键依据", prompt)

    def test_citation_instruction_attaches_to_question(self):
        question = "根据本地文档，风控团队如何优化99分位？请引用资料来源。"
        requirements = extract_requirements(question)
        self.assertEqual(len(requirements), 1)
        self.assertIn("请引用资料来源", requirements[0].text)
        self.assertIn(requirements[0].quote, question)

    def evidence(self, identity, content, source_ref=None):
        return Evidence(
            evidence_id=identity, workspace_id="workspace", run_id="run",
            source_type="knowledge", source_ref=source_ref or identity,
            content=content, content_hash=identity, document_revision=identity,
            retrieved_at="2026-09-20T00:00:00Z",
        )

    def test_unrelated_metrics_are_independent(self):
        evidence = [self.evidence("a", "请求耗时为100ms。"), self.evidence("b", "连接超时为200ms。")]
        self.assertEqual(detect_numeric_conflicts(evidence), [])

    def test_same_claim_different_sources_conflicts(self):
        evidence = [self.evidence("a", "请求耗时为100ms。"), self.evidence("b", "请求耗时为200ms。")]
        self.assertEqual(len(detect_numeric_conflicts(evidence)), 1)

    def test_local_documents_request(self):
        result = evaluate_route_rules(RoutingInput(
            question="根据本地文档，风控团队如何优化99分位？请引用资料来源。"
        ))
        self.assertTrue(result.decision.needs_knowledge)

    def test_explicit_web_request(self):
        result = evaluate_route_rules(RoutingInput(
            question="请联网搜索 Python 官方文档中 asyncio.TaskGroup 的用途，附上来源。"
        ))
        self.assertTrue(result.decision.needs_web)

    def test_no_web_request(self):
        result = evaluate_route_rules(RoutingInput(question="不要联网搜索，请解释天空为什么是蓝色。"))
        self.assertFalse(result.decision.needs_web)

    def test_followup_preserves_history(self):
        model = OpenAICompatChatModel('https://api.deepseek.com', 'deepseek-flash')
        messages = asyncio.run(_prepare_chat_messages(
            model, None, '把刚才的解释缩短。', '',
            [{'role': 'assistant', 'content': '蓝光更容易散射。'}], 'off', None,
            max_tokens=512, compact_history=False,
        ))
        prompt = str(messages)
        self.assertIn("蓝光更容易散射。", prompt)
        self.assertIn("把刚才的解释缩短。", prompt)


if __name__ == "__main__":
    unittest.main()
