import asyncio
import json
import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from app.agent.answer_stream import AnswerStream
from app.agent.app_store import AppStore
from app.agent.contracts import Evidence, EvidenceSourceType, RunStatus, utc_now_iso
from app.agent.research import content_hash_of, evidence_id_for
from app.agent.runtime import RunManager, reasoning_publisher, synthesis_model_for_chat
from app.agent.verification import assign_citation_ids, publish_answer, synthesize_answer
from app.core.workspace_context import WorkspaceContextFactory
from app.core.workspace_store import WorkspaceStore

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / '.tmp_probe' / 'answer-stream'


@unittest.skipUnless(os.environ.get('EVERYTHING_RAG_LIVE_EVAL') == '1', '需要真实模型调用')
class AnswerStreamLiveTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=OUTPUT)
        self.store = await AppStore.open(Path(self.temporary.name) / 'app.db')
        factory = WorkspaceContextFactory(
            Path(self.temporary.name), WorkspaceStore(Path(self.temporary.name)),
            app_store=self.store,
        )
        self.manager = RunManager(store=self.store, context_factory=factory)
        await self.store.create_workspace('default', 'default')
        await self.store.create_conversation('conversation', 'default')

    async def asyncTearDown(self):
        await self.manager.aclose()
        await self.store.close()
        self.temporary.cleanup()

    async def save_events(self, run_id, name):
        events = await self.manager.read_events(run_id)
        (OUTPUT / f'{name}.json').write_text(
            json.dumps([event.model_dump(mode='json') for event in events], ensure_ascii=False),
            encoding='utf-8',
        )
        return events

    async def test_general_answer_arrives_during_generation(self):
        run = await self.manager.create_run(
            workspace_id='default', conversation_id='conversation', reasoning_level='off',
            question='仅使用通用知识，简短解释Python的for循环，用一个Markdown二级标题、两项无序列表和一个保留缩进的三行代码块回答。',
        )
        started = time.monotonic()
        await self.manager.start(run.id)
        first_delta = None
        frames = []
        async with asyncio.timeout(300):
            async for frame in self.manager.stream_events(run.id):
                frames.append(frame)
                if '"type":"answer.delta"' in frame and first_delta is None:
                    first_delta = time.monotonic() - started
                    self.assertEqual((await self.manager.require_run(run.id)).status, RunStatus.RUNNING)
                    self.assertFalse(any(
                        message.role == 'assistant'
                        for message in await self.store.list_messages('conversation')
                    ))
        self.assertIsNotNone(first_delta)
        elapsed = time.monotonic() - started
        result = await self.manager.wait(run.id)
        self.assertEqual(result.status, RunStatus.COMPLETED)
        events = await self.save_events(run.id, 'general')
        drafts = [event for event in events if event.type == 'answer.delta']
        self.assertGreater(len(drafts), 1)
        self.assertTrue(all(not event.payload['draft'] and not event.payload['replace'] for event in drafts))
        self.assertEqual(len({event.payload['answer_id'] for event in drafts}), 1)
        finished_calls = [event for event in events if (
            (event.payload.get('model_call') or {}).get('stage') == 'general_research'
            and event.payload['model_call'].get('ended_at')
        )]
        self.assertLess(drafts[0].event_seq, finished_calls[0].event_seq)
        self.assertFalse(any(event.type == 'model.reasoning' for event in events))
        cursor = drafts[len(drafts) // 2].event_seq
        replay = [frame async for frame in self.manager.stream_events(run.id, after_seq=cursor)]
        self.assertEqual(replay, frames[cursor:])
        artifact = await self.store.get_answer(result.answer_id)
        official = [event for event in events if event.type == 'answer.delta' and not event.payload['draft']]
        self.assertEqual(''.join(event.payload['text'] for event in official), artifact.content)
        last_stream = drafts[-1].payload['answer_id']
        streamed = ''.join(
            event.payload['text'] for event in drafts if event.payload['answer_id'] == last_stream
        )
        self.assertEqual(artifact.content, streamed.strip())
        self.assertIn('\n', artifact.content)
        messages = await self.store.list_messages('conversation')
        self.assertEqual(messages[-1].content, artifact.content)
        print(f'正文首次到达 {first_delta:.2f} 秒，完成 {elapsed:.2f} 秒，增量 {len(drafts)} 条')

    async def test_sourced_answer_streams_readable_paragraphs(self):
        source = ROOT / 'README.md'
        content = source.read_text(encoding='utf-8')
        question = '根据README介绍这个项目的用途和启动方法。'
        run = await self.manager.create_run(
            workspace_id='default', conversation_id='conversation',
            reasoning_level='off', question=question,
        )
        await self.store.update_run_status(run.id, RunStatus.RUNNING)
        digest = content_hash_of(content)
        source_ref = f'{source}#read:0'
        evidence = Evidence(
            evidence_id=evidence_id_for(EvidenceSourceType.KNOWLEDGE, source_ref, digest),
            workspace_id='default', run_id=run.id, source_type=EvidenceSourceType.KNOWLEDGE,
            source_ref=source_ref, content=content, content_hash=digest, source_end=len(content),
            retrieved_at=utc_now_iso(), supported_requirements=['req-readme'],
            document_title=source.name,
        )
        reviews = []
        model = synthesis_model_for_chat(
            reasoning_level='off', on_review=reviews.append,
            on_reasoning=reasoning_publisher(
                self.manager, run.id, 'phase:synthesis', '正在整理回答',
                answer_stages=frozenset({'synthesis_review', 'synthesis_correction'}),
            ),
        )
        requirements = {'req-readme': question}
        candidate = await asyncio.to_thread(
            synthesize_answer, question, [], extra_evidence=[evidence],
            requirements=requirements, workspace_id='default', run_id=run.id, model=model,
        )
        assignment = assign_citation_ids(candidate.content, candidate.evidence)
        candidate = candidate.model_copy(update={
            'content': assignment.content, 'citations': assignment.citations,
        })
        streamed_events = [event for event in await self.manager.read_events(run.id) if event.type == 'answer.delta']
        artifact = publish_answer(
            candidate, requirements=requirements, answer_id=streamed_events[-1].payload['answer_id'],
        )
        await self.store.commit_answer(artifact=artifact)
        events = await self.save_events(run.id, 'sourced')
        drafts = [event for event in events if event.type == 'answer.delta']
        self.assertGreaterEqual(len(drafts), 1)
        self.assertTrue(all(not event.payload['draft'] and not event.payload['replace'] for event in drafts))
        self.assertEqual(''.join(event.payload['text'] for event in drafts), artifact.content)
        self.assertEqual(len({event.payload['answer_id'] for event in drafts}), 1)
        for event in drafts:
            self.assertNotIn('"paragraphs"', event.payload['text'])
            self.assertNotIn('"evidence_id"', event.payload['text'])
        call_end = next(event for event in events if (
            (event.payload.get('model_call') or {}).get('stage') == 'synthesis_review'
            and event.payload['model_call'].get('ended_at')
        ))
        self.assertLess(drafts[0].event_seq, call_end.event_seq)
        self.assertTrue(artifact.citations)
        self.assertIn('[S1]', artifact.content)
        raw = next(item['raw'] for item in reviews if item.get('accepted') is True)
        stream = AnswerStream(evidence=[evidence], question=question)
        emitted = ''
        # 按字符边界重放真实模型响应，检查 JSON 转义和未完成字段。
        for end in range(1, len(raw) + 1):
            delta = stream.update('replay', 'synthesis_review', raw[:end])
            if delta:
                emitted = delta['text'] if delta['replace'] else emitted + delta['text']
        self.assertEqual(emitted, artifact.content)
        self.assertFalse(any((event.payload.get('model_call') or {}).get('stage') == 'semantic_review' for event in events))
        print(f'有来源正文增量 {len(drafts)} 条，引用 {len(artifact.citations)} 项')

    async def test_cancel_stops_answer_deltas_and_resume_completes(self):
        run = await self.manager.create_run(
            workspace_id='default', conversation_id='conversation', reasoning_level='off',
            question='仅使用通用知识，详细解释HTTP缓存，约1500字。',
        )
        await self.manager.start(run.id)
        cancelled = False
        async with asyncio.timeout(300):
            async for frame in self.manager.stream_events(run.id):
                if '"type":"answer.delta"' in frame and not cancelled:
                    cancelled = True
                    await self.manager.cancel(run.id)
        self.assertTrue(cancelled)
        self.assertEqual((await self.manager.require_run(run.id)).status, RunStatus.CANCELLED)
        before_resume = await self.save_events(run.id, 'cancelled')
        self.assertEqual(before_resume[-1].type, 'run.cancelled')
        self.assertFalse(any(
            message.role == 'assistant'
            for message in await self.store.list_messages('conversation')
        ))
        await self.manager.resume(run.id)
        result = await self.manager.wait(run.id, timeout=300)
        self.assertEqual(result.status, RunStatus.COMPLETED)
        events = await self.save_events(run.id, 'resumed')
        drafts = [event for event in events if event.type == 'answer.delta']
        self.assertGreater(len({event.payload['answer_id'] for event in drafts}), 1)

    async def test_recorded_discussion_uses_one_model_call(self):
        path = Path.home() / '.everything-rag' / 'app.db'
        with sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True) as connection:
            history = connection.execute(
                'SELECT role,content FROM messages WHERE conversation_id=? AND sequence<5 ORDER BY sequence',
                ('5af72dddc61b42e9b0871bbe382a8175',),
            ).fetchall()
        self.assertTrue(history)
        for role, content in history:
            await self.store.append_message(
                conversation_id='conversation', role=role, content=content,
            )
        for index, question in enumerate((
            '你有什么补充的说法？', '我说你怎么看货拉拉他们的做法？',
        )):
            run = await self.manager.create_run(
                workspace_id='default', conversation_id='conversation',
                reasoning_level='low', question=question,
            )
            started = time.monotonic()
            await self.manager.start(run.id)
            result = await self.manager.wait(run.id, timeout=300)
            events = await self.save_events(run.id, f'discussion-{index}')
            self.assertEqual(result.status, RunStatus.COMPLETED)
            self.assertFalse(any(event.type == 'tool.started' for event in events))
            calls = [event.payload['model_call'] for event in events if (
                (event.payload.get('model_call') or {}).get('ended_at')
            )]
            self.assertEqual([call['stage'] for call in calls], ['general_research'])
            deltas = [event for event in events if event.type == 'answer.delta']
            self.assertGreater(len(deltas), 1)
            artifact = await self.store.get_answer(result.answer_id)
            self.assertEqual(''.join(event.payload['text'] for event in deltas), artifact.content)
            self.assertTrue(all(event.payload['answer_id'] == artifact.answer_id for event in deltas))
            self.assertTrue(all(not event.payload['replace'] and not event.payload['draft'] for event in deltas))
            print(f'真实追问 {index + 1}：{time.monotonic() - started:.2f} 秒，模型调用 {len(calls)} 次')

    async def test_requested_verification_finishes_before_publication(self):
        run = await self.manager.create_run(
            workspace_id='default', conversation_id='conversation', reasoning_level='off',
            question='仅使用通用知识，核实“Python的list支持索引”这个说法是否正确，简短回答。',
        )
        await self.manager.start(run.id)
        result = await self.manager.wait(run.id, timeout=300)
        events = await self.save_events(run.id, 'verified-general')
        self.assertEqual(result.status, RunStatus.COMPLETED)
        calls = [event for event in events if (event.payload.get('model_call') or {}).get('ended_at')]
        self.assertEqual([event.payload['model_call']['stage'] for event in calls], [
            'general_research', 'general_verification',
        ])
        deltas = [event for event in events if event.type == 'answer.delta']
        self.assertGreater(deltas[0].event_seq, calls[-1].event_seq)
        artifact = await self.store.get_answer(result.answer_id)
        self.assertEqual(''.join(event.payload['text'] for event in deltas), artifact.content)


if __name__ == '__main__':
    unittest.main()
