import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from app.agent.app_store import AppStore
from app.agent.contracts import ReasoningLevel
from app.agent.runtime import ApprovalGate, RunManager
from app.agent.tools import ToolContext, build_default_registry
from app.api import deps


class ProcessStreamTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.output = Path(__file__).resolve().parents[2] / '.tmp_probe' / 'process-stream'
        self.output.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=self.output)
        self.store = await AppStore.open(Path(self.temporary.name) / 'app.db')
        await self.store.create_workspace('workspace', '过程验证')
        await self.store.create_conversation('conversation', 'workspace')
        self.manager = RunManager(store=self.store)

    async def asyncTearDown(self):
        await self.manager.aclose()
        await self.store.close()
        self.temporary.cleanup()

    async def save_events(self, run_id, name):
        events = await self.manager.read_events(run_id)
        payload = [event.model_dump(mode='json') for event in events]
        (self.output / name).write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        return events

    async def test_real_model_stream_and_replay(self):
        run = await self.manager.create_run(
            workspace_id='workspace', conversation_id='conversation',
            question='请用通用知识解释天空为什么是蓝色，每个问题回答一句话；请用通用知识解释海水有时为什么呈绿色。',
            reasoning_level=ReasoningLevel.LOW,
        )
        await self.manager.start(run.id)
        stream = self.manager.stream_events(run.id)
        reasoning_before_answer = False
        saw_answer = False
        async with asyncio.timeout(180):
            async for frame in stream:
                if '"type":"model.reasoning"' in frame or '"type": "model.reasoning"' in frame:
                    self.assertFalse(saw_answer)
                    reasoning_before_answer = True
                if 'answer.delta' in frame:
                    saw_answer = True
        events = await self.save_events(run.id, 'model-events.json')
        errors = [event.payload for event in events if event.type == 'run.failed']
        self.assertFalse(errors, errors)
        self.assertTrue(reasoning_before_answer)
        self.assertTrue(saw_answer)
        reasoning = [event for event in events if event.type == 'model.reasoning']
        self.assertGreater(len(reasoning), 1)
        self.assertTrue(all(event.payload['text'] for event in reasoning))
        self.assertIn('phase:synthesis', {event.payload['step_id'] for event in reasoning})
        task_ids = {event.task_id for event in reasoning if event.task_id is not None}
        self.assertGreaterEqual(len(task_ids), 2)
        for stream_id in {event.payload['stream_id'] for event in reasoning}:
            self.assertEqual(len({event.task_id for event in reasoning if event.payload['stream_id'] == stream_id}), 1)
        replay = await self.manager.read_events(run.id, after_seq=reasoning[0].event_seq)
        self.assertEqual(
            [event.model_dump() for event in replay],
            [event.model_dump() for event in events if event.event_seq > reasoning[0].event_seq],
        )
        await self.store.create_conversation('off', 'workspace')
        off = await self.manager.create_run(
            workspace_id='workspace', conversation_id='off',
            question='请用一句话解释彩虹形成的原因，只使用通用知识。',
            reasoning_level=ReasoningLevel.OFF,
        )
        finished = await self.manager.run_to_completion(off.id, timeout=180)
        self.assertEqual(str(finished.status), 'completed')
        self.assertFalse(any(event.type == 'model.reasoning' for event in await self.manager.read_events(off.id)))

    async def test_real_document_read_results(self):
        context = deps.get_workspace_context_factory().resolve()
        vector_store = deps.get_workspace_vector_store(context)
        blocks = vector_store.list_blocks()
        block_id, _text, metadata = next(
            block for block in blocks if '12-openclaw' in block[0] and len(block[1]) > 500
        )
        run = await self.store.create_run(conversation_id='conversation', workspace_id='workspace')
        tool_context = ToolContext(workspace=context, vector_store=vector_store, run_id=run.id)
        gate = ApprovalGate(manager=self.manager, registry=build_default_registry(), run=run, state={})
        result = await gate.execute('read_document', {
            'document_id': block_id.split('::', 1)[0],
            'heading': metadata['heading_path'],
        }, tool_context)
        self.assertTrue(result.data['text'])
        events = await self.save_events(run.id, 'document-events.json')
        completed = events[-1].payload
        self.assertIn('已读取', completed['summary'])
        self.assertEqual(completed['sources'][0]['text'], result.data['text'])
        self.assertNotIn('未找到', completed['summary'])
        await gate.execute('read_document', {
            'document_id': block_id.split('::', 1)[0],
            'heading': '过程验证中不存在的章节名称',
        }, tool_context)
        empty_events = await self.manager.read_events(run.id)
        self.assertEqual(empty_events[-1].payload['summary'], '该文档没有匹配的章节')


if __name__ == '__main__':
    unittest.main()
