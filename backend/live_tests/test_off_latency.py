import asyncio
import os
import tempfile
import time
import unittest
from pathlib import Path

from app.agent.app_store import AppStore
from app.agent.contracts import ReasoningLevel, RunStatus
from app.agent.runtime import (
    _EXECUTION_DEADLINE,
    GENERAL_MODEL_TOTAL_TIMEOUT_SECONDS,
    RUN_ACTIVE_TIMEOUT_SECONDS,
    ModelCallTiming,
    RunManager,
    _chat_text,
    _run_general_blocking,
    general_review_required,
    model_time_remaining,
    simple_general_route,
)
from app.agent.tools import ToolUnavailableError
from app.core.settings_store import get_settings_store
from app.core.workspace_context import WorkspaceContextFactory
from app.core.workspace_store import WorkspaceStore
from app.generation.providers import (
    OpenAICompatChatModel,
    create_chat_model_from_config,
)
from app.generation.tokenizer import PromptCounter, official_message_payload


class OffPolicyTests(unittest.TestCase):
    def test_review_selection_respects_question_and_level(self):
        for level in ReasoningLevel:
            with self.subTest(level=level):
                self.assertFalse(general_review_required('什么叫做银弹？', level))
                self.assertFalse(general_review_required('解释能够保证可靠性的技术。', level))
                self.assertTrue(general_review_required('核查这份回答是否正确。', level))
        self.assertFalse(general_review_required('比较两种方法的适用条件。', 'off'))
        self.assertTrue(general_review_required('比较两种方法的适用条件。', 'high'))

    def test_definition_route_preserves_history_and_source_requirements(self):
        self.assertTrue(simple_general_route('什么叫做银弹？', [], ReasoningLevel.OFF))
        for question in ['银弹是否能解决问题？', '联网解释银弹。', '解释银弹以及它的适用条件。']:
            with self.subTest(question=question):
                self.assertFalse(simple_general_route(question, [], 'off'))
        self.assertFalse(simple_general_route('解释这个意思。', ['上一轮问题'], 'off'))
        self.assertFalse(simple_general_route('什么叫做银弹？', [], 'high'))

    def test_off_request_contains_disabled_parameter(self):
        for endpoint, model in [
            ('https://api.deepseek.com', 'deepseek-flash'),
            ('https://api.commandcode.ai/provider/v1', 'deepseek/deepseek-v4.1-flash'),
        ]:
            provider = OpenAICompatChatModel(endpoint, model)
            request = provider.build_request(
                [{'role': 'user', 'content': '解释银弹。'}], stream=True, reasoning_level='off',
            )
            self.assertEqual(request['extra_body']['thinking'], {'type': 'disabled'})
            if 'commandcode.ai' in endpoint:
                self.assertEqual(request['reasoning_effort'], 'off')
            else:
                self.assertNotIn('reasoning_effort', request)
            self.assertNotIn('max_tokens', request)
            limited = provider.build_request(
                [{'role': 'user', 'content': '解释银弹。'}], stream=True,
                reasoning_level='off', max_tokens=8192,
            )
            self.assertEqual(limited['max_tokens'], 8192)

    def test_shared_deadline_and_review_limits(self):
        for level in ReasoningLevel:
            with self.subTest(level=level):
                self.assertEqual(model_time_remaining(level), GENERAL_MODEL_TOTAL_TIMEOUT_SECONDS)
        self.assertEqual(model_time_remaining('off', 'general_verification'), 10)
        self.assertEqual(model_time_remaining('high', 'general_verification'), 30)
        started = time.monotonic()
        deadline = started + 0.05
        token = _EXECUTION_DEADLINE.set(deadline)
        try:
            self.assertLessEqual(model_time_remaining('off'), deadline - started)
            self.assertLessEqual(model_time_remaining('high'), deadline - started)
        finally:
            _EXECUTION_DEADLINE.reset(token)
        expired = _EXECUTION_DEADLINE.set(time.monotonic() - 1)
        try:
            with self.assertRaises(TimeoutError):
                model_time_remaining('off')
        finally:
            _EXECUTION_DEADLINE.reset(expired)

    def test_gateway_off_uses_official_disabled_encoding(self):
        model = OpenAICompatChatModel(
            'https://api.commandcode.ai/provider/v1', 'deepseek/deepseek-v4.1-flash',
        )
        request = model.build_request(
            [{'role': 'user', 'content': '核对这段正文。'}], stream=True, reasoning_level='off',
        )
        counter = PromptCounter()
        encoding = counter.encode(request)
        official = {**request, 'reasoning_effort': 'none'}
        self.assertEqual(encoding.token_ids, counter.encode(official).token_ids)
        self.assertGreater(encoding.token_count, 0)
        self.assertEqual(official_message_payload(request)['reasoning_effort'], 'none')
        self.assertEqual(request['reasoning_effort'], 'off')
        self.assertEqual(model.context_budget(request).local_prompt_tokens, encoding.token_count)


@unittest.skipUnless(os.environ.get('EVERYTHING_RAG_LIVE_EVAL') == '1', '需要真实模型调用')
class OffProviderLiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_provider_off_complete_answers(self):
        model = create_chat_model_from_config(get_settings_store().load().chat)
        async with asyncio.timeout(GENERAL_MODEL_TOTAL_TIMEOUT_SECONDS):
            result = await model.complete(
                [{'role': 'user', 'content': '只回答：你好。'}],
                reasoning_level='off', max_tokens=2048,
            )
            self.assertIn('你好', result.content)
            self.assertEqual(result.finish_reason, 'stop')
            self.assertEqual(result.reasoning_content, '')
            self.assertEqual(result.usage.reasoning_tokens, 0)

    async def test_real_provider_enabled_thinking_still_answers(self):
        model = create_chat_model_from_config(get_settings_store().load().chat)
        content = ''
        async with asyncio.timeout(30):
            async for item in model.stream_result(
                [{'role': 'user', 'content': '只回答：你好。'}],
                reasoning_level='low', max_tokens=2048,
            ):
                if getattr(item, 'kind', '') == 'content':
                    content += item.text
                if hasattr(item, 'response'):
                    self.assertTrue(item.succeeded)
        self.assertIn('你好', content)

    async def test_real_provider_off_stream_answers(self):
        config = get_settings_store().load().chat
        model = create_chat_model_from_config(config)
        for method in ['stream_result', 'stream_chat']:
            started = time.monotonic()
            with self.subTest(method=method):
                content = ''
                async with asyncio.timeout(GENERAL_MODEL_TOTAL_TIMEOUT_SECONDS):
                    async for item in getattr(model, method)(
                        [{'role': 'user', 'content': '用一句话解释银弹。'}],
                        reasoning_level='off', max_tokens=2048,
                    ):
                        if getattr(item, 'kind', '') == 'content':
                            content += item.text
                        if hasattr(item, 'response'):
                            self.assertTrue(item.succeeded)
                            self.assertEqual(item.response.reasoning_content, '')
                            self.assertEqual(item.response.usage.reasoning_tokens, 0)
                self.assertIn('银弹', content)
                self.assertLess(time.monotonic() - started, GENERAL_MODEL_TOTAL_TIMEOUT_SECONDS)

    async def test_real_model_cancelled_at_shared_deadline(self):
        config = get_settings_store().load().chat
        timing = ModelCallTiming(stage='general_research')
        token = _EXECUTION_DEADLINE.set(time.monotonic() + 0.2)
        started = time.monotonic()
        try:
            with self.assertRaises(TimeoutError):
                await _chat_text(config, '解释银弹。', 'off', timing=timing)
            self.assertLess(time.monotonic() - started, 2)
            self.assertTrue(timing.finished_event.is_set())
            self.assertEqual(timing.outcome, 'stream_total_timeout')
        finally:
            _EXECUTION_DEADLINE.reset(token)

    async def test_worker_inherits_shared_deadline(self):
        config = get_settings_store().load().chat
        token = _EXECUTION_DEADLINE.set(time.monotonic() + 0.2)
        started = time.monotonic()
        try:
            with self.assertRaises(ToolUnavailableError):
                await asyncio.to_thread(_run_general_blocking, config, '解释银弹。', 'off')
            self.assertLess(time.monotonic() - started, 2)
        finally:
            _EXECUTION_DEADLINE.reset(token)

    async def test_real_graph_does_not_route_or_review_simple_definition(self):
        directory = Path(__file__).resolve().parents[2] / '.tmp_probe'
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            store = await AppStore.open(Path(temporary) / 'app.db')
            workspace_store = WorkspaceStore(Path(temporary))
            factory = WorkspaceContextFactory(Path(temporary), workspace_store, app_store=store)
            manager = RunManager(store=store, context_factory=factory)
            try:
                await store.create_workspace('default', 'default')
                await store.create_conversation('conversation', 'default')
                run = await manager.create_run(
                    workspace_id='default', conversation_id='conversation',
                    question='什么叫做银弹？', reasoning_level='off',
                )
                started = time.monotonic()
                result = await manager.run_to_completion(run.id, timeout=RUN_ACTIVE_TIMEOUT_SECONDS)
                elapsed = time.monotonic() - started
                self.assertLess(elapsed, RUN_ACTIVE_TIMEOUT_SECONDS)
                self.assertEqual(result.status, RunStatus.COMPLETED)
                events = await manager.read_events(run.id)
                stages = {
                    event.payload['model_call']['stage'] for event in events
                    if event.payload.get('model_call')
                }
                self.assertEqual(stages, {'general_research'})
                answer = next(event for event in events if event.type == 'answer.meta')
                self.assertEqual(answer.payload['answer_basis'], 'general')
                messages = await store.list_messages('conversation')
                self.assertIn('银弹', messages[-1].content)
                print(f'真实流程：{elapsed:.2f} 秒，状态 {result.status.value}')
            finally:
                await manager.aclose()
                await store.close()

    async def test_real_graph_completes_long_answer_with_thinking_off(self):
        directory = Path(__file__).resolve().parents[2] / '.tmp_probe'
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            store = await AppStore.open(Path(temporary) / 'app.db')
            workspace_store = WorkspaceStore(Path(temporary))
            factory = WorkspaceContextFactory(Path(temporary), workspace_store, app_store=store)
            manager = RunManager(store=store, context_factory=factory)
            try:
                await store.create_workspace('default', 'default')
                await store.create_conversation('conversation', 'default')
                run = await manager.create_run(
                    workspace_id='default', conversation_id='conversation',
                    question=(
                        '什么叫做数据库事务？请仅使用通用知识，用约2500个汉字详细解释ACID、'
                        '隔离级别、并发控制、日志恢复以及转账案例，每个主题都提供具体例子。'
                    ),
                    reasoning_level='off',
                )
                started = time.monotonic()
                result = await manager.run_to_completion(run.id, timeout=RUN_ACTIVE_TIMEOUT_SECONDS)
                elapsed = time.monotonic() - started
                events = await manager.read_events(run.id)
                errors = [event.payload for event in events if event.type == 'run.failed']
                self.assertFalse(errors, errors)
                self.assertEqual(result.status, RunStatus.COMPLETED)
                self.assertFalse(any(event.type == 'model.reasoning' for event in events))
                messages = await store.list_messages('conversation')
                self.assertEqual(messages[-1].role, 'assistant')
                self.assertGreater(len(messages[-1].content), 1000)
                print(f'关闭思考的长回答：{elapsed:.2f} 秒，状态 {result.status.value}')
            finally:
                await manager.aclose()
                await store.close()


if __name__ == '__main__':
    unittest.main()
