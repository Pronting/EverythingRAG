import asyncio
import logging
import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import uuid4

from app.agent.runtime import (
    _EXECUTION_DEADLINE,
    _comparison_requires_joint_reasoning,
    _run_general_blocking,
)
from app.agent.tools import ToolUnavailableError
from app.core.settings_store import get_settings_store
from app.generation.model_runtime import MODEL_OWNER, model_runtime
from app.vectorstore.embedder import create_embedder_from_config


class EmbeddingEvents(logging.Handler):
    def __init__(self):
        super().__init__()
        self.requests = []

    def emit(self, record):
        if record.getMessage().startswith("query_embedding_timing "):
            self.requests.append(record.getMessage())


@unittest.skipUnless(os.environ.get("EVERYTHING_RAG_LIVE_EVAL") == "1", "需要真实模型调用")
class ResponseOptimizationTests(unittest.IsolatedAsyncioTestCase):
    def test_joint_comparison_policy(self):
        self.assertFalse(_comparison_requires_joint_reasoning(
            "比较货拉拉、TencentDB Agent Memory 和 OpenClaw 的存储、召回、更新与适用条件。",
        ))
        for question in [
            "用同一张表格比较三种记忆设计的存储、召回和更新。",
            "比较这三种记忆设计，判断哪种更适合多实例团队服务，并解释理由。",
            "请用一段话、最多500字比较三种记忆设计。",
            "Compare these designs and calculate the monthly cost for each workload.",
        ]:
            with self.subTest(question=question):
                self.assertTrue(_comparison_requires_joint_reasoning(question))

    async def test_parallel_model_streams(self):
        config = get_settings_store().load().chat
        owner = f"parallel-test-{uuid4().hex}"
        token = MODEL_OWNER.set(owner)
        try:
            answers = await asyncio.gather(*(
                asyncio.to_thread(
                    _run_general_blocking, config,
                    f"只回答：你好，编号{index}。", "low", max_tokens=2048,
                ) for index in range(3)
            ))
            for index, answer in enumerate(answers):
                self.assertIn(str(index), answer)
        finally:
            await model_runtime.close_owner(owner)
            MODEL_OWNER.reset(token)

    async def test_exact_embedding_cache_and_shared_request(self):
        embedder = create_embedder_from_config(get_settings_store().load().embed)
        text = f"查询记忆系统的召回职责，校验标识{uuid4().hex}"
        barrier = Barrier(2)
        events = EmbeddingEvents()
        logger = logging.getLogger("app.vectorstore.embedder")
        logger.addHandler(events)

        def request():
            barrier.wait(timeout=10)
            return embedder.embed_queries([text, text])

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(request)
                second = executor.submit(request)
                left, right = first.result(timeout=60), second.result(timeout=60)
            self.assertEqual(left, right)
            self.assertEqual(left[0], left[1])
            self.assertGreater(len(left[0]), 0)
            self.assertEqual(len(events.requests), 1)
            expected = list(left[0])
            left[0][0] += 1
            self.assertEqual(embedder.embed_query(text), expected)
            self.assertEqual(len(events.requests), 1)
        finally:
            logger.removeHandler(events)

    async def test_model_connection_reuse_deadline_and_close(self):
        config = get_settings_store().load().chat
        owner = f"live-test-{uuid4().hex}"
        token = MODEL_OWNER.set(owner)
        client = None
        try:
            answer = await asyncio.to_thread(
                _run_general_blocking, config, "只回答：你好。", "off", max_tokens=64,
            )
            self.assertIn("你好", answer)

            async def connection():
                return model_runtime.model(config)._shared_client

            client = await asyncio.wrap_future(model_runtime.submit(connection))
            self.assertIsNotNone(client)
            answer = await asyncio.to_thread(
                _run_general_blocking, config, "只回答：你好。", "off", max_tokens=64,
            )
            self.assertIn("你好", answer)
            self.assertIs(client, await asyncio.wrap_future(model_runtime.submit(connection)))
            deadline = _EXECUTION_DEADLINE.set(time.monotonic() + 0.15)
            started = time.monotonic()
            try:
                with self.assertRaises(ToolUnavailableError):
                    await asyncio.to_thread(
                        _run_general_blocking, config, "详细解释数据库事务。", "off",
                    )
                self.assertLess(time.monotonic() - started, 3)
            finally:
                _EXECUTION_DEADLINE.reset(deadline)
            answer = await asyncio.to_thread(
                _run_general_blocking, config, "只回答：你好。", "off", max_tokens=64,
            )
            self.assertIn("你好", answer)
        finally:
            await model_runtime.close_owner(owner)
            MODEL_OWNER.reset(token)
        self.assertTrue(client.is_closed())


if __name__ == "__main__":
    unittest.main()
