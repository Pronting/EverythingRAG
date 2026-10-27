import asyncio
import logging
import math
import os
import socket
import unittest
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from openai import APIConnectionError
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

from app.core.settings_store import get_settings_store
from app.generation.base import StreamResult
from app.generation.providers import OpenAICompatChatModel, create_chat_model_from_config
from app.vectorstore.embedder import create_embedder_from_config


class RetryObserver(logging.Handler):
    def __init__(self, loop):
        super().__init__()
        self.loop = loop
        self.ready = asyncio.Event()
        self.count = 0

    def emit(self, record):
        if record.msg.startswith('provider_connection_retry'):
            self.count += 1
            self.loop.call_soon_threadsafe(self.ready.set)


@asynccontextmanager
async def observe_retries():
    observer = RetryObserver(asyncio.get_running_loop())
    logger = logging.getLogger('app.core.provider_retry')
    logger.addHandler(observer)
    try:
        yield observer
    finally:
        logger.removeHandler(observer)


@asynccontextmanager
async def restored_provider(base_url):
    # 首次连接使用真实未监听端口；观察到重试后启动转发服务，响应全部来自真实模型。
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
    app = FastAPI()
    requests = []
    async with httpx.AsyncClient(timeout=90) as upstream:
        @app.post('/{path:path}')
        async def forward(path: str, request: Request):
            requests.append(path)
            outgoing = upstream.build_request(
                'POST', base_url.rstrip('/') + '/' + path,
                content=await request.body(),
                headers={
                    'authorization': request.headers['authorization'],
                    'content-type': 'application/json',
                },
            )
            response = await upstream.send(outgoing, stream=True)
            return StreamingResponse(
                response.aiter_bytes(), status_code=response.status_code,
                media_type=response.headers.get('content-type'),
                background=BackgroundTask(response.aclose),
            )

        server = uvicorn.Server(uvicorn.Config(app, log_level='error', lifespan='off'))
        async with observe_retries() as observer:
            async def restore():
                await observer.ready.wait()
                await server.serve(sockets=[listener])

            serving = asyncio.create_task(restore())
            try:
                yield f'http://127.0.0.1:{port}', observer, requests
            finally:
                if observer.ready.is_set():
                    server.should_exit = True
                    await asyncio.wait_for(serving, 5)
                else:
                    serving.cancel()
                    await asyncio.gather(serving, return_exceptions=True)
                listener.close()


class ConnectionFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_persistent_connection_failure_has_three_attempts(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(('127.0.0.1', 0))
            provider = OpenAICompatChatModel(
                f'http://127.0.0.1:{listener.getsockname()[1]}', 'connection-check', timeout=15,
            )
            async with observe_retries() as observer:
                with self.assertRaises(APIConnectionError):
                    await provider.complete([{'role': 'user', 'content': '连接检查'}])
                self.assertEqual(observer.count, 2)

    async def test_cancellation_interrupts_retry_wait(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(('127.0.0.1', 0))
            provider = OpenAICompatChatModel(
                f'http://127.0.0.1:{listener.getsockname()[1]}', 'connection-check', timeout=15,
            )
            async with observe_retries() as observer:
                task = asyncio.create_task(provider.complete([{'role': 'user', 'content': '连接检查'}]))
                await asyncio.wait_for(observer.ready.wait(), 10)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(observer.count, 1)

    async def test_request_deadline_limits_connection_wait(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(('127.0.0.1', 0))
            provider = OpenAICompatChatModel(
                f'http://127.0.0.1:{listener.getsockname()[1]}', 'connection-check', timeout=0.1,
            )
            async with observe_retries() as observer:
                with self.assertRaises((TimeoutError, APIConnectionError)):
                    await asyncio.wait_for(
                        provider.complete([{'role': 'user', 'content': '连接检查'}]), 2,
                    )
                self.assertEqual(observer.count, 0)


@unittest.skipUnless(os.environ.get('EVERYTHING_RAG_LIVE_EVAL') == '1', '需要真实模型调用')
class ConnectionRecoveryLiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_recovers_before_answer_and_finishes_once(self):
        config = get_settings_store().load().chat
        original = create_chat_model_from_config(config)
        async with restored_provider(config.base_url) as (url, observer, requests):
            provider = OpenAICompatChatModel(
                url, config.model,
                api_key=config.api_key.get_secret_value() if config.api_key else None,
                capabilities=original.capabilities, timeout=60,
            )
            chunks = [chunk async for chunk in provider.stream_result(
                [{'role': 'user', 'content': '请用一句话说明HTTP缓存的作用。'}],
                max_tokens=256,
            )]
            results = [chunk for chunk in chunks if isinstance(chunk, StreamResult)]
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].succeeded, results[0].failure_reasons)
            self.assertTrue(results[0].response.content.strip())
            self.assertEqual(observer.count, 1)
            self.assertEqual(requests, ['chat/completions'])
            print('真实模型流式请求：连接恢复，完整响应一次', flush=True)

    async def test_embedding_recovers_and_returns_real_vectors(self):
        config = get_settings_store().load().embed
        async with restored_provider(config.base_url) as (url, observer, requests):
            embedder = create_embedder_from_config(config.model_copy(update={'base_url': url}))
            try:
                vector = await asyncio.to_thread(embedder.embed_query, '货拉拉 AI native 需求管理平台')
                self.assertEqual(len(vector), embedder.dim)
                self.assertTrue(all(math.isfinite(value) for value in vector))
                self.assertEqual(observer.count, 1)
                self.assertEqual(requests, ['embeddings'])
                print('真实嵌入请求：连接恢复，向量维度', len(vector), flush=True)
            finally:
                embedder._ensure_client().close()
