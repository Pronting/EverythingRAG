import asyncio
import socket

import pytest
from openai import APIConnectionError, AsyncOpenAI

from app.generation.providers import OpenAICompatChatModel, exception_type_chain


class ObservedProvider(OpenAICompatChatModel):
    def __init__(self, base_url: str) -> None:
        super().__init__(base_url=base_url, model="resource-check", timeout=5.0)
        self.clients: list[AsyncOpenAI] = []

    def _new_client(self) -> AsyncOpenAI:
        client = super()._new_client()
        self.clients.append(client)
        return client


def test_construction_creates_no_http_client() -> None:
    provider = ObservedProvider("http://127.0.0.1:1/v1")
    assert provider.clients == []


def test_owned_clients_close_normally_and_on_cancellation() -> None:
    async def check() -> None:
        provider = ObservedProvider("http://127.0.0.1:1/v1")
        async with provider._new_client() as first:
            assert not first.is_closed()
        assert first.is_closed()
        entered = asyncio.Event()

        async def wait_with_client() -> None:
            async with provider._new_client():
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(wait_with_client())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.clients[1].is_closed()
        assert provider.clients[0] is not provider.clients[1]

    asyncio.run(check())


@pytest.mark.parametrize("method", ["stream_chat", "stream_result", "complete"])
def test_connection_failure_closes_owned_client(method: str) -> None:
    # 端口由真实 socket 保留且未监听，连接会被本机拒绝。
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]

        async def check() -> None:
            provider = ObservedProvider(f"http://127.0.0.1:{port}/v1")
            with pytest.raises(APIConnectionError) as failed:
                result = getattr(provider, method)([{"role": "user", "content": "resource check"}])
                if method == "complete":
                    await result
                else:
                    async for _ in result:
                        pass
            assert len(provider.clients) == 1
            assert provider.clients[0].is_closed()
            names = exception_type_chain(failed.value)
            assert names[0] == "openai.APIConnectionError"
            assert "httpx.ConnectError" in names
            assert all("127.0.0.1" not in name for name in names)

        asyncio.run(check())


def test_exception_chain_respects_suppressed_context() -> None:
    with pytest.raises(ValueError) as failed:
        try:
            int("private input")
        except ValueError:
            raise ValueError("private credential") from None
    assert exception_type_chain(failed.value) == ["builtins.ValueError"]


def test_exception_chain_bounds_cycles() -> None:
    error = ValueError("private credential")
    error.__cause__ = error
    assert exception_type_chain(error) == ["builtins.ValueError"]
