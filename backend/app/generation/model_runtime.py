import asyncio
import hashlib
import json
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from threading import Event, Lock, Thread
from typing import Any

from app.agent.contracts import DEFAULT_MODEL_REQUEST_CONCURRENCY
from app.generation import providers

MODEL_OWNER: ContextVar[str] = ContextVar("model_owner", default="")


class ModelRuntime:
    def __init__(self) -> None:
        self._lock = Lock()
        self._ready = Event()
        self._thread: Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._gate: asyncio.Semaphore | None = None
        self._models: dict[tuple[str, str], Any] = {}
        self._selections: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._tasks: dict[str, set[asyncio.Task[Any]]] = {}

    def selection_key(self, stage: str, config: Any, messages: Any, level: Any) -> tuple[str, str] | None:
        owner = MODEL_OWNER.get()
        if not owner or stage not in {"section_selection", "history_selection"}:
            return None
        payload = config.model_dump(mode="json")
        if config.api_key is not None:
            payload["api_key"] = config.api_key.get_secret_value()
        fingerprint = hashlib.sha256(json.dumps(
            [payload, messages, str(level)], ensure_ascii=False, sort_keys=True,
        ).encode()).hexdigest()
        return owner, fingerprint

    def selection(self, key: tuple[str, str] | None) -> str | None:
        if key is None or key not in self._selections:
            return None
        self._selections.move_to_end(key)
        return self._selections[key]

    def remember_selection(self, key: tuple[str, str] | None, content: str) -> None:
        if key is None:
            return
        self._selections[key] = content
        self._selections.move_to_end(key)
        while len(self._selections) > 256:
            self._selections.popitem(last=False)

    def _serve(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._gate = asyncio.Semaphore(DEFAULT_MODEL_REQUEST_CONCURRENCY)
        self._ready.set()
        self._loop.run_forever()

    def submit(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        with self._lock:
            if self._thread is None:
                self._thread = Thread(target=self._serve, name="model-runtime", daemon=True)
                self._thread.start()
        self._ready.wait()

        async def guarded() -> Any:
            owner = MODEL_OWNER.get()
            task = asyncio.current_task()
            self._tasks.setdefault(owner, set()).add(task)
            try:
                async with self._gate:
                    return await operation()
            finally:
                self._tasks[owner].discard(task)
                if not self._tasks[owner]:
                    del self._tasks[owner]

        return asyncio.run_coroutine_threadsafe(guarded(), self._loop)

    def model(self, config: Any) -> Any:
        owner = MODEL_OWNER.get()
        if not owner or asyncio.get_running_loop() is not self._loop:
            return providers.create_chat_model_from_config(config)
        payload = config.model_dump(mode="json")
        if config.api_key is not None:
            payload["api_key"] = config.api_key.get_secret_value()
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        key = (owner, fingerprint)
        if key not in self._models:
            model = providers.create_chat_model_from_config(config)
            if isinstance(model, providers.OpenAICompatChatModel):
                model.enable_client_reuse()
            self._models[key] = model
        return self._models[key]

    async def close_owner(self, owner: str) -> None:
        if self._loop is None:
            return

        async def close() -> None:
            active = tuple(self._tasks.get(owner, ()))
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)
            for key in [key for key in self._selections if key[0] == owner]:
                del self._selections[key]
            for key in [key for key in self._models if key[0] == owner]:
                model = self._models.pop(key)
                if isinstance(model, providers.OpenAICompatChatModel):
                    await model.aclose()

        await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(close(), self._loop))


model_runtime = ModelRuntime()
