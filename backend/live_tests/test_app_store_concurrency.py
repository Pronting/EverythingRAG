from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest

from app.agent.app_store import AppStore
from app.agent.contracts import OperationStatus
from app.core.workspace_context import _shared_paths
from app.memory.history import HistorySearch, WorkspaceHistoryIndex


@pytest.mark.asyncio
async def test_transaction_cancellation_keeps_other_task_write():
    directory = Path(__file__).resolve().parents[2] / ".tmp_probe" / "sqlite-concurrency"
    directory.mkdir(parents=True, exist_ok=True)
    store = await AppStore.open(directory / f"{uuid4().hex}.db")
    entered = asyncio.Event()
    writing = asyncio.Event()

    async def cancelled_transaction():
        async with store.transaction():
            await store.create_workspace("cancelled", "cancelled")
            entered.set()
            await asyncio.Event().wait()

    async def independent_write():
        await entered.wait()
        writing.set()
        await store.create_workspace("independent", "independent")

    try:
        first = asyncio.create_task(cancelled_transaction())
        second = asyncio.create_task(independent_write())
        await writing.wait()
        await asyncio.sleep(0.05)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        assert await store.get_workspace("cancelled") is None
        assert await store.get_workspace("independent") is not None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_parallel_tool_operation_writes_with_transactions():
    directory = Path(__file__).resolve().parents[2] / ".tmp_probe" / "sqlite-concurrency"
    directory.mkdir(parents=True, exist_ok=True)
    store = await AppStore.open(directory / f"{uuid4().hex}.db")

    async def operation(index: int):
        operation_id = f"read-{index}"
        await store.register_operation(
            operation_id=operation_id, tool_name="read_document", args_hash=str(index),
        )
        await store.start_operation_attempt(operation_id)
        await store.finish_operation(operation_id, OperationStatus.SUCCEEDED)
        record = await store.get_operation(operation_id)
        assert record is not None
        assert record.attempts == 1
        assert record.status == OperationStatus.SUCCEEDED

    async def transaction(index: int):
        async with store.transaction():
            await store.create_workspace(str(index), str(index))
            await asyncio.sleep(0)

    try:
        await asyncio.gather(
            *(operation(index) for index in range(50)),
            *(transaction(index) for index in range(20)),
        )
        assert len(await store.list_workspaces()) == 20
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_parallel_transactions_and_nested_calls():
    directory = Path(__file__).resolve().parents[2] / ".tmp_probe" / "sqlite-concurrency"
    directory.mkdir(parents=True, exist_ok=True)
    store = await AppStore.open(directory / f"{uuid4().hex}.db")

    async def write(index: int):
        async with store.transaction():
            async with store.transaction():
                await store.create_workspace(str(index), str(index))
                await asyncio.sleep(0)
                assert await store.get_workspace(str(index)) is not None

    try:
        await asyncio.gather(*(write(index) for index in range(30)))
        assert len(await store.list_workspaces()) == 30
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_history_index_maintenance_and_message_writes():
    directory = Path(__file__).resolve().parents[2] / ".tmp_probe" / "sqlite-concurrency" / uuid4().hex
    directory.mkdir(parents=True, exist_ok=True)
    store = await AppStore.open(directory / "app.db")
    paths = _shared_paths(directory, "history")
    index = WorkspaceHistoryIndex(store, workspace_id="history", paths=paths)
    try:
        await store.create_workspace("history", "history")
        await store.create_conversation("conversation", "history")
        await asyncio.gather(
            *(store.append_message(
                conversation_id="conversation", role="user", content=f"并发历史记录编号 {number}",
            ) for number in range(20)),
            index.rebuild(), index.prune_orphans(), index._has_index_difference(),
        )
        assert await index._has_index_difference() is False
        assert await store.history_index_size() == 20
        messages = await store.list_messages("conversation")
        assert [message.sequence for message in messages] == list(range(1, 21))
        with HistorySearch(directory / "app.db", workspace_id="history", paths=paths) as search:
            assert len(search.search("并发历史记录", limit=50)) == 20
    finally:
        await store.close()
