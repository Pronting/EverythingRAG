import asyncio
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


async def initialize(mode):
    from app.agent.app_store import AppStore
    from app.api import deps
    from app.core.config import settings
    from app.core.workspace_context import commit_storage_version
    from app.main import _initialize_history_indexes

    store = await AppStore.open(settings.data_dir / 'app.db')
    deps.register_app_store(store)
    try:
        await store.create_workspace('default', 'default')
        await store.create_workspace('history-check', 'history-check')
        if mode == 'migrated':
            commit_storage_version(settings.data_dir)
        await _initialize_history_indexes(store)
        factory = deps.get_workspace_context_factory()
        assert factory.resolve('default').workspace_id == 'default'
        expected = {'default', 'history-check'} if mode == 'migrated' else {'default'}
        assert {item.id for item in factory.workspace_store.list()} == expected
        if mode == 'migrated':
            assert factory.resolve('history-check').workspace_id == 'history-check'
    finally:
        deps.register_app_store(None)
        await store.close()


class HistoryStartupLayoutTests(unittest.TestCase):
    def check_layout(self, mode):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root / '.tmp_probe') as temporary:
            environment = dict(os.environ)
            environment['EVERYTHING_RAG_DATA_DIR'] = temporary
            environment['PYTHONPATH'] = str(root / 'backend')
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), mode],
                env=environment, capture_output=True, timeout=20, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode('utf-8', errors='replace'))

    def test_legacy_layout_initializes_registered_workspaces(self):
        self.check_layout('legacy')

    def test_migrated_layout_initializes_database_workspaces(self):
        self.check_layout('migrated')


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] in {'legacy', 'migrated'}:
        asyncio.run(initialize(sys.argv[1]))
    else:
        unittest.main()
