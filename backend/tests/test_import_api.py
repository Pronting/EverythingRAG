"""导入 API 测试：异步启动 / 轮询状态 / 404 / mode 校验 / 坏目录（fake pipeline，零网络）。

用 FakePipeline 替换 get_import_pipeline（不加载真实 bge-m3 / Chroma）；
conftest 注入的 FakeVectorStore 保证 /api/status 不触碰真实 ~/.everything-rag。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.api.routes import imports as imports_mod
from app.ingestion.pipeline import IngestReport, ProgressSnapshot
from app.main import app
from tests.fakes import FakeVectorStore


class FakePipeline:
    """逐文件回调 on_progress 两次、返回固定报告；记录导入目录。"""

    def __init__(self, report: IngestReport | None = None) -> None:
        self.report = report or IngestReport(
            files_scanned=2, files_parsed=2, files_skipped=0, chunks=3, blocks_upserted=3
        )
        self.ingested: list[str] = []
        self._snapshots = [
            ProgressSnapshot(files_scanned=2, files_parsed=1, files_skipped=0, chunks=1),
            ProgressSnapshot(files_scanned=2, files_parsed=2, files_skipped=0, chunks=3),
        ]

    def ingest(self, root_dir: Path, on_progress=None) -> IngestReport:  # type: ignore[no-untyped-def]
        self.ingested.append(str(root_dir))
        for snapshot in self._snapshots:
            if on_progress is not None:
                on_progress(snapshot)
        return self.report


class BlockingPipeline(FakePipeline):
    """发完进度后阻塞，直到 release 事件置位（用于验证 running 中间态）。"""

    def __init__(self, release: threading.Event) -> None:
        super().__init__()
        self._release = release

    def ingest(self, root_dir: Path, on_progress=None) -> IngestReport:  # type: ignore[no-untyped-def]
        self.ingested.append(str(root_dir))
        for snapshot in self._snapshots:
            if on_progress is not None:
                on_progress(snapshot)
        assert self._release.wait(timeout=5)
        return self.report


@pytest.fixture
def client() -> None:
    app.dependency_overrides[imports_mod.get_import_pipeline] = lambda: FakePipeline()
    with TestClient(app) as test_client:
        yield test_client
    # 清理交由 conftest 的 _isolate_local_deps teardown（clear 全部）


def _write_md(root: Path, name: str = "a.md") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text("# A\n\n内容\n", encoding="utf-8")
    return path


def _wait_done(client: TestClient, task_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.get(f"/api/import/status/{task_id}").json()
        if data["status"] != "running":
            return data
        time.sleep(0.01)
    raise AssertionError("task did not reach terminal state")


# ---------------------------------------------------------------- 1. 启动 + 轮询到 done


def test_start_import_returns_task_and_completes(client: TestClient, tmp_path: Path) -> None:
    """POST /api/import -> 202 + task_id；轮询到 done，report/progress 字段齐全。"""
    root = tmp_path / "docs"
    _write_md(root)

    resp = client.post("/api/import", json={"dir": str(root), "mode": "full"})
    assert resp.status_code == 202
    task_id = resp.json()["task_id"]
    assert task_id

    data = _wait_done(client, task_id)
    assert data["status"] == "done"
    assert data["error"] is None
    assert data["report"]["files_scanned"] == 2
    assert data["report"]["files_parsed"] == 2
    assert data["report"]["chunks"] == 3
    assert data["report"]["errors"] == []
    assert data["progress"]["chunks"] == 3  # 终态进度对齐报告


def test_import_running_state_exposes_progress(client: TestClient, tmp_path: Path) -> None:
    """阻塞管线：POST 后轮询能看到 running + 实时进度，release 后转为 done。"""
    release = threading.Event()
    app.dependency_overrides[imports_mod.get_import_pipeline] = lambda: BlockingPipeline(release)
    root = tmp_path / "docs"
    _write_md(root)

    resp = client.post("/api/import", json={"dir": str(root)})
    task_id = resp.json()["task_id"]

    time.sleep(0.1)  # 给后台线程时间发进度快照
    data = client.get(f"/api/import/status/{task_id}").json()
    assert data["status"] == "running"
    assert data["report"] is None
    assert data["progress"]["files_parsed"] == 2  # 实时进度可见

    release.set()
    done = _wait_done(client, task_id)
    assert done["status"] == "done"


# ---------------------------------------------------------------- 2. 校验


def test_import_status_unknown_404(client: TestClient) -> None:
    """未知 task_id -> 404。"""
    resp = client.get("/api/import/status/no-such-task")
    assert resp.status_code == 404


def test_import_mode_incremental_400(client: TestClient, tmp_path: Path) -> None:
    """mode=incremental -> 400 + 清晰「未实现」信息。"""
    root = tmp_path / "docs"
    _write_md(root)
    resp = client.post("/api/import", json={"dir": str(root), "mode": "incremental"})
    assert resp.status_code == 400
    assert "增量同步" in resp.json()["detail"]


def test_import_invalid_dir_400(client: TestClient) -> None:
    """目录不存在 -> 400，不启动任务。"""
    resp = client.post("/api/import", json={"dir": "C:/no/such/dir"})
    assert resp.status_code == 400
    assert "目录不存在" in resp.json()["detail"]


def test_import_mode_invalid_422(client: TestClient, tmp_path: Path) -> None:
    """mode 不是 full/incremental -> 422（Pydantic Literal 校验）。"""
    root = tmp_path / "docs"
    _write_md(root)
    resp = client.post("/api/import", json={"dir": str(root), "mode": "garbage"})
    assert resp.status_code == 422


# ---------------------------------------------------------------- 3. /api/status 知识计数


def test_status_knowledge_counts_reflect_store() -> None:
    """/api/status 的 knowledge 计数来自注入的 vector store（file/chunk/image/last_sync）。"""
    app.dependency_overrides[deps.get_vector_store] = lambda: FakeVectorStore(file_count=3, chunk_count=7)
    client = TestClient(app)
    data = client.get("/api/status").json()
    assert data["knowledge"]["file_count"] == 3
    assert data["knowledge"]["chunk_count"] == 7
    assert data["knowledge"]["image_count"] == 0
    assert data["knowledge"]["last_sync_at"] is None  # 未导入过 -> None
    assert data["knowledge"]["needs_rebuild"] is False


def test_status_last_sync_at_after_import(client: TestClient, tmp_path: Path) -> None:
    """成功导入后 /api/status 的 last_sync_at 非空。"""
    root = tmp_path / "docs"
    _write_md(root)
    resp = client.post("/api/import", json={"dir": str(root)})
    _wait_done(client, resp.json()["task_id"])

    data = client.get("/api/status").json()
    assert data["knowledge"]["last_sync_at"] is not None


# ---------------------------------------------------------------- 4. 上传式导入


def _upload_files(
    client: TestClient,
    entries: list[tuple[str, bytes]],
) -> dict:
    """用 multipart 上传一组文件（filename 携带相对路径），返回响应。"""
    files = [("files", (name, content, "text/markdown")) for name, content in entries]
    return client.post("/api/import/upload", files=files)


def test_upload_import_returns_task_and_completes(client: TestClient) -> None:
    """上传 2 个 .md（含子目录相对路径）-> 202 + task_id；轮询到 done，报告齐全。"""
    resp = _upload_files(
        client,
        [("a.md", "# A\n\n内容\n".encode()), ("sub/b.md", "# B\n\n内容\n".encode())],
    )
    assert resp.status_code == 202
    task_id = resp.json()["task_id"]

    try:
        data = _wait_done(client, task_id)
        assert data["status"] == "done"
        assert data["report"]["files_scanned"] == 2
        assert data["report"]["files_parsed"] == 2
        assert data["report"]["chunks"] == 3
    finally:
        # 上传文档持久在 data_dir/documents/<task_id>，测试后清理避免污染真实数据目录
        import shutil

        from app.core.config import settings

        shutil.rmtree(settings.data_dir / "documents" / task_id, ignore_errors=True)


def test_upload_rejects_path_traversal(client: TestClient) -> None:
    """filename 含 ../ 穿越 -> 400，不启动任务。"""
    resp = _upload_files(client, [("../evil.md", b"# x\n")])
    assert resp.status_code == 400
    assert "非法文件路径" in resp.json()["detail"]


def test_upload_rejects_absolute_path(client: TestClient) -> None:
    """filename 为绝对路径（盘符/根）-> 400。"""
    resp = _upload_files(client, [("C:/Users/me/x.md", b"# x\n")])
    assert resp.status_code == 400
    assert "非法文件路径" in resp.json()["detail"]


def test_upload_requires_markdown(client: TestClient) -> None:
    """只上传非 .md 文件 -> 400「没有 Markdown」。"""
    resp = _upload_files(client, [("note.txt", b"hello")])
    assert resp.status_code == 400
    assert "Markdown" in resp.json()["detail"]


def test_upload_without_files_422(client: TestClient) -> None:
    """不带文件字段 -> 422（FastAPI 必填校验）。"""
    resp = client.post("/api/import/upload")
    assert resp.status_code == 422


def test_upload_dedupes_colliding_filenames(client: TestClient) -> None:
    """多选上传同名文件（来自不同文件夹）-> 不互相覆盖，存为去重路径。

    浏览器多选文件夹/文件时可能出现同名（如两个不同目录都有 readme.md）；
    后端须按文件名加序号去重，保证每个文件独立入库。
    """
    import shutil

    from app.core.config import settings

    resp = _upload_files(
        client,
        [("a.md", b"# A1\n"), ("a.md", b"# A2\n")],
    )
    assert resp.status_code == 202
    task_id = resp.json()["task_id"]
    try:
        data = _wait_done(client, task_id)
        assert data["status"] == "done"
        doc_dir = settings.data_dir / "documents" / task_id
        # 两个同名文件都保留，第二个加序号
        assert (doc_dir / "a.md").is_file()
        assert (doc_dir / "a (1).md").is_file()
    finally:
        shutil.rmtree(settings.data_dir / "documents" / task_id, ignore_errors=True)
