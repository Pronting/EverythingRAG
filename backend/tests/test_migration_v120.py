"""V1.1.0 → V1.2.0 迁移（PRD 15.1–15.3；A26，T01）。

全部素材都是 ``tmp_path`` 里构造的**假旧数据**：本文件**绝不**触碰用户真实的
``~/.everything-rag/``（真实迁移 A26 需要用户显式确认 + 完整备份后才执行）。

覆盖 A26 的四项主张：
1. 停止写入后备份（D16：排除 ``models/`` 并记录原因）、保留向量 / 文件 / 图片 / 会话；
2. **远程模型调用数为 0**（用 outbound 审计断言）；
3. 缺失工作区记录与旧记忆状态按规则转换；
4. 幂等 + 断点续跑，且 collection 计数按公共 API 枚举（不按目录个数）。
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from app.agent.app_store import AppStore
from app.core.migration import (
    BACKUP_EXCLUDED,
    MIGRATION_REVIEW_FILENAME,
    MigrationAbortedError,
    V120Migration,
    convert_legacy_memory,
    stable_message_id,
)
from app.core.outbound import outbound_client
from app.core.workspace_context import (
    WORKSPACE_LAYOUT_VERSION,
    committed_storage_version,
)
from app.core.workspace_store import WorkspaceStore
from app.ingestion.image_state_store import ImageStateStore, ImageTask
from app.ingestion.state_store import DocumentStateStore
from app.memory.models import MemoryCategory, MemoryStatus
from app.memory.store import MemoryStore

EMBED_DIM = 8
COLLECTION_A = "chunks__legacy-embed__v7"
COLLECTION_B = "chunks__legacy-embed__v6"
GHOST_WORKSPACE = "ghost-workspace"
UPLOAD_NAME = "笔记.md"
UPLOAD_BODY = "# 旧上传\n\n迁移必须原样保留字节与 mtime。\n"


def _vector(seed: float) -> list[float]:
    return [seed + index / 100.0 for index in range(EMBED_DIM)]


def _build_legacy(tmp_path: Path) -> Path:
    """在 tmp 里造一个 V1.1.0 旧布局数据目录（含两份 collection）。"""

    data = tmp_path / "legacy"
    (data / "documents").mkdir(parents=True)

    # 1) 工作区与会话（会话引用的工作区记录缺失 → PRD 15.2 第 2 步的归档兜底）
    (data / "workspaces.json").write_text(
        json.dumps(
            [
                {"id": "default", "name": "默认", "is_default": True},
                {"id": "research", "name": "研究"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (data / "conversations.json").write_text(
        json.dumps(
            [
                {
                    "id": "conv-1",
                    "title": "风控讨论",
                    "workspace_id": "default",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-02T00:00:00+00:00",
                    "messages": [
                        {"role": "user", "content": "请用中文回答：知识库管理系统的 P99 是多少？"},
                        {
                            "role": "assistant",
                            "content": "P99 是 300ms。",
                            "answer_basis": "knowledge",
                        },
                    ],
                },
                {
                    "id": "conv-2",
                    "title": "已归档的旧会话",
                    "workspace_id": GHOST_WORKSPACE,
                    "archived": True,
                    "parent_id": "conv-1",
                    "messages": [{"role": "user", "content": "这条引用了不存在的工作区"}],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # 2) 上传文件（保留字节与 mtime）
    upload = data / "documents" / UPLOAD_NAME
    upload.write_text(UPLOAD_BODY, encoding="utf-8")

    # 3) 状态库与图片库
    state = DocumentStateStore(data / "state.db")
    state.upsert_fingerprint(
        str(upload), "content-hash", upload.stat().st_mtime, len(UPLOAD_BODY), 1
    )
    state.register_source(data / "documents")
    state.close()

    images = ImageStateStore(data / "image_state.db")
    image_hash = "b" * 32
    images.save_image_bytes(image_hash, _png_bytes())
    images.save_description(image_hash, "旧识图结果")
    images.add_task(
        ImageTask(
            block_id="image-task-1",
            src="https://example.invalid/old.png",
            doc_id="doc-1",
            source_file=str(upload),
            heading_path="",
            anchor="",
        )
    )
    images.close()

    # 4) 真实 Chroma：两个 collection，显式 embeddings
    client = chromadb.PersistentClient(
        path=str(data), settings=ChromaSettings(anonymized_telemetry=False)
    )
    for name, count, offset in ((COLLECTION_A, 3, 0.0), (COLLECTION_B, 2, 10.0)):
        collection = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
        collection.add(
            ids=[f"{name}-{index}" for index in range(count)],
            documents=[f"旧块 {name} #{index}" for index in range(count)],
            embeddings=[_vector(offset + index) for index in range(count)],
            metadatas=[{"source_file": str(upload)} for _ in range(count)],
        )

    # 5) D16：models/ 排除，表外路径原地保留；vision 报告纳入快照
    (data / "models").mkdir()
    (data / "models" / "bge-m3.onnx").write_bytes(b"onnx" * 256)
    (data / ".chroma-write.lock").write_bytes(b"\0")
    (data / "vision_batch_report.json").write_text("{}", encoding="utf-8")
    # 真实的 state.pre-image-v7.db 是一份 state.db 的历史镜像（真 SQLite 库）
    with sqlite3.connect(str(data / "state.pre-image-v7.db")) as conn:
        conn.execute("CREATE TABLE documents (path TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO documents VALUES ('legacy-path')")

    # 6) 旧记忆库（分类转换表的全部情形）
    _build_legacy_memory(data / "memory.db")
    return data


def _build_legacy_memory(path: Path) -> None:
    store = MemoryStore(path)

    def add(entry_id: str, category: MemoryCategory, statement: str, **kwargs: Any) -> None:
        entry = store.add(
            statement=statement,
            category=category,
            confidence=0.9,
            source_conversation_id="conv-1",
            source_message_index=0,
            quote=kwargs.pop("quote", "请用中文回答：P99 是多少？"),
            status=kwargs.pop("status", MemoryStatus.ACTIVE),
            user_pinned=kwargs.pop("user_pinned", False),
        )
        # 旧 ID 由内容 hash 派生；为了让断言稳定，直接把主键改成可读值
        with sqlite3.connect(str(path)) as conn:
            conn.execute("UPDATE memories SET id = ? WHERE id = ?", (entry_id, entry.id))
            conn.commit()

    add("mem-pref", MemoryCategory.PREFERENCE, "请用中文回答，语气简洁")
    add("mem-constraint", MemoryCategory.CONSTRAINT, "部署环境必须离线，不能联网")
    add("mem-decision", MemoryCategory.DECISION, "决定采用 RRF 混合检索", user_pinned=True)
    add("mem-conclusion", MemoryCategory.CONCLUSION, "讨论认为 BM25 召回偏低")
    add("mem-identity", MemoryCategory.IDENTITY, "这个项目的负责人是张三")
    add("mem-identity-other", MemoryCategory.IDENTITY, "作者喜欢长跑")
    add("mem-forgotten", MemoryCategory.DECISION, "已遗忘的旧决定", status=MemoryStatus.FORGOTTEN)
    add("mem-pending", MemoryCategory.PREFERENCE, "待确认的偏好", status=MemoryStatus.PENDING)
    add("mem-noquote", MemoryCategory.DECISION, "没有引用原话，无法核验", quote="")
    store.close()


def _png_bytes() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (3, 3), (0, 128, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


async def _run_migration(
    target: Path,
    legacy: Path,
    *,
    job_id: str | None = None,
    batch_size: int = 2,
) -> tuple[Any, Any]:
    target.mkdir(parents=True, exist_ok=True)
    app_store = await AppStore.open(target / "app.db")
    migration = V120Migration(
        target,
        app_store=app_store,
        workspace_store=WorkspaceStore(target),
        batch_size=batch_size,
        verify_vectors_fully=True,
    )
    report = await migration.run(source_dir=legacy, job_id=job_id)
    return app_store, report


def _vector_map(chroma_root: Path, collection_name: str) -> dict[str, list[float]]:
    client = chromadb.PersistentClient(
        path=str(chroma_root), settings=ChromaSettings(anonymized_telemetry=False)
    )
    collection = client.get_collection(collection_name)
    raw = collection.get(include=["embeddings"])
    embeddings = raw.get("embeddings")
    if embeddings is None:
        return {}
    return {
        str(block_id): [float(value) for value in vector]
        for block_id, vector in zip(raw.get("ids") or [], embeddings, strict=False)
    }


def _assert_same_vectors(left: dict[str, list[float]], right: dict[str, list[float]]) -> None:
    """按 block ID 逐值比对（Chroma 以 float32 存储，用 float32 epsilon 比对）。"""

    assert set(left) == set(right)
    for block_id, vector in left.items():
        other = right[block_id]
        assert len(other) == len(vector), block_id
        assert all(abs(a - b) <= 1e-5 for a, b in zip(vector, other, strict=False)), block_id


@pytest.fixture(autouse=True)
def _clean_outbound() -> None:
    outbound_client.clear()


async def test_a26_migration_preserves_knowledge_and_makes_no_remote_calls(
    tmp_path: Path,
) -> None:
    legacy = _build_legacy(tmp_path)
    target = tmp_path / "v120-data"
    old_upload = legacy / "documents" / UPLOAD_NAME
    old_mtime = old_upload.stat().st_mtime

    app_store, report = await _run_migration(target, legacy)
    try:
        workspace_root = target / "workspaces" / "default"

        # --- 向量：全部 collection 身份、维度与数值一致（PRD 15.2 第 6 步 / 15.3）---
        assert report.collections == {COLLECTION_A: 3, COLLECTION_B: 2}
        assert report.vector_count == 5
        for name in (COLLECTION_A, COLLECTION_B):
            _assert_same_vectors(
                _vector_map(workspace_root / "chroma", name), _vector_map(legacy, name)
            )

        # --- 文件：字节与 mtime 保留（PRD 15.2 第 5 步 / 15.3）---
        new_upload = workspace_root / "documents" / UPLOAD_NAME
        assert new_upload.read_bytes() == old_upload.read_bytes()
        assert abs(new_upload.stat().st_mtime - old_mtime) < 1.0
        assert report.uploads == 1

        # --- 图片：原图与已完成识别内容保留，未完成任务可继续 ---
        with sqlite3.connect(str(workspace_root / "image_state.db")) as conn:
            originals = dict(conn.execute("SELECT content_hash, data FROM image_originals"))
            tasks = conn.execute("SELECT block_id, status FROM image_tasks").fetchall()
        assert originals["b" * 32] == _png_bytes()
        assert tasks == [("image-task-1", "pending")]

        # --- 会话与消息：内容、顺序、归属、归档、分支一致 ---
        conv1 = await app_store.get_conversation("conv-1")
        assert conv1 is not None
        assert conv1.workspace_id == "default"
        assert await app_store.message_count("conv-1") == 2
        messages = await app_store.list_messages("conv-1")
        assert [message.role for message in messages] == ["user", "assistant"]
        assert messages[1].content == "P99 是 300ms。"
        assert messages[1].source_snapshot.get("answer_basis") == "knowledge"

        # --- 缺失工作区记录 → 建 archived 记录并保留原 id 与归属 ---
        ghost = await app_store.get_workspace(GHOST_WORKSPACE)
        assert ghost is not None
        assert ghost.status.value == "archived"
        assert "已归档" in ghost.name
        conv2 = await app_store.get_conversation("conv-2")
        assert conv2 is not None and conv2.workspace_id == GHOST_WORKSPACE
        assert conv2.status.value == "archived"

        # --- 旧消息下标 → 新消息 ID 映射 ---
        assert report.validation["message_map_conversations"] == 2

        # --- PRD 11.5：历史索引必须用与查询侧相同的 cut_for_search 模式 ---
        # 实测（jieba 0.42.1）：`知识库管理系统` 在 cut 下是 2 个词，在 cut_for_search
        # 下是 6 个词（多出 知识 / 管理 / 系统 / 管理系）。若导入侧用 cut，查询侧按子词
        # 做 AND 就会查不中（A22 / A26）。本断言只对「分叉样例」有效：注意 `固定快照`
        # 在两种模式下**相同**，用它做断言会变成永真。
        with sqlite3.connect(str(target / "app.db")) as conn:
            row = conn.execute(
                "SELECT tokens FROM history_fts WHERE message_id = ?",
                (stable_message_id("conv-1", 0),),
            ).fetchone()
        assert row is not None, "迁移必须建立本区历史全文索引"
        tokens = set(str(row[0]).split())
        assert {"知识", "管理系"} <= tokens, f"分词模式未与查询侧对齐：{sorted(tokens)}"

        # --- D16 备份：排除 models/，纳入分段与状态库，并记录理由 ---
        backup = Path(report.backup_dir)  # type: ignore[arg-type]
        assert backup.is_dir()
        assert not (backup / "models").exists()
        assert list(report.excluded) == list(BACKUP_EXCLUDED)
        for name in (
            "chroma.sqlite3",
            "state.db",
            "image_state.db",
            "conversations.json",
            "workspaces.json",
            ".chroma-write.lock",
            "vision_batch_report.json",
            "documents",
            "state.pre-image-v7.db",
        ):
            assert (backup / name).exists(), name
        reason = report.to_dict()["excluded_reason"]
        assert "fastembed" in reason

        # D16 第 1 条：分段目录（HNSW 图本体）必须与 chroma.sqlite3 一起进快照，
        # 否则恢复出来的是「有逻辑记录但没有近邻图」的残缺库。
        segment_dirs = [
            item.name
            for item in legacy.iterdir()
            if item.is_dir() and re.match(r"^[0-9a-fA-F]{8}-", item.name)
        ]
        assert segment_dirs, "旧库必须真的产生分段目录"
        for name in segment_dirs:
            assert (backup / name).is_dir(), name
            assert (backup / name / "header.bin").is_file(), name

        # --- 表外路径原地保留、不删除 ---
        assert (legacy / "models" / "bge-m3.onnx").is_file()
        assert (legacy / "state.pre-image-v7.db").is_file()

        # --- 来源路径已改写为本区新路径（PRD 15.2 第 7 步）---
        assert report.validation["rewritten_paths"] >= 1
        with sqlite3.connect(str(workspace_root / "state.db")) as conn:
            stored_path = conn.execute("SELECT path FROM documents").fetchone()[0]
        assert stored_path == str(new_upload)

        # --- 校验通过后才提交 storage_schema_version ---
        assert report.validation["ok"] is True
        assert committed_storage_version(target) == WORKSPACE_LAYOUT_VERSION

        # --- A26：远程模型调用数为 0（PRD 15.1）---
        assert outbound_client.is_empty()
        assert report.validation["outbound_entries"] == 0
    finally:
        await app_store.close()


async def test_a26_missing_workspace_and_memory_states_follow_the_rules(
    tmp_path: Path,
) -> None:
    legacy = _build_legacy(tmp_path)
    target = tmp_path / "v120-data"
    app_store, report = await _run_migration(target, legacy)
    try:
        # 旧记忆：可转换的进 memory.json，其余进迁移审阅记录
        assert report.memory_imported >= 5
        assert report.memory_reviewed >= 3

        memory_file = target / "workspaces" / "default" / "memory.json"
        payload = json.loads(memory_file.read_text(encoding="utf-8"))
        kinds = {entry["id"]: entry["kind"] for entry in payload["entries"]}
        assert kinds["mem-pref"] == "response_preference"
        assert kinds["mem-constraint"] == "progress"
        assert kinds["mem-decision"] == "confirmed_conclusion"
        assert kinds["mem-conclusion"] == "progress"
        assert kinds["mem-identity"] == "topic"
        statuses = {entry["id"]: entry["status"] for entry in payload["entries"]}
        # 旧 forgotten 保持状态，不能被自动重新启用
        assert statuses["mem-forgotten"] == "forgotten"

        review = json.loads((target / MIGRATION_REVIEW_FILENAME).read_text(encoding="utf-8"))
        reviewed = {entry["memory_id"] for entry in review["entries"]}
        # 与工作区主题无关的身份内容、旧 pending、引用无法核验的一律进审阅
        assert {"mem-identity-other", "mem-pending", "mem-noquote"} <= reviewed
        assert "mem-pref" not in reviewed

        # 来源消息 ID 已通过消息映射回填（引用可核验）
        entry = next(item for item in payload["entries"] if item["id"] == "mem-pref")
        assert entry["source_message_ids"]
    finally:
        await app_store.close()


async def test_a26_is_idempotent_when_rerun(tmp_path: Path) -> None:
    legacy = _build_legacy(tmp_path)
    target = tmp_path / "v120-data"
    app_store, first = await _run_migration(target, legacy)
    try:
        # 已提交标记 → 直接幂等返回
        same_store, second = await _run_migration(target, legacy)
        try:
            assert second.already_migrated is True
        finally:
            await same_store.close()

        # 去掉标记后用同一个 job 重跑：已完成阶段全部跳过，不产生重复行
        (target / "storage.json").unlink()
        third_store, third = await _run_migration(target, legacy, job_id=first.job_id)
        try:
            assert third.already_migrated is False
            assert await third_store.message_count("conv-1") == 2
            assert third.uploads == first.uploads == 1
            assert third.collections == first.collections
            assert len(list((target / "workspaces" / "default" / "documents").iterdir())) == 1
        finally:
            await third_store.close()
    finally:
        await app_store.close()


async def test_a26_resumes_after_interruption(tmp_path: Path) -> None:
    legacy = _build_legacy(tmp_path)
    target = tmp_path / "v120-data"

    target.mkdir(parents=True, exist_ok=True)
    app_store = await AppStore.open(target / "app.db")
    migration = V120Migration(target, app_store=app_store, workspace_store=WorkspaceStore(target))
    original = migration._stage_copy_uploads

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("模拟断点：复制上传文件时进程退出")

    migration._stage_copy_uploads = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await migration.run(source_dir=legacy)

    # 中断后：未提交布局版本，迁移记录可续跑
    assert committed_storage_version(target) == 0
    jobs = [record for record in await _jobs(app_store)]
    assert len(jobs) == 1
    assert jobs[0].status == "interrupted"
    assert "import_json" in jobs[0].validation.get("completed_stages", [])
    job_id = jobs[0].id

    migration._stage_copy_uploads = original  # type: ignore[method-assign]
    report = await migration.run(source_dir=legacy, job_id=job_id)

    assert committed_storage_version(target) == WORKSPACE_LAYOUT_VERSION
    assert report.uploads == 1
    assert await app_store.message_count("conv-1") == 2
    assert len(list((target / "workspaces" / "default" / "documents").iterdir())) == 1
    record = await app_store.get_migration_job(job_id)
    assert record is not None and record.status == "completed"
    await app_store.close()


async def test_a26_collection_counts_come_from_the_api_not_directories(
    tmp_path: Path,
) -> None:
    """D16 第 4 条：多出几个分段目录也不能改变枚举结果。"""

    legacy = _build_legacy(tmp_path)
    # 残留/孤儿分段目录（真实库里 15 个目录只对应 8 个 segment）
    for name in (
        "deadbeef-0000-0000-0000-000000000001",
        "deadbeef-0000-0000-0000-000000000002",
    ):
        stray = legacy / name
        stray.mkdir()
        (stray / "header.bin").write_bytes(b"junk")
        (stray / "data_level0.bin").write_bytes(b"junk" * 32)

    target = tmp_path / "v120-data"
    app_store, report = await _run_migration(target, legacy)
    try:
        assert report.collections == {COLLECTION_A: 3, COLLECTION_B: 2}
        assert report.validation["collections_source"] == 2
        assert report.validation["collections_target"] == 2
        # 备份把分段目录一起带走（按快照清单），但这不参与 collection 计数
        assert "chroma.sqlite3" in report.included
        backup = Path(report.backup_dir)  # type: ignore[arg-type]
        for name in (
            "deadbeef-0000-0000-0000-000000000001",
            "deadbeef-0000-0000-0000-000000000002",
        ):
            assert (backup / name).is_dir(), name
    finally:
        await app_store.close()


async def test_a26_aborts_without_backup_when_disk_is_insufficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D16 第 6 条：余量不足 → 中止，不得降级为「不备份」。"""

    legacy = _build_legacy(tmp_path)
    target = tmp_path / "v120-data"

    class _Usage:
        total = 1
        used = 1
        free = 0

    monkeypatch.setattr(shutil, "disk_usage", lambda _path: _Usage())

    target.mkdir(parents=True, exist_ok=True)
    app_store = await AppStore.open(target / "app.db")
    migration = V120Migration(target, app_store=app_store, workspace_store=WorkspaceStore(target))
    with pytest.raises(MigrationAbortedError):
        await migration.run(source_dir=legacy)

    assert committed_storage_version(target) == 0
    assert not list(tmp_path.glob("legacy.migration-backup-*"))
    jobs = await _jobs(app_store)
    assert jobs and jobs[0].status == "failed"
    await app_store.close()


def test_memory_conversion_table_matches_prd_15_2() -> None:
    """PRD 15.2 类别转换表的逐条单元断言（不依赖真实数据目录）。"""

    preference = convert_legacy_memory(
        category="preference", status="active", statement="请用中文回答", quote="原话"
    )
    assert preference.kind is not None and preference.kind.value == "response_preference"
    constraint = convert_legacy_memory(
        category="constraint", status="active", statement="部署必须离线", quote="原话"
    )
    assert constraint.kind is not None and constraint.kind.value == "progress"
    decision = convert_legacy_memory(
        category="decision",
        status="active",
        statement="采用 RRF",
        quote="原话",
        user_confirmed=True,
    )
    assert decision.kind is not None and decision.kind.value == "confirmed_conclusion"
    conclusion = convert_legacy_memory(
        category="conclusion", status="active", statement="讨论中", quote="原话"
    )
    assert conclusion.kind is not None and conclusion.kind.value == "progress"
    identity = convert_legacy_memory(
        category="identity", status="active", statement="这个项目的负责人", quote="原话"
    )
    assert identity.kind is not None and identity.kind.value == "topic"

    assert convert_legacy_memory(
        category="identity", status="active", statement="喜欢长跑", quote="原话"
    ).needs_review
    assert convert_legacy_memory(
        category="preference", status="pending", statement="x", quote="原话"
    ).needs_review
    assert convert_legacy_memory(
        category="decision", status="active", statement="x", quote=""
    ).needs_review
    assert convert_legacy_memory(
        category="decision", status="active", statement="x", quote="原话", source_resolved=False
    ).needs_review
    # 旧 forgotten / superseded / orphaned 保持状态
    conversion = convert_legacy_memory(
        category="decision", status="forgotten", statement="x", quote="原话"
    )
    assert conversion.status is not None and conversion.status.value == "forgotten"
    assert not conversion.needs_review


async def _jobs(app_store: AppStore) -> list[Any]:
    """读取全部迁移记录（AppStore 只提供按 id 查询，这里直接查表）。"""

    rows = await app_store._fetch_all("SELECT id FROM migration_jobs ORDER BY created_at")
    records = []
    for row in rows:
        record = await app_store.get_migration_job(str(row["id"]))
        if record is not None:
            records.append(record)
    return records
