"""导入 Markdown 知识库到向量库（MVP 无上传界面时的命令行入口）。

用法（从仓库根目录）：
    backend/.venv/Scripts/python scripts/import_md.py --dir <你的文档目录>

说明：
    - 扫描 <dir> 下所有 .md，解析 → 语义切块 → 云端嵌入 → 写入向量库
      （~/.everything-rag，与 Web 服务共用同一数据目录）。
    - 嵌入走云端 OpenAI 兼容 /embeddings（配置来自 backend/.env 或设置页 config.json）。
    - 导入完成后，运行中的服务（/api/chat）即可基于这些文档问答。
    - 幂等：同一内容重复导入不产生重复块（block_id 稳定，upsert 覆盖）。
    - 隐私：文档正文经云端嵌入出网（opt-in）；单文件失败只记异常类型名，不泄露路径/正文。

示例：
    backend/.venv/Scripts/python scripts/import_md.py --dir "C:/Users/me/Documents/knowledge"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 进入 backend 目录：保证能 import app 包，且 .env（backend/.env）被正确读取。
BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
ORIGINAL_CWD = Path.cwd()
os.chdir(BACKEND_DIR)
sys.path.insert(0, str(BACKEND_DIR))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="导入 Markdown 知识库到 Everything RAG 向量库",
    )
    parser.add_argument("--dir", required=True, help="包含 Markdown 文档的目录（绝对路径或相对当前目录）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from app.core.config import get_settings
    from app.core.settings_store import get_settings_store
    from app.ingestion.pipeline import IngestionPipeline
    from app.vectorstore.chroma_store import create_vector_store
    from app.vectorstore.embedder import create_embedder_from_config

    args = parse_args(argv)
    root = Path(args.dir)
    if not root.is_absolute():
        root = ORIGINAL_CWD / root
    if not root.is_dir():
        print(f"[错误] 目录不存在: {root}")
        return 2

    settings = get_settings()
    embedder = create_embedder_from_config(get_settings_store().load().embed)
    print(f"数据目录 : {settings.data_dir}")
    print(f"导入目录 : {root}")
    print(f"嵌入模型 : {embedder.fingerprint}（云端 OpenAI 兼容）")
    print("开始导入...")
    report = IngestionPipeline(
        embedder,
        create_vector_store(persist_dir=settings.data_dir, embedder=embedder),
    ).ingest(root)

    print()
    print("=== 导入完成 ===")
    print(f"扫描文件 : {report.files_scanned}")
    print(f"成功解析 : {report.files_parsed}")
    print(f"跳过     : {report.files_skipped}")
    print(f"生成块   : {report.chunks}")
    print(f"写入块   : {report.blocks_upserted}")
    if report.errors:
        print(f"失败原因 : {', '.join(report.errors)}")
    else:
        print("失败原因 : 无")
    print()
    if report.blocks_upserted == 0:
        print("提示：没有写入任何块——请确认目录下有 .md 文件且内容可解析。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
