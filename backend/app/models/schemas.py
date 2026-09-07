"""数据模型：块元数据 schema（对齐技术选型决策文档 T1/T4）。

注意：Chroma 元数据仅收标量(str/int/float/bool)。非标量字段
（如 heading_path / image_path 列表）放 state.db 双写，见 T4。
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class SourceType(StrEnum):
    DOCUMENT = "document"
    CONVERSATION = "conversation"
    IMAGE_DESCRIPTION = "image_description"


class BlockMetadata(BaseModel):
    """向量库块的元数据与来源映射（块 ID -> 文件/对话/平台，可回溯）。"""

    block_id: str
    doc_id: str
    source_type: SourceType
    source_file: str
    platform: str = "local"
    created_time: str | None = None
    heading_path: str | None = None  # 非标量 -> state.db 双写
    heading_aliases: str | None = None  # JSON 字符串；聚合短节的全部原始标题路径
    anchor: str | None = None
    chunk_type: str = "text"  # text / code / table / image_description
    image_path: str | None = None  # 图片原图路径 / 图床 URL（仅展示溯源）
    image_content_hash: str | None = None  # 图片降采样后内容哈希（同图去重键）
    image_index_text: str | None = None  # 独立检索摘要；正文保存完整证据
    image_representation_version: int | None = None
    # 对话专属
    message_role: str | None = None
    message_seq: int | None = None
    dedup_key: str | None = None


class HealthStatus(BaseModel):
    app: dict
    config: dict
    knowledge: dict
    privacy: dict
