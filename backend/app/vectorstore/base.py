"""向量库适配层接口（对齐技术选型决策文档 T1/T4）。

所有外部能力走薄接口 + 工厂注册；MVP 实现为 ChromaVectorStore。
更换后端（LanceDB/Qdrant）只改工厂实现，不碰业务层。
"""
from __future__ import annotations

from typing import Any, Protocol

from app.models.schemas import BlockMetadata


class VectorStore(Protocol):
    """统一向量库接口。

    约束（对齐 T4）：
    - 元数据仅收标量(str/int/float/bool)，非标量放 state.db 双写；
    - 单一 collection（名称含嵌入模型指纹），余弦距离。
    """

    def upsert(
        self,
        blocks: list[tuple[str, str, list[float], BlockMetadata]],
    ) -> None:
        """按 block_id 幂等写入「向量 + 块文本 + 元数据 + 来源」。
        blocks: (block_id, text, vector, metadata) 列表；text 为块原文
        （Chroma 存储必须包含文本，检索返回带文本的块）。
        """
        ...

    def delete_by_ids(self, block_ids: list[str]) -> None:
        """按 block_id 删除（增量同步 / 重建索引用）。"""
        ...

    def delete_by_where(self, where: dict[str, Any]) -> None:
        """按元数据条件批量删除（如文件删除时按 source_file 级联清块）。"""
        ...

    def get_blocks_by_source(self, source_file: str) -> dict[str, str]:
        """返回某来源文件的全部块：{block_id: text}（增量同步块级比对用）。

        只取文本不取向量（读便宜）；空文件 / 未知来源返回 {}。
        """
        ...

    def list_blocks(self) -> list[tuple[str, str, dict[str, Any]]]:
        """枚举全部块：[(block_id, text, metadata)]（混合检索建 BM25 索引用）。

        只取文本 + 元数据（不取向量，读便宜）；空库返回 []。
        """
        ...

    def query(
        self,
        vector: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """余弦检索 top_k，返回含 metadata / 相似度的块；where 支持平台/来源/时间过滤。"""
        ...

    def count(self) -> int:
        """返回库中块总数；0 = 尚未建立索引（检索层据此判定「未建索引」）。"""
        ...

    def count_files(self) -> int:
        """返回库中去重后的源文件数（同一文件多块只算 1；/api/status 展示用）。"""
        ...


def create_vector_store(**kwargs: Any) -> VectorStore:
    """工厂：MVP 返回 ChromaVectorStore；后续可替换为 LanceDB/Qdrant。

    参数透传给 chroma 工厂（persist_dir / collection_name / embedder）：
    collection_name 缺省 = chunks__bge-m3__v1，embedder 缺省惰性 FastEmbedEmbedder。
    """
    from app.vectorstore.chroma_store import create_vector_store as _chroma_factory

    return _chroma_factory(**kwargs)
