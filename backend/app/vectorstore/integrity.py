"""向量索引完整性报告（不包含 ID、正文、路径或向量值）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class IndexIntegrityReport:
    healthy: bool
    logical_count: int
    unique_id_count: int
    embedding_count: int
    unreadable_embedding_count: int
    invalid_dimension_count: int
    non_finite_embedding_count: int
    ann_result_count: int
    ann_unique_id_count: int
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
