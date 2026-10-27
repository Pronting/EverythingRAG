from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KnowledgeContext:
    vector_store: Any
    embedder: Any
    state_store: Any
    rerank: Callable[[], Any]
    max_result_chars: int = 24000
    current_fingerprint: Callable[[], str] | None = None
