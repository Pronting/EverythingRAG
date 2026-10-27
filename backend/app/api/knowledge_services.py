import threading

from app.api import deps
from app.core.config import settings
from app.core.settings_store import get_settings_store
from app.knowledge.context import KnowledgeContext
from app.knowledge.models import KnowledgeError
from app.knowledge.service import KnowledgeService

_SERVICES: dict[int, KnowledgeService] = {}
_LOCK = threading.Lock()


def knowledge_service_for(context=None) -> KnowledgeService:
    configuration = get_settings_store().load().embed
    if not configuration.base_url or not configuration.model:
        raise KnowledgeError("EMBEDDING_NOT_CONFIGURED")
    context = context or deps.get_workspace_context()
    store = deps.get_workspace_vector_store(context)
    with _LOCK:
        service = _SERVICES.get(id(store))
        if service is None:
            _SERVICES.clear()
            service = KnowledgeService(KnowledgeContext(
                vector_store=store, embedder=store._embedder,
                state_store=deps.workspace_state_store(context),
                rerank=lambda: deps.build_rerank_runtime(get_settings_store()),
                max_result_chars=settings.mcp_max_result_chars,
                current_fingerprint=lambda: deps.get_embedder().fingerprint,
            ))
            _SERVICES[id(store)] = service
        return service
