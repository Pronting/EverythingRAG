"""共享依赖：向量库注册表 + 导入任务存储单例 + 增量同步状态库。

- get_vector_store：按完整运行时 Provider 配置的不可逆摘要复用本地向量库包装器；
  collection 仍由嵌入模型 fingerprint 隔离。更换地址/密钥会换新客户端，但同名模型
  不会仅因凭证轮换而强制重建向量空间。
- get_import_task_store：全局唯一导入任务存储（重启丢失，单用户本地可接受）。
- get_state_store：增量同步的 SQLite 状态库单例（指纹表/来源根持久化）。
- conversation_store / workspace_store：会话与对话分区（JSON 持久化）单例。
以上均可被 app.dependency_overrides 替换（测试用 fake）。
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from threading import RLock
from typing import Any

from app.core.config import settings
from app.core.conversation_store import ConversationStore
from app.core.settings_store import EmbedModelConfig, get_settings_store
from app.core.workspace_store import WorkspaceStore
from app.generation.chat_service import RerankRuntime
from app.ingestion.image_state_store import ImageStateStore
from app.ingestion.import_task import ImportTaskStore
from app.ingestion.state_store import DocumentStateStore
from app.vectorstore.base import VectorStore
from app.vectorstore.chroma_store import create_vector_store
from app.vectorstore.embedder import Embedder, create_embedder_from_config

#: 全局唯一导入任务存储（进程内，重启丢失）
import_task_store = ImportTaskStore()

#: 全局唯一会话存储（持久化到 data_dir/conversations.json）
conversation_store = ConversationStore(settings.data_dir)

#: 全局唯一工作区存储（持久化到 data_dir/workspaces.json，保证「默认」工作区存在）
workspace_store = WorkspaceStore(settings.data_dir)

#: 按运行时 Provider 配置摘要分 key 的向量库注册表。
#: collection 名仍由 ``Embedder.fingerprint`` 决定；这里仅避免包装器继续持有旧客户端。
_vector_store_registry: dict[str, VectorStore] = {}

#: 按运行时 Provider 配置摘要分 key 的嵌入器注册表（进程内单例）。
#: 关键：CloudEmbedder 的 dim 是惰性解析的（首次 embed 才知道）；向量库与导入/检索
#: 管线必须共享同一实例，否则向量库侧 dim 未解析会在 upsert 校验时抛 RuntimeError。
_embedder_registry: dict[str, Embedder] = {}
_registry_lock = RLock()


def _embedder_runtime_cache_key(config: EmbedModelConfig) -> str:
    """返回完整 Provider 配置的不可逆进程内缓存键，绝不保存或暴露明文密钥。"""

    api_key = config.api_key.get_secret_value() if config.api_key is not None else None
    provider_config = config.model_dump(mode="json")
    provider_config["api_key"] = api_key
    payload = json.dumps(
        provider_config,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"embed-runtime-v1:{hashlib.sha256(payload).hexdigest()}"


def _get_embedder_and_cache_key() -> tuple[Embedder, str]:
    """原子读取当前配置并返回共享嵌入器及其运行时缓存键。"""

    config = get_settings_store().load().embed
    cache_key = _embedder_runtime_cache_key(config)
    with _registry_lock:
        cached = _embedder_registry.get(cache_key)
        if cached is not None:
            return cached, cache_key

        # This is a single-user local service.  Keeping inactive clients has no latency benefit
        # and would retain rotated API secrets in memory, so only the current runtime survives.
        _embedder_registry.clear()
        _vector_store_registry.clear()
        embedder = create_embedder_from_config(config)
        _embedder_registry[cache_key] = embedder
        return embedder, cache_key


def get_embedder() -> Embedder:
    """返回与当前完整 Provider 设置匹配的进程内共享嵌入器。"""

    embedder, _cache_key = _get_embedder_and_cache_key()
    return embedder


def get_vector_store() -> VectorStore:
    """返回与当前完整 Provider 设置匹配的进程内共享向量库包装器。"""

    with _registry_lock:
        embedder, cache_key = _get_embedder_and_cache_key()
        store = _vector_store_registry.get(cache_key)
        if store is None:
            store = create_vector_store(persist_dir=settings.data_dir, embedder=embedder)
            _vector_store_registry[cache_key] = store
        return store


def get_import_task_store() -> ImportTaskStore:
    """返回全局导入任务存储单例。"""
    return import_task_store


@lru_cache
def get_state_store() -> DocumentStateStore:
    """返回增量同步状态库单例（SQLite，落在 data_dir/state.db，重启持久）。"""
    return DocumentStateStore(settings.data_dir / "state.db")


@lru_cache
def get_image_state_store() -> ImageStateStore:
    """返回识图任务 + 内容去重的状态库单例（SQLite，落在 data_dir/image_state.db）。

    持久化：图片任务表（重启续跑）+ 图片描述缓存（跨 URL / 跨导入内容去重）。
    """
    return ImageStateStore(settings.data_dir / "image_state.db")


def current_corpus_identity() -> str | None:
    """当前语料身份哈希（PRD §6.5）：collection 结构版本 + 嵌入 fingerprint + 块 id 清单。

    只读块 id，不读正文，因此结果可安全写进档案与设置页。取不到向量库或块清单时返回
    ``None``（**不猜**）：调用方据此把「无从证明语料一致」保守地标为 ``needs_validation``。
    调用点是设置页的状态快照与启用校验（低频用户动作），不在每轮问答路径上。
    """

    from app.retrieval.reranker import corpus_identity_hash
    from app.vectorstore.chroma_store import INGESTION_SCHEMA_VERSION

    try:
        embedder = get_embedder()
        store = get_vector_store()
        block_ids = [str(block_id) for block_id, _text, _meta in store.list_blocks()]
    except Exception:  # noqa: BLE001 -- 配置缺失 / 向量库不可用不得阻断设置页状态展示
        return None
    return corpus_identity_hash(
        collection=f"ingestion-schema-v{INGESTION_SCHEMA_VERSION}",
        embed_fingerprint=str(getattr(embedder, "fingerprint", "")),
        block_ids=block_ids,
    )


def build_rerank_runtime() -> RerankRuntime:
    """按当前设置组装筛选运行态（Provider + 档案快照）。

    * 未启用 -> 空运行态（ChatService 走旧检索路径并在 meta 标 ``disabled``）；
    * 已启用但配置不可用（缺 endpoint/模型/密钥/档案、档案失配或非 validated）
      -> ``config_error_code``，ChatService 终止本轮并指向设置，**不静默回落**旧路径；
    * 可用 -> 构造真实 Provider 与档案快照。

    只读设置与本地档案文件；不发起任何外呼（外呼只在真实问答/连接测试时发生）。
    """

    from app.core.settings_store import (
        RerankProfileStore,
        get_settings_store,
        validate_rerank_enable,
    )
    from app.retrieval.rerank_provider import SiliconFlowRerankProvider
    from app.retrieval.reranker import RerankError, RerankProviderType

    rerank = get_settings_store().load().rerank
    if not rerank.enabled:
        return RerankRuntime()
    if str(rerank.provider_type) != str(RerankProviderType.SILICONFLOW_RERANK):
        return RerankRuntime(
            config_error_code="RERANK_PROFILE_MISMATCH",
            config_error_message="当前版本只支持已验收的筛选服务类型，请在设置中重新选择",
        )
    profiles = RerankProfileStore.for_data_dir(settings.data_dir).load()
    try:
        profile = validate_rerank_enable(rerank, profiles)
    except RerankError as exc:
        return RerankRuntime(
            config_error_code=exc.code.value,
            config_error_message=exc.user_message,
        )
    except Exception:  # noqa: BLE001 -- 配置解析异常也不得阻断服务启动
        return RerankRuntime(
            config_error_code="RERANK_NOT_CONFIGURED",
            config_error_message="筛选服务配置无效，请在设置中检查",
        )
    return RerankRuntime(
        provider=SiliconFlowRerankProvider(
            model=str(rerank.model or "").strip(),
            api_key=rerank.api_key,
            endpoint=rerank.endpoint,
            instruction_text=profile.instruction_text,
            connect_timeout_seconds=rerank.connect_timeout_seconds,
            read_timeout_seconds=rerank.read_timeout_seconds,
            token_counter=_rerank_token_counter(),
        ),
        profile=profile,
        config_error_code=None,
    )


@lru_cache(maxsize=1)
def _rerank_token_counter() -> Any | None:
    """可选的 token 计数函数（PRD §6.3：token 数不能用字符数替代）。

    只有能拿到成熟 tokenizer 时才提供；拿不到返回 ``None``，让 Provider 在能力声明了限制时
    显式拒绝构造——绝不退化成按字符数估算。
    """

    try:
        import tiktoken  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001 -- 离线环境取不到编码表时不提供计数函数
        return None
    return lambda text: len(encoding.encode(text))


def resume_pending_image_tasks() -> int:
    """应用启动续跑：识图已配置时，加载未完成图片任务后台补做；返回投递数。

    延迟 import（vision/pipeline/worker）避免模块级循环依赖；失败静默降级（续跑是
    尽力而为，不影响应用启动）。仅当有「pending」任务时才起工作线程。
    """
    from app.core.outbound import outbound_client
    from app.core.settings_store import get_settings_store, is_vision_configured
    from app.generation.vision import create_vision_model_from_config
    from app.ingestion.image_fetch import ImageFetcher
    from app.ingestion.image_worker import ImageWorker
    from app.ingestion.pipeline import IngestionPipeline

    app_settings = get_settings_store().load()
    if not is_vision_configured(app_settings.vision):
        return 0
    try:
        vision = create_vision_model_from_config(app_settings.vision, outbound=outbound_client)
        pipeline = IngestionPipeline(
            embedder=get_embedder(),
            vectorstore=get_vector_store(),
            vision=vision,
            fetcher=ImageFetcher(outbound=outbound_client),
            state_store=get_image_state_store(),
        )
    except Exception:  # noqa: BLE001 -- 嵌入/识图未配置等：续跑降级，不阻断启动
        return 0
    worker = ImageWorker(
        pipeline.process_image_task_and_upsert,
        get_image_state_store(),
        settings.vision_concurrency,
    )
    return worker.resume_pending()
