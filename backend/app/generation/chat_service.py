"""ChatService：组装问答链路（嵌入 -> 检索 -> 上下文 -> 流式生成 -> SSE 帧）。

产出 SSE 帧序列（格式稳定，前端任务 8 解析依赖）：
    meta（来源块）-> token(s) -> done（正常终帧）；
任何阶段异常 -> 单个 error 帧（终帧，不崩、不产生 done）。

错误消息脱敏：仅透传领域异常（RetrievalError / ChatProviderError）的受控文案，
其余异常一律收敛为通用提示，不泄漏正文 / 文件路径 / 密钥。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.generation.base import ChatModel
from app.generation.providers import ChatProviderError
from app.retrieval.vector_retriever import RetrievalError, RetrievedChunk, VectorRetriever
from app.vectorstore.embedder import Embedder

_EMBED_ERROR_MESSAGE = "嵌入服务暂时不可用，请稍后重试"
_CHAT_ERROR_MESSAGE = "对话服务暂时不可用，请稍后重试"

_SYSTEM_PROMPT = (
    "你是 Everything RAG 个人知识助手。请严格基于给定的参考上下文作答，"
    "不要编造上下文之外的内容；若上下文无法回答该问题，请如实说明。"
    "引用来源时只能使用上下文中标注的序号，不得虚构来源。"
)


class ChatService:
    """RAG 问答编排：查询嵌入 -> 来源检索 -> 上下文组装 -> 对话模型流式生成。"""

    def __init__(
        self,
        embedder: Embedder,
        retriever: VectorRetriever,
        chat_model: ChatModel,
    ) -> None:
        self._embedder = embedder
        self._retriever = retriever
        self._chat_model = chat_model

    async def stream_answer(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """逐帧产出 SSE 帧；对话模型首帧失败时不发 meta，直接 error 终帧。"""
        try:
            chunks = self._retrieve_context(message)
        except (RetrievalError, ChatProviderError) as exc:
            yield _error_frame(str(exc))
            return
        except Exception:  # noqa: BLE001 -- 嵌入器为 Protocol，异常不可枚举，兜底不崩
            yield _error_frame(_EMBED_ERROR_MESSAGE)
            return

        meta = self._meta_frame(chunks)

        try:
            # MEDIUM-2：stream_chat 调用并入首帧 try，同步抛错也收敛为单个 error 帧
            stream = self._chat_model.stream_chat(_build_messages(message, chunks))
            first_token = await anext(stream)
        except StopAsyncIteration:
            yield meta
            yield _done_frame()
            return
        except ChatProviderError as exc:
            yield _error_frame(str(exc))
            return
        except Exception:  # noqa: BLE001 -- 对话模型为 Protocol，异常不可枚举，兜底不崩
            yield _error_frame(_CHAT_ERROR_MESSAGE)
            return

        yield meta
        yield _token_frame(first_token)
        try:
            async for token in stream:
                yield _token_frame(token)
        except ChatProviderError as exc:
            yield _error_frame(str(exc))
            return
        except Exception:  # noqa: BLE001 -- 对话模型为 Protocol，异常不可枚举，兜底不崩
            yield _error_frame(_CHAT_ERROR_MESSAGE)
            return
        yield _done_frame()

    def _retrieve_context(self, message: str) -> list[RetrievedChunk]:
        vector = self._embedder.embed_texts([message])[0]
        return self._retriever.retrieve(vector)

    def _meta_frame(self, chunks: list[RetrievedChunk]) -> dict[str, Any]:
        sources = [
            {
                "block_id": chunk.block_id,
                "text": chunk.text,
                "source_file": chunk.source_file,
                "similarity": chunk.similarity,
                "heading_path": chunk.heading_path,
                "anchor": chunk.anchor,
                "chunk_type": chunk.chunk_type,
                "platform": chunk.platform,
            }
            for chunk in chunks
        ]
        return {"type": "meta", "sources": sources}


def _build_messages(
    message: str,
    chunks: list[RetrievedChunk],
) -> list[dict[str, Any]]:
    """组装 system（约束提示词）+ user（问题 + 带来源序号的上下文）。"""
    context = "\n\n".join(f"[{index}] {chunk.text}" for index, chunk in enumerate(chunks, start=1))
    user_content = f"问题：{message}\n\n参考上下文：\n{context}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _token_frame(token: str) -> dict[str, Any]:
    return {"type": "token", "text": token}


def _done_frame() -> dict[str, Any]:
    return {"type": "done"}


def _error_frame(message: str) -> dict[str, Any]:
    return {"type": "error", "message": message}
