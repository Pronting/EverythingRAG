"""问答生成层接口（对齐技术选型决策文档 T1/T3）。

对话模型走 OpenAI 兼容可插拔（默认本地 Ollama/llama.cpp，可选云端）；
流式输出 SSE：POST /api/chat + 前端 fetch/ReadableStream，帧契约 meta/token/done/error。
识图模型复用同一底层 client，见 VisionModel。
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol


class ChatModel(Protocol):
    """统一对话模型接口（支持流式）。"""

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """流式生成：逐 token 产出文本增量。
        若上游为 OpenAI 兼容 chat/completions stream，在此层透传为 token 帧。
        """
        ...

    @property
    def supports_image_input(self) -> bool:
        """能力位：是否支持多模态输入（v0.2 喂图增强用，MVP 恒 False）。"""
        ...


class VisionModel(Protocol):
    """识图模型接口（方案 B：图片 -> 文字描述，入库一次性、非流式）。"""

    def describe(self, image_bytes: bytes, prompt: str) -> str:
        """把图片转为文字描述（走与 ChatModel 相同的底层 OpenAI 兼容 client）。"""
        ...
