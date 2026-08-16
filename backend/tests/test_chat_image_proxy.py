"""ChatService 识图代理单测：纯文本模型 + 贴图 → 识图转文字注入；多模态/未配置分支。

实验性功能：用户在聊天框贴图，当对话模型为纯文本（supports_image_input=False）时，
复用已配识图模型把图片转成文字描述，注入纯文本模型的上下文；多模态模型则跳过代理。
"""

from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image

from app.generation.base import ChatChunk
from app.generation.chat_service import ChatService
from app.retrieval.vector_retriever import RetrievedChunk


def _png_data_uri() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (0, 128, 255)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class FakeEmbedder:
    fingerprint = "fake"
    dim = 4

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]


class FakeRetriever:
    def retrieve(self, *args: object, **kwargs: object) -> list[RetrievedChunk]:
        return []


class FakeVision:
    def __init__(self, description: str = "【画面主体】架构图：系统分三层，网关、服务、存储。") -> None:
        self._description = description
        self.calls = 0

    def describe(self, image_bytes: bytes, prompt: str = "", mime: str = "image/jpeg") -> str:
        self.calls += 1
        return self._description


class FakeChatModel:
    def __init__(self, supports_image: bool = False) -> None:
        self._supports_image = supports_image
        self.messages: list[dict[str, Any]] = []

    @property
    def supports_image_input(self) -> bool:
        return self._supports_image

    async def stream_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.messages = messages
        yield ChatChunk("content", "好")


async def _collect(service: ChatService, images: list[str]) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    async for frame in service.stream_answer("这张图讲了什么？", images=images):
        frames.append(frame)
    return frames


async def test_pure_text_model_proxy_describes_image() -> None:
    """纯文本模型 + 贴图 + 已配识图 -> 识图被调用一次，描述注入 user 消息。"""
    vision = FakeVision()
    chat = FakeChatModel(supports_image=False)
    service = ChatService(FakeEmbedder(), FakeRetriever(), chat, vision=vision)

    frames = await _collect(service, [_png_data_uri()])

    assert vision.calls == 1
    assert [f["type"] for f in frames] == ["meta", "token", "done"]
    # 描述注入纯文本模型的 user 消息
    user_message = chat.messages[-1]["content"]
    assert "【用户图片 1 的识别内容】" in user_message
    assert "架构图" in user_message


async def test_pure_text_model_without_vision_errors() -> None:
    """纯文本模型 + 贴图 + 未配识图 -> 单个 error 帧，提示配置识图模型。"""
    chat = FakeChatModel(supports_image=False)
    service = ChatService(FakeEmbedder(), FakeRetriever(), chat, vision=None)

    frames = await _collect(service, [_png_data_uri()])

    assert [f["type"] for f in frames] == ["error"]
    assert "未配置识图模型" in frames[0]["message"]


async def test_multimodal_model_skips_proxy() -> None:
    """多模态模型 + 贴图 -> 识图代理不生效（vision 不被调用）。"""
    vision = FakeVision()
    chat = FakeChatModel(supports_image=True)
    service = ChatService(FakeEmbedder(), FakeRetriever(), chat, vision=vision)

    frames = await _collect(service, [_png_data_uri()])

    assert vision.calls == 0
    assert [f["type"] for f in frames] == ["meta", "token", "done"]


async def test_no_images_no_describe() -> None:
    """不贴图 -> 识图代理完全不触发。"""
    vision = FakeVision()
    chat = FakeChatModel(supports_image=False)
    service = ChatService(FakeEmbedder(), FakeRetriever(), chat, vision=vision)

    await _collect(service, [])

    assert vision.calls == 0
