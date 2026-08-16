"""识图 provider：OpenAI 兼容多模态模型（方案 B：图片 -> 文字描述，非流式）。

对齐 T5 §3：底层复用 OpenAI 兼容 client（与 ChatModel / CloudEmbedder 同协议），
上层 VisionModel 薄接口；单图 -> 描述，非流式、确定性（temperature 0.2）。
本地 Ollama / llama.cpp 与云端 API 同一请求体，仅 config 不同。出网经
OutboundClient 记审计（provider="vision"）；api_key 绝不落 config / 日志。
"""

from __future__ import annotations

import base64
import logging
import time
from datetime import UTC, datetime
from urllib.parse import urlsplit

from openai import OpenAI

logger = logging.getLogger(__name__)

#: 识图瞬时错误（5xx / 429 / 网络超时）重试次数与指数退避（T5 §5.4：默认重试 2 次）。
MAX_RETRIES = 2
_RETRY_BACKOFF_S = (2.0, 8.0)  # 2s -> 8s

#: 描述最大长度（字符）：模型不严格服从字数时，硬截断兜底（T5 §4.4），
#: 避免输出过长拖慢速度。目标 600~700 字，硬上限 1000 字。
MAX_DESCRIBE_CHARS = 1000

from app.core.outbound import OutboundClient, OutboundEvent
from app.core.settings_store import VisionModelConfig
from app.generation.base import VisionModel

#: 入库识图用描述模板（T5 §4.1 七段 + 视觉专家定位 + 长度自适应）。
#: 长度约束：信息少的图简短、信息多的图详细，一般 300~1000 字，最多 1200 字——
#: 检索要的是「关键词密度高」，不是几千字的完整复述，过长的描述反而稀释专有名词。
DESCRIBE_PROMPT = (
    "你是专业的视觉理解模型（Vision Expert），唯一任务是「识图」：把图片中的关键信息"
    "完整、客观地提取成文字，供个人知识库检索与问答使用。\n\n"
    "请按下面的固定结构输出，覆盖所有可见信息，但**长度按图片信息密度灵活控制**：\n\n"
    "【画面主体】图的主要类型（架构图/流程图/表格截图/聊天记录截图/界面截图/文档截图/代码截图/照片/图表…）与一句话整体概括。\n"
    "【关键实体与场景】图中的对象、人物、场景、组织、系统组件、命名实体、图标与颜色标识。\n"
    "【图中文字(OCR)】列出图中可见文字：标题、按钮、标签、菜单项、聊天消息要点、数据值、注释、水印。保持原文语言与拼写，不要翻译、不要改数字。\n"
    "【图表与数据】(若为图表/表格) 图表类型、坐标轴含义与单位、系列/图例、趋势、关键数值、占比分布；表格的行列主题与关键条目。\n"
    "【关系与结构】(若为结构图) 节点与连线方向、流程步骤顺序、层级包含关系、依赖关系。\n"
    "【与上下文的关系】(若给定上下文) 本图在所属文档中起什么作用、呼应什么主题；未提供则写「未提供上下文」。\n"
    "【可检索标签】8-12 个逗号分隔的检索词，含专有名词、缩写、数字、主题词。\n\n"
    "只描述客观可见内容；看不清/不确定的如实写「看不清」，严禁编造。\n"
    "长度要求：描述控制在 600~700 字左右，最多不超过 1000 字；信息少的图可更短。不要为凑字数而编造或重复。"
)

#: 聊天「识图代理」用模板：与入库 DESCRIBE_PROMPT 同源，同样长度自适应。
CHAT_IMAGE_PROMPT = (
    "你是专业的视觉理解模型（Vision Expert），唯一任务是「识图」：把图片中的关键信息"
    "完整、客观地提取成文字，供一个纯文本模型据此回答用户关于这张图的问题。\n"
    "必须完整包含：\n"
    "1. 画面主体与整体内容（这是什么图、主要表达什么）；\n"
    "2. 图中所有文字（OCR，保持原文语言与拼写，不要翻译、不要改数字，尽量保留位置线索）；\n"
    "3. 图表/表格的数据分布、坐标轴、图例、趋势、关键数值与占比（若有）；\n"
    "4. 结构关系（节点/连线/流程/层级，若有）；\n"
    "5. 关键实体、专有名词、数字、颜色标识。\n"
    "只描述客观可见内容，看不清/不确定的如实说明，严禁编造。\n"
    "长度控制在 600~700 字左右，最多不超过 1000 字；信息少的图可更短。"
)


class VisionProviderError(Exception):
    """识图 provider 配置缺失 / 调用失败（消息可读、不泄露密钥）。"""


class OpenAICompatibleVision:
    """OpenAI 兼容 chat/completions 多模态识图（单图 -> 描述，非流式）。

    图片以 base64 data URI 内联传输（本地 Ollama 与云端 API 同协议）；
    outbound 可选注入后每次出网记录一条审计事件。
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        outbound: OutboundClient | None = None,
    ) -> None:
        self._model = model
        self._outbound = outbound
        self._destination = _destination_host(base_url)
        self._client = OpenAI(base_url=base_url, api_key=api_key or "local")

    def describe(
        self,
        image_bytes: bytes,
        prompt: str = DESCRIBE_PROMPT,
        mime: str = "image/jpeg",
    ) -> str:
        """把图片转文字描述：单轮、非流式、temperature 0.2、max_tokens 1024。"""
        data_uri = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]
        started_at = _utcnow() if self._outbound is not None else None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=800,
                )
                break
            except BaseException as exc:
                # 瞬时错误（5xx/429/网络超时）指数退避重试；永久错误不重试直接失败
                if attempt >= MAX_RETRIES or not _is_transient(exc):
                    self._record(started_at, None)
                    raise VisionProviderError(f"识图调用失败: {type(exc).__name__}") from exc
                logger.debug("识图瞬时错误重试 %d/%d: %s", attempt + 1, MAX_RETRIES, type(exc).__name__)
                time.sleep(_RETRY_BACKOFF_S[attempt])
        self._record(started_at, 200)
        content = response.choices[0].message.content if response.choices else ""
        if not content:
            raise VisionProviderError("识图返回空描述")
        return content[:MAX_DESCRIBE_CHARS]  # 硬截断：严格 ≤ MAX_DESCRIBE_CHARS 字

    def _record(self, started_at: datetime | None, status: int | None) -> None:
        """记录一条出网审计事件；未注入 outbound / 未开始计时则跳过。"""
        if self._outbound is None or started_at is None:
            return
        self._outbound.record(
            OutboundEvent(
                provider="vision",
                destination=self._destination,
                method="chat.completions",
                status=status,
                started_at=started_at,
                duration_ms=_elapsed_ms(started_at),
            )
        )


def create_vision_model_from_config(
    vision: VisionModelConfig,
    outbound: OutboundClient | None = None,
) -> VisionModel:
    """从设置页 VisionModelConfig 构造识图模型；配置缺失 raise VisionProviderError。"""
    if not vision.base_url or not vision.model:
        raise VisionProviderError("识图模型未配置，缺少: Base URL / 模型名")
    api_key = vision.api_key.get_secret_value() if vision.api_key is not None else None
    return OpenAICompatibleVision(
        base_url=vision.base_url,
        model=vision.model,
        api_key=api_key,
        outbound=outbound,
    )


def _is_transient(exc: BaseException) -> bool:
    """是否瞬时错误（可重试）：HTTP 5xx / 429 限流 / 网络与超时类异常。"""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status >= 500 or status == 429
    name = type(exc).__name__
    return name in ("APIConnectionError", "APITimeoutError", "ConnectionError", "TimeoutError")


def _destination_host(base_url: str) -> str:
    """从 base_url 提取仅 host 的审计目的地（绝不含路径 / query / key）。"""
    hostname = urlsplit(base_url).hostname
    if hostname:
        return hostname
    head = base_url.split("/", 1)[0]
    return head.split("?", 1)[0].split("#", 1)[0] or "unknown"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _elapsed_ms(started_at: datetime) -> float:
    return max(0.0, (datetime.now(UTC) - started_at).total_seconds() * 1000.0)
