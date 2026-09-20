"""Short model capability probes; only synthetic content leaves this process."""
import asyncio
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from app.core.outbound import OutboundEvent, outbound_client
from app.generation.capabilities import (
    CAPABILITY_ENV_VAR,
    CapabilityConfigurationError,
    describe_capabilities,
    resolve_capabilities,
)


def probe_capabilities(model: str, base_url: str | None = None) -> dict[str, Any]:
    """能力探测（纯本地）：按 provider 配置解析能力，零联网、零副作用。

    未知模型 / 未声明的端点 -> ``success=False`` + 明确配置错误消息（含环境变量
    与内置模型名），界面照消息提示用户声明能力即可，不猜测、不静默降级（PRD 9.2）。
    """
    try:
        capabilities = resolve_capabilities(model, base_url)
    except CapabilityConfigurationError as exc:
        return {"success": False, "message": str(exc), "capabilities": None}
    return {
        "success": True,
        "message": "能力已声明",
        "capabilities": describe_capabilities(capabilities),
        "source": CAPABILITY_ENV_VAR,
    }


async def probe_model(kind: str, base_url: str, model: str, api_key: str | None) -> dict:
    started = datetime.now(UTC)
    timer = perf_counter()
    status = None
    try:
        async with asyncio.timeout(20):
            async with AsyncOpenAI(base_url=base_url, api_key=api_key or "local",
                                   timeout=18, max_retries=0) as client:
                if kind == "embed":
                    response = await client.embeddings.create(model=model, input=["Connection test"])
                    if len(response.data) != 1 or not response.data[0].embedding:
                        raise ValueError("invalid embedding response")
                else:
                    content = "Reply with OK."
                    if kind == "vision":
                        content = [
                            {"type": "text", "text": "Describe this image in one word."},
                            {"type": "image_url", "image_url": {"url":
                             "data:image/png;base64,"
                             "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAXklEQVR4nO3PMQ0A"
                             "MAzAsPInvYLYYVWKESTzjhsd8KsBrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGt"
                             "Aa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BbQHKU9LC7/CP1AAAAABJ"
                             "RU5ErkJggg=="}},
                        ]
                    response = await client.chat.completions.create(
                        model=model, messages=[{"role": "user", "content": content}],
                        max_tokens=32,
                    )
                    if not response.choices:
                        raise ValueError("invalid chat response")
        status = 200
        return {"success": True, "message": "连接成功", "latency_ms": round((perf_counter()-timer)*1000)}
    except (TimeoutError, APITimeoutError):
        message = "连接超时，请检查服务地址或稍后重试"
    except APIConnectionError:
        message = "无法连接服务，请检查地址和网络"
    except APIStatusError as exc:
        status = exc.status_code
        message = {
            401: "认证失败，请检查 API Key", 403: "没有访问权限，请检查 API Key 或模型权限",
            404: "服务地址或模型不存在", 429: "请求受限或额度不足，请稍后重试",
            400: "模型不支持此请求，请检查模型名称和服务类型",
        }.get(status, "模型服务暂时不可用，请稍后重试")
    except Exception:  # noqa: BLE001 -- never return provider bodies or credentials
        message = "测试失败，服务未返回有效的模型响应"
    finally:
        outbound_client.record(OutboundEvent(
            provider=kind, destination=urlsplit(base_url).hostname or "unknown",
            method="connection-test", status=status, started_at=started,
            duration_ms=(perf_counter()-timer)*1000,
        ))
    return {"success": False, "message": message, "latency_ms": round((perf_counter()-timer)*1000)}
