import logging

from openai import APIConnectionError
from tenacity import (
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    stop_before_delay,
    wait_fixed,
)

logger = logging.getLogger(__name__)


def _log_retry(state: RetryCallState) -> None:
    error = state.outcome.exception()
    logger.warning(
        "provider_connection_retry attempt=%s error_type=%s",
        state.attempt_number, type(error).__name__,
    )


def connection_retry_options(timeout: float) -> dict:
    # 重试限于连接异常，次数和等待时间受当前请求预算约束。
    return {
        "retry": retry_if_exception_type(APIConnectionError),
        "stop": stop_after_attempt(3) | stop_before_delay(timeout),
        "wait": wait_fixed(0.5),
        "before_sleep": _log_retry,
        "reraise": True,
    }
