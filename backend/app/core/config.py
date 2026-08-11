"""配置中心（MVP 骨架）：环境变量 + config.json，对齐技术选型决策文档 T6。

读取优先级：环境变量（前缀 EVERYTHING_RAG_）> config.json > 默认值。
MVP 阶段以环境变量为主；后续配置向导落地后并入 config.json 管理。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

APP_VERSION = "0.1.0"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EVERYTHING_RAG_",
        env_file=".env",
        extra="ignore",
    )

    app_version: str = APP_VERSION
    # 本地数据目录（向量库 / state.db / 配置，均不出本机）
    data_dir: Path = Path.home() / ".everything-rag"
    # 本服务绑定（本地回环，不开放远程访问）
    host: str = "127.0.0.1"
    port: int = 8000  # 实际运行用随机端口（见 scripts/run.py）
    # CORS：同源部署默认关闭，保留配置位
    enable_cors: bool = False
    cors_origins: list[str] = []
    # 功能开关（对齐 PRD 4.3 图片 / 4.7 配置向导）
    vision_enabled: bool = False
    wizard_completed: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
