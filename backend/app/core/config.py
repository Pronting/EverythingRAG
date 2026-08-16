"""配置中心（MVP 骨架）：环境变量 + config.json，对齐技术选型决策文档 T6。

读取优先级：环境变量（前缀 EVERYTHING_RAG_）> config.json > 默认值。
MVP 阶段以环境变量为主；后续配置向导落地后并入 config.json 管理。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
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
    # 识图 provider（OpenAI 兼容多模态；图床图片 -> 文字描述，key 仅运行期从环境读取）
    vision_base_url: str | None = None  # env: EVERYTHING_RAG_VISION_BASE_URL
    vision_model: str | None = None  # env: EVERYTHING_RAG_VISION_MODEL
    vision_api_key: SecretStr | None = None  # env: EVERYTHING_RAG_VISION_API_KEY
    # 对话 provider（OpenAI 兼容云端；API key 仅运行期从环境读取，绝不落本类值）
    chat_provider_type: str = "openai_compatible"
    chat_base_url: str | None = None  # env: EVERYTHING_RAG_CHAT_BASE_URL
    chat_model: str | None = None  # env: EVERYTHING_RAG_CHAT_MODEL
    # API key 值（SecretStr：repr / 序列化自动打码）。读取来源：.env 或环境变量
    # EVERYTHING_RAG_CHAT_API_KEY；留空时回退到 chat_api_key_env 指定的环境变量名。
    chat_api_key: SecretStr | None = None  # env: EVERYTHING_RAG_CHAT_API_KEY
    # 备用：key 所在的环境变量名（非 key 值本身），供自定义变量名场景
    chat_api_key_env: str = "EVERYTHING_RAG_CHAT_API_KEY"
    # 嵌入 provider（OpenAI 兼容云端 /embeddings；本地 fastembed/bge-m3 已移除）
    embed_base_url: str | None = None  # env: EVERYTHING_RAG_EMBED_BASE_URL
    embed_model: str | None = None  # env: EVERYTHING_RAG_EMBED_MODEL
    embed_api_key: SecretStr | None = None  # env: EVERYTHING_RAG_EMBED_API_KEY
    # 联网搜索 provider（Tavily / SearXNG；出网需显式启用 + 配置，默认关闭零外发）
    search_provider_type: str = "tavily"  # env: EVERYTHING_RAG_SEARCH_PROVIDER_TYPE
    search_enabled: bool = False  # env: EVERYTHING_RAG_SEARCH_ENABLED
    search_base_url: str | None = None  # env: EVERYTHING_RAG_SEARCH_BASE_URL（SearXNG 实例）
    search_api_key: SecretStr | None = None  # env: EVERYTHING_RAG_SEARCH_API_KEY
    search_max_results: int = 5  # env: EVERYTHING_RAG_SEARCH_MAX_RESULTS


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
