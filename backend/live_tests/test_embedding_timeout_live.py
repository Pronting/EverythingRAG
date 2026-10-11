import json
import logging
import math

from app.core.config import settings
from app.core.settings_store import get_settings_store
from app.vectorstore.embedder import create_embedder_from_config


def test_real_query_embedding_records_request_outcome(caplog):
    embedder = create_embedder_from_config(get_settings_store().load().embed)
    configured_timeout = settings.query_embed_timeout_seconds
    with caplog.at_level(logging.WARNING, logger="app.vectorstore.embedder"):
        vector = embedder.embed_query("《穷查理宝典》 现值100元 年收益率5% 一年后终值 计算 例子 原文 引用")
    assert len(vector) == embedder.dim
    assert all(math.isfinite(value) for value in vector)
    entries = [record.getMessage() for record in caplog.records if record.getMessage().startswith("query_embedding_timing ")]
    assert len(entries) == 1
    timing = json.loads(entries[0].removeprefix("query_embedding_timing "))
    assert timing["outcome"] == "completed"
    assert timing["error_type"] is None
    assert timing["cause_type"] is None
    assert timing["request_seconds"] > 0
    assert timing["queue_seconds"] >= 0
    assert "query" not in timing
    assert "api_key" not in timing
    assert settings.query_embed_timeout_seconds == configured_timeout
