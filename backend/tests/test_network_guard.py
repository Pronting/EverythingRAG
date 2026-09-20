"""隔离与出网守卫的**非永真**证据（CLAUDE.md §11）。

本文件回答三个问题，每一条都可被"改坏实现"证伪：

1. **守卫真的拦得住吗** —— 非回环连接立即失败并给出可读原因；回环仍然放行
   （Windows 的 ``socket.socketpair()`` 走 127.0.0.1，事件循环与本地测试服务器都靠它）。
2. **隔离真的挡掉真实凭证吗** —— 默认运行里 ``get_settings_store().load()`` 是空配置、
   三个模型装配点都是 ``None``；同时用「真实 ``.env`` 回退」对照证明这个风险**真实存在**
   （不是防御性空转）：撤掉隔离后同样的调用就能看到真实端点与密钥。
3. **live 用例还能看见真实配置吗** —— 显式 ``EVERYTHING_RAG_LIVE_EVAL=1`` 时三层隔离全部
   不装；用**子进程**跑 ``tests/test_netguard_scope.py`` 的互补判据实证，全程不产生外呼。
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from app.agent import runtime as agent_runtime
from app.core.settings_store import SettingsStore
from tests import netguard

BACKEND_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = BACKEND_ROOT / ".env"


# =====================================================================
# ① 守卫：非回环立即失败，回环放行
# =====================================================================


@pytest.mark.allow_outbound
def test_guard_fails_loudly_on_non_loopback_connect() -> None:
    """故意触发一次非回环连接：必须**立即**以可读原因失败（而不是静默真出网）。

    ``allow_outbound`` 只豁免「本轮不得有连接尝试」的记账断言 —— 这条用例要的正是
    一次被拦下的尝试。若守卫被移除，这里会变成一次**真实外呼**（或超时/连接失败），
    对 ``OutboundBlockedError`` 的断言必然红 ⇒ 非永真。
    """

    netguard.clear_blocked_attempts()
    with pytest.raises(netguard.OutboundBlockedError) as info:
        socket.create_connection(("93.184.216.34", 80), timeout=2)

    message = str(info.value)
    assert "非回环" in message and "93.184.216.34" in message
    assert netguard.LIVE_EVAL_ENVS[0] in message  # 指名显式授权开关
    assert "CLAUDE.md" in message  # 指名红线条目

    attempts = netguard.blocked_attempts()
    assert [item["host"] for item in attempts] == ["93.184.216.34"]
    assert attempts[0]["port"] == 80
    assert attempts[0]["where"], "必须记录触发位置，便于定位写测试的人"


@pytest.mark.allow_outbound
def test_guard_covers_socket_connect_and_connect_ex() -> None:
    """三条出网入口都要覆盖：``socket.connect`` / ``connect_ex`` / ``create_connection``。"""

    netguard.clear_blocked_attempts()
    raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(netguard.OutboundBlockedError):
            raw.connect(("93.184.216.34", 80))
        with pytest.raises(netguard.OutboundBlockedError):
            raw.connect_ex(("93.184.216.34", 80))
    finally:
        raw.close()
    assert len(netguard.blocked_attempts()) == 2


def test_guard_allows_loopback() -> None:
    """回环放行：连一个没人监听的本地端口应以连接错误结束，而不是被守卫拦下。"""

    with pytest.raises(OSError) as info:
        socket.create_connection(("127.0.0.1", 9), timeout=1)
    assert not isinstance(info.value, netguard.OutboundBlockedError)
    assert netguard.blocked_attempts() == [], "回环连接不得被记账"


def test_loopback_predicate_table() -> None:
    for host in ("127.0.0.1", "127.0.0.2", "::1", "localhost", "LOCALHOST.", "[::1]", ""):
        assert netguard.is_loopback(host) is True, host
    for host in ("93.184.216.34", "example.com", "10.0.0.1", "::ffff:8.8.8.8"):
        assert netguard.is_loopback(host) is False, host


# =====================================================================
# ② 隔离真的挡掉真实凭证（并用 .env 回退对照证明风险真实存在）
# =====================================================================


def test_settings_isolation_exposes_no_real_credentials() -> None:
    """默认运行：设置存储被换成隔离解析器（种离线配置），三处模型装配点都走「未配置 ⇒ None」。"""

    from app.core import settings_store as settings_store_mod

    assert netguard.settings_store_isolated() is True
    assert netguard.chat_config_isolated() is True
    assert netguard.outbound_guard_installed() is True

    store = settings_store_mod.get_settings_store()
    settings = store.load()
    assert isinstance(store, SettingsStore)
    # 与真实凭证唯一的相似之处是"字段齐备"：对话/联网/识图/筛选全部未配置。
    assert settings.chat.base_url is None and settings.chat.api_key is None
    assert settings.search.enabled is False and settings.search.api_key is None
    assert settings.rerank.enabled is False and settings.rerank.api_key is None
    assert settings.vision.base_url is None and settings.vision.api_key is None
    # 嵌入只留**回环 + 丢弃端口**的占位（真实装配路径要能构造嵌入器；误用则在本机立刻失败）。
    assert settings.embed.base_url == netguard.SEED_EMBED_BASE_URL
    assert netguard.is_loopback("127.0.0.1") is True

    # 三个装配点（PRD 8.1/8.2 的模型侧）都拿不到模型 ⇒ 各自走确定性/结构化分支。
    assert agent_runtime.general_model_for_chat() is None
    assert agent_runtime.synthesis_model_for_chat() is None
    assert agent_runtime.model_verifier_for_chat() is None


def test_env_fallback_would_leak_real_credentials_without_the_isolation(
    tmp_path: Path,
) -> None:
    """非永真对照：**没有** config.json 时 ``load()`` 会回退 ``.env``。

    这条用例不做任何出网，只证明「隔离是必要的」：同样一次
    ``SettingsStore(无 config.json 的目录).load()``，在没有隔离时会拿到真实端点与密钥，
    而 :func:`test_settings_isolation_exposes_no_real_credentials` 证明默认运行里拿不到。
    """

    if not ENV_FILE.is_file():
        pytest.skip(f"本机没有 {ENV_FILE}（真实凭证不在仓库里），无法对照")

    leaked = SettingsStore(tmp_path / "no-config-here").load()

    assert leaked.chat.base_url, ".env 回退必须让真实 chat 端点可见（风险真实存在）"
    assert leaked.chat.api_key is not None
    # 正因如此，缺少隔离时 provider 会被真的构造出来（_chat_provider_available 只做构造期校验）
    from app.agent.runtime import _chat_provider_available

    assert _chat_provider_available(leaked.chat) is True


def test_documented_seam_still_injects_a_fake_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """隔离不是"一刀切":验收用例 patch ``create_chat_model_from_config`` 的接法仍然有效。

    ``test_general_model_wiring`` / ``test_seam_assembly`` / ``test_lifespan_assembly`` 靠这条
    接缝断言「已配置 ⇒ 可调用」；如果隔离直接把三个装配点钉成 ``None``，那条真实装配路径的
    覆盖就会被抹掉 —— 本用例守住这一点。
    """

    class FakeChunk:
        kind = "content"
        text = "答案"

    class FakeModel:
        supports_image_input = False

        async def stream_chat(self, messages: list[dict[str, object]], **kwargs: object):
            del messages, kwargs
            yield FakeChunk()

    monkeypatch.setattr(
        "app.generation.providers.create_chat_model_from_config",
        lambda config: FakeModel(),
    )

    call = agent_runtime.general_model_for_chat()
    assert call is not None, "注入假 provider 后装配点必须可用（隔离只挡真实凭证）"


@pytest.mark.allow_outbound
def test_bypassing_the_isolation_would_really_reach_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """最强非永真证据：把装配输入换回**真实**实现后，真实 provider 被构造出来、其调用被守卫拦下。

    步骤与含义：

    1. 换回真实 ``get_settings_store`` / ``_chat_config_from_settings`` ⇒ 读到 ``.env`` 的
       真实端点与密钥（隔离针对的就是这条路径）；
    2. ``_chat_provider_available`` 为真 ⇒ ``general_model_for_chat()`` 真的会返回可调用对象
       （不是永远 ``None`` 的永真断言）；
    3. **真的发起一次请求** ⇒ 守卫在解析/建连处抛 :class:`OutboundBlockedError`。

    因此本用例全程**没有数据离开本机**，却证明了「不隔离时这一步会真的出网」：删掉出网守卫
    后第 3 步会变成一次真实外呼（本工作流实测过一次 —— 守卫补上 ``getaddrinfo`` 咽喉之前，
    这一步真的拿到了远端 401 响应）。

    ``allow_outbound`` 只豁免"本轮不得有连接尝试"的记账断言：本用例要的正是那一次被拦下的
    尝试。
    """

    if not ENV_FILE.is_file():
        pytest.skip(f"本机没有 {ENV_FILE}（真实凭证不在仓库里），无法验证")

    # 撤掉两层隔离：设置存储 + 装配输入都换回真实实现。
    from app.core import settings_store as settings_store_mod

    real_store = settings_store_mod.get_settings_store.__wrapped__  # type: ignore[attr-defined]
    monkeypatch.setattr(settings_store_mod, "get_settings_store", real_store)
    real_config = agent_runtime._chat_config_from_settings.__wrapped__  # type: ignore[attr-defined]
    monkeypatch.setattr(agent_runtime, "_chat_config_from_settings", real_config)

    config = agent_runtime._chat_config_from_settings()
    assert config.base_url, "未隔离时必须能看到 .env 的真实端点（隔离针对的就是它）"
    assert agent_runtime._chat_provider_available(config) is True, (
        "未隔离时真实 provider 会被构造出来"
    )
    assert agent_runtime.general_model_for_chat() is not None

    with pytest.raises(netguard.OutboundBlockedError) as info:
        asyncio.run(agent_runtime._chat_text(config, "ping"))

    assert "非回环" in str(info.value)


# =====================================================================
# ③ live 用例仍能在显式开关下看见真实配置
# =====================================================================


def test_live_switch_disables_isolation() -> None:
    """子进程带 ``EVERYTHING_RAG_LIVE_EVAL=1`` 跑互补判据：三层隔离必须全部不装。

    子进程**不做任何外呼**：它只断言「隔离状态与 live 开关互补」。这条证据证明 live 用例
    （``test_acceptance_v120._live_settings()`` 只读真实 ``~/.everything-rag/config.json``、
    ``test_live_rag_accuracy`` 需要真实查询嵌入）不会被我加的隔离挡住。
    """

    env = {**os.environ, "EVERYTHING_RAG_LIVE_EVAL": "1"}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/test_netguard_scope.py",
            "-k",
            "follows_the_live_switch",
        ],
        cwd=BACKEND_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    output = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode == 0, output
    assert "1 passed" in output, output
