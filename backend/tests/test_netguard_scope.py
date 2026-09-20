"""隔离作用域自证：三层隔离的**装/不装**必须与 live 开关严格互补。

本文件只有一条判据（见 :func:`test_isolation_state_follows_the_live_switch`），它在**两种**
运行方式下都成立，因此可以作为「live 开关真的会关掉三层隔离」的实证：

* 默认运行（父进程，正常全量收集）：开关关 ⇒ 三层隔离**必须**都已装上；
* 显式 live 运行：`tests/test_network_guard.py::test_live_switch_disables_isolation`
  用子进程跑本文件的这条判据并带上 ``EVERYTHING_RAG_LIVE_EVAL=1``，此时开关开 ⇒
  三层隔离**必须**都没装（否则 live 用例的 ``_live_settings()`` / 真实外呼会被挡）。

逐层检查（而不是只看一个全局标志）才是有效证据：只要有一层与开关不互补，就说明
``conftest`` 的判据被改坏了，两种运行方式里必有一种会红。
"""

from __future__ import annotations

from tests import netguard


def test_isolation_state_follows_the_live_switch() -> None:
    """三层隔离的装/不装 == 非 live 开关（互补关系）。"""

    live = netguard.live_eval_enabled()
    expected = not live
    layers = {
        "isolation_active": netguard.isolation_active(),
        "settings_store_isolated": netguard.settings_store_isolated(),
        "chat_config_isolated": netguard.chat_config_isolated(),
        "outbound_guard_installed": netguard.outbound_guard_installed(),
    }
    wrong = {name: value for name, value in layers.items() if value is not expected}
    assert not wrong, (
        f"live_eval_enabled()={live} ⇒ 三层隔离应全部为 {expected}，实际不符：{wrong}"
    )


def test_default_run_has_no_live_switch() -> None:
    """默认（未显式授权）运行不得被误判成 live：否则隔离会被静默跳过。"""

    assert netguard.live_eval_enabled() is False, (
        "本文件应在默认运行（未设 "
        f"{'/'.join(netguard.LIVE_EVAL_ENVS)}）下被收集"
    )
