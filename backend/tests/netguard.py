"""测试卫生：全局隔离真实凭证与模型装配 + 阻断非回环出网（CLAUDE.md §11）。

**为什么需要**（本会话实测过两次真实外呼）：``backend/.env`` 存有**真实明文 provider
凭证**，而 ``SettingsStore.load()`` 在 ``config.json`` 缺失时会回退 ``.env``。测试进程里
``settings.data_dir`` 虽被隔离到 tmp，但「没有 config.json ⇒ 回退 .env」这条路让真实端点与
密钥仍然可见；任何未隔离的模型装配点都会据此构造出真实 provider 并**真的出网** ——
正是隐私红线「纯文本路径默认零外发」要挡住的事。

**三层隔离**（``conftest`` 默认全部装上）：

1. **设置存储**：``get_settings_store`` 指向 tmp 下**已写入空 ``config.json``** 的存储 ——
   显式写盘而不是留空文件，``load()`` 因此永不回退 ``.env``。所有持有该函数引用的模块一起
   替换（按 ``sys.modules`` 扫描），避免「某模块 import 时拿了旧引用」把凭证漏进来。
2. **模型装配输入**：``agent_runtime._chat_config_from_settings`` 返回**空**
   ``ChatModelConfig``，三个装配点（``general_model_for_chat`` / ``synthesis_model_for_chat``
   / ``model_verifier_for_chat``）因此都走它们各自文档化的「未配置 ⇒ None」分支：general 给
   结构化失败、synthesis 回落确定性渲染、verify 只剩程序侧检查。
   刻意**不**把那三个函数直接钉成 ``None``：验收类用例
   （``test_general_model_wiring`` / ``test_seam_assembly`` / ``test_lifespan_assembly``）靠
   patch ``create_chat_model_from_config`` 注入假模型并断言「已配置 ⇒ 可调用」，直接钉死会把
   那条真实装配路径的覆盖抹掉。隔离它们的**共同输入**既等效又不损失覆盖。
3. **出网守卫**：任何非回环 ``connect`` / ``connect_ex`` / ``create_connection`` 立即抛
   :class:`OutboundBlockedError`（``BaseException`` 子类 —— provider 内部常见的
   ``except Exception`` 不得把它吞成「提供方失败」），并记录到 :func:`blocked_attempts`；
   用例结束时由 ``conftest`` 兜底断言，即使某处 ``except BaseException`` 吞了也抓得到。

**live 豁免**：设 ``EVERYTHING_RAG_LIVE_EVAL=1``（或 ``EVERYTHING_RAG_RUN_LIVE_EVAL=1``）时
三层隔离**全部不装** —— live 用例是用户显式授权的真实配置 / 真实外呼路径
（``test_acceptance_v120.py`` 的 ``_live_settings()``、``test_live_rag_accuracy.py``）。
默认（未设）时 live 用例被各自 ``skipif`` 跳过，且隔离全开 ⇒ 不会外呼。
"""

from __future__ import annotations

import os
import socket
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

__all__ = [
    "ALLOW_OUTBOUND_MARKER",
    "LIVE_EVAL_ENVS",
    "OutboundBlockedError",
    "blocked_attempts",
    "chat_config_isolated",
    "clear_blocked_attempts",
    "install_chat_config_isolation",
    "install_isolation",
    "install_outbound_guard",
    "install_settings_isolation",
    "is_loopback",
    "isolated_settings_store",
    "isolation_active",
    "isolation_enabled",
    "live_eval_enabled",
    "outbound_guard_installed",
    "reset_isolation",
    "settings_store_isolated",
]

#: 显式授权真实配置 / 真实外呼的开关（与各 live 用例自己的 skipif 判据一致）。
LIVE_EVAL_ENVS = ("EVERYTHING_RAG_LIVE_EVAL", "EVERYTHING_RAG_RUN_LIVE_EVAL")

#: 仅**守卫自证**用例使用的标记：允许该用例故意触发一次被阻断的外呼而不算违规。
ALLOW_OUTBOUND_MARKER = "allow_outbound"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: 回环与「本机通配」地址：放行。Windows 的 ``socket.socketpair()`` 走 127.0.0.1，
#: 事件循环自管道与本地测试服务器都依赖它，因此回环**必须**放行。
_LOOPBACK_HOSTS = frozenset(
    {"127.0.0.1", "::1", "localhost", "localhost.", "0.0.0.0", "::", ""}
)

_blocked: list[dict[str, Any]] = []
_isolation_installed = False


def live_eval_enabled() -> bool:
    """是否处于「用户显式授权」的 live 运行（任一 lives 开关为真值）。"""

    return any(
        os.environ.get(name, "").strip().lower() in _TRUTHY for name in LIVE_EVAL_ENVS
    )


def isolation_enabled() -> bool:
    """本轮运行是否应当装上隔离：默认装，显式 live 运行不装（互补关系）。"""

    return not live_eval_enabled()


def isolation_active() -> bool:
    """本进程**当前**是否已装上隔离（供 ``test_netguard_scope`` 在两种运行方式下自证）。"""

    return _isolation_installed


#: 三层隔离各自打在替换对象上的标记（逐层自证用；等价于「这一层真的装上了」）。
_MARKER = "__netguard_isolated__"


def settings_store_isolated() -> bool:
    """设置存储入口是否已被替换成 tmp 空存储。"""

    from app.core import settings_store as settings_store_mod

    return bool(getattr(settings_store_mod.get_settings_store, _MARKER, False))


def chat_config_isolated() -> bool:
    """模型装配输入（``_chat_config_from_settings``）是否已被替换成空配置。"""

    from app.agent import runtime as agent_runtime

    return bool(getattr(agent_runtime._chat_config_from_settings, _MARKER, False))


def outbound_guard_installed() -> bool:
    """非回环连接守卫是否已装上。"""

    return bool(getattr(socket.socket.connect, _MARKER, False))


class OutboundBlockedError(BaseException):
    """测试进程里出现非回环连接尝试（默认运行禁止真实外呼）。

    刻意继承 ``BaseException``：provider 实现里普遍用 ``except Exception`` 把失败转成
    「提供方错误 / 走兜底」，那样会让一次真实外呼被静默吞掉、测试仍然全绿 —— 恰好是红线
    最坏的失败模式。``BaseException`` 只被显式的 ``except BaseException`` 捕获，
    而 ``conftest`` 结束时的记账断言连那种情况也能抓到。
    """


def blocked_attempts() -> list[dict[str, Any]]:
    """本轮已阻断的非回环连接尝试（host / port / 调用位置）。"""

    return [dict(item) for item in _blocked]


def clear_blocked_attempts() -> None:
    _blocked.clear()


def is_loopback(host: Any) -> bool:
    """地址是否属于回环（或本地通配）：字符串/字节/整数端口形式都按文本判定。"""

    if isinstance(host, bytes):
        host = host.decode("utf-8", "replace")
    text = str(host or "").strip().strip("[]").lower()
    if text in _LOOPBACK_HOSTS:
        return True
    return text.startswith("127.")


def _caller_hint() -> str:
    """触发点摘要（跳过本模块自身的栈帧），让失败信息直接指向写测试的人。"""

    frames = [
        frame
        for frame in traceback.extract_stack()
        if "tests/netguard.py" not in str(frame.filename or "")
    ]
    tail = frames[-3:]
    return " <- ".join(f"{Path(str(f.filename)).name}:{f.lineno}" for f in tail)


def _record(host: Any, port: Any) -> None:
    text = host.decode("utf-8", "replace") if isinstance(host, bytes) else str(host)
    _blocked.append({"host": text, "port": port, "where": _caller_hint()})


def _message(host: Any, port: Any) -> str:
    text = host.decode("utf-8", "replace") if isinstance(host, bytes) else host
    return (
        "测试试图连接非回环地址 "
        f"{text}:{port}（{_caller_hint()}）。默认测试运行**禁止**任何真实外呼"
        "（CLAUDE.md §11「纯文本路径默认零外发」）：请把该依赖换成替身 / "
        "httpx.MockTransport，或改用设置注入的假 provider；"
        f"确需真实外呼时用显式授权开关（{'/'.join(LIVE_EVAL_ENVS)}）跑 live 用例。"
    )


def _enforce_host(host: Any, port: Any) -> None:
    """主机名/地址层面的判定（``None`` = 本机，放行）。"""

    if host is None:
        return
    if is_loopback(host):
        return
    _record(host, port)
    raise OutboundBlockedError(_message(host, port))


def _enforce(address: Any) -> None:
    """``(host, port)`` 形式的地址；AF_UNIX 等非网络地址放行。"""

    if not isinstance(address, (tuple, list)) or not address:
        return
    _enforce_host(address[0], address[1] if len(address) > 1 else None)


def install_outbound_guard(monkeypatch: Any) -> None:
    """装上非回环出网守卫。

    **五个咽喉，缺一不可**（本工作流实测过一次真实外呼漏网）：

    * ``socket.getaddrinfo`` —— **主机名**解析总入口。Windows 的 proactor 事件循环用
      ``ConnectEx`` 在原始句柄上建连，会**绕过** Python 层的 ``socket.socket.connect``；
      但绕不过解析：``asyncio.BaseEventLoop.getaddrinfo`` 在调用时才取
      ``socket.getaddrinfo``，因此在这里拦得住 async 客户端（httpx / openai / anyio）。
    * ``socket.socket.connect`` / ``connect_ex`` —— 同步客户端与手写 socket。
    * ``socket.create_connection`` —— ``urllib`` / 同步 httpx 等。
    * ``asyncio.BaseEventLoop.create_connection`` / ``sock_connect`` —— 异步客户端的实际建连
      入口（anyio 走 ``loop.create_connection``），即使目标是 IP 字面量（跳过 DNS）也拦得住。
    """

    import asyncio

    original_getaddrinfo = socket.getaddrinfo
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection
    loop_cls = asyncio.base_events.BaseEventLoop
    original_loop_create_connection = loop_cls.create_connection
    original_loop_sock_connect = loop_cls.sock_connect

    def guarded_getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        _enforce_host(host, port)
        return original_getaddrinfo(host, port, *args, **kwargs)

    def guarded_connect(sock: Any, address: Any, *args: Any, **kwargs: Any) -> Any:
        _enforce(address)
        return original_connect(sock, address, *args, **kwargs)

    def guarded_connect_ex(sock: Any, address: Any, *args: Any, **kwargs: Any) -> Any:
        _enforce(address)
        return original_connect_ex(sock, address, *args, **kwargs)

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
        _enforce(address)
        return original_create_connection(address, *args, **kwargs)

    async def guarded_loop_create_connection(
        self: Any, protocol_factory: Any, host: Any = None, port: Any = None, **kwargs: Any
    ) -> Any:
        _enforce_host(host, port)
        return await original_loop_create_connection(
            self, protocol_factory, host, port, **kwargs
        )

    async def guarded_loop_sock_connect(self: Any, sock: Any, address: Any) -> Any:
        _enforce(address)
        return await original_loop_sock_connect(self, sock, address)

    guarded = (
        guarded_getaddrinfo,
        guarded_connect,
        guarded_connect_ex,
        guarded_create_connection,
        guarded_loop_create_connection,
        guarded_loop_sock_connect,
    )
    for item in guarded:
        setattr(item, _MARKER, True)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    monkeypatch.setattr(loop_cls, "create_connection", guarded_loop_create_connection)
    monkeypatch.setattr(loop_cls, "sock_connect", guarded_loop_sock_connect)


#: 离线安全的种子嵌入端点：**回环 + 丢弃端口**。任何误用都会在本机立刻连接失败，
#: 绝不指向真实服务（真实端点/密钥只在 ``.env`` 或用户 config.json 里，均被隔离挡在外面）。
SEED_EMBED_BASE_URL = "http://127.0.0.1:9/v1"
SEED_EMBED_MODEL = "offline-test-embed"
SEED_EMBED_KEY = "sk-offline-test-not-a-real-key"

#: 按数据目录缓存解析出的设置存储（同一路径返回同一实例，保持"单例"语义）。
_store_cache: dict[str, Any] = {}


def seed_settings() -> Any:
    """隔离用的**种子设置**：对话/检索/识图/筛选全部未配置，只有嵌入占位可构造。

    嵌入为什么不能留空：真实装配路径（``deps.get_workspace_vector_store``）会在解析向量库时
    构造嵌入器，配置为空会抛 ``ValueError``，让「用替身 provider 的装配类用例」因为与它们
    无关的原因失败。给一个**回环 + 丢弃端口**的占位端点即可满足构造，同时任何误用的嵌入
    调用都会在 127.0.0.1:9 上立刻失败（不出本机、也不会静默成功）。
    """

    from pydantic import SecretStr

    from app.core.settings_store import AppSettings, EmbedModelConfig

    return AppSettings(
        embed=EmbedModelConfig(
            base_url=SEED_EMBED_BASE_URL,
            model=SEED_EMBED_MODEL,
            api_key=SecretStr(SEED_EMBED_KEY),
        )
    )


def isolated_settings_store(_root: Path | None = None) -> Any:
    """**按当前 ``settings.data_dir``** 解析设置存储，并在缺 ``config.json`` 时种一份离线配置。

    刻意按调用时解析（而不是钉住夹具执行时的目录）：真实实现就是
    ``SettingsStore(get_settings().data_dir)``，晚解析让「自己把 ``settings.data_dir``
    指到 tmp 并写入配置」的用例（如 ``test_chat_rerank`` 的运行态装配用例）继续按其本意工作，
    而「没有配置文件」这条路径被**种子**堵住 ⇒ ``load()`` 永不回退 ``.env``。
    """

    from app.core.config import settings
    from app.core.settings_store import SettingsStore

    path = Path(str(settings.data_dir))
    cache_key = str(path)
    store = _store_cache.get(cache_key)
    if store is None:
        store = SettingsStore(path)
        if not store.path.is_file():
            store.save(seed_settings())
        _store_cache[cache_key] = store
    return store


def _settings_store_aliases(original: Any) -> list[tuple[ModuleType, str]]:
    """扫出**生产模块**里持有该函数引用的属性（``import`` 时取了旧引用的模块）。

    只扫 ``app.*``：测试模块也会 ``from app.core.settings_store import get_settings_store``，
    但那里它常被当作 **``dependency_overrides`` 的键**（身份必须与原函数一致），替换掉会让
    ``dependency_overrides[get_settings_store]`` 查不到而 KeyError。生产侧才是要挡的入口。

    用 ``module.__dict__.get`` 而不是 ``getattr``：既不触发 PEP 562 的模块级 ``__getattr__``
    （惰性代理可能抛），也就不需要 try/except。
    """

    found: list[tuple[ModuleType, str]] = []
    for module in list(sys.modules.values()):
        if module is None or not isinstance(module, ModuleType):
            continue
        if not str(module.__dict__.get("__name__", "")).startswith("app."):
            continue
        if module.__dict__.get("get_settings_store") is original:
            found.append((module, "get_settings_store"))
    return found


def install_settings_isolation(monkeypatch: Any, store: Any | None = None) -> int:
    """把 ``get_settings_store``（含所有 ``app.*`` 别名）指向隔离解析器；返回替换处数。

    ``store`` 传 None 时用 :func:`isolated_settings_store`（按调用时 ``settings.data_dir``
    解析 + 种离线配置），这样"晚导入"的生产代码与"早导入"的模块别名都拿到隔离结果。
    """

    from app.core import settings_store as settings_store_mod

    original = settings_store_mod.get_settings_store

    def replacement() -> Any:
        return store if store is not None else isolated_settings_store()

    # 既有夹具会调用 `deps.get_settings_store.cache_clear()`：保留同名方法避免 AttributeError。
    replacement.cache_clear = original.cache_clear  # type: ignore[attr-defined]
    replacement.__wrapped__ = original  # type: ignore[attr-defined]
    setattr(replacement, _MARKER, True)

    replaced = 0
    for module, attribute in _settings_store_aliases(original):
        monkeypatch.setattr(module, attribute, replacement)
        replaced += 1
    if replaced == 0:  # pragma: no cover - 只在导入结构被彻底改写时发生
        monkeypatch.setattr(settings_store_mod, "get_settings_store", replacement)
        replaced = 1
    return replaced


def install_chat_config_isolation(monkeypatch: Any) -> None:
    """把三个模型装配点的**共同输入**钉成空配置（⇒ 各自走「未配置 ⇒ None」分支）。"""

    from app.agent import runtime as agent_runtime
    from app.core.settings_store import ChatModelConfig

    original = agent_runtime._chat_config_from_settings

    def empty_chat_config() -> Any:
        return ChatModelConfig()

    setattr(empty_chat_config, _MARKER, True)
    # 保留原实现引用：自证用例可以拿它换回"真实配置"路径，验证隔离针对的确实是真实风险。
    empty_chat_config.__wrapped__ = original  # type: ignore[attr-defined]
    monkeypatch.setattr(agent_runtime, "_chat_config_from_settings", empty_chat_config)


def install_isolation(monkeypatch: Any, tmp_path: Path | None = None) -> int:
    """装上三层隔离（设置存储 / 模型装配输入 / 出网守卫），返回被替换的设置入口数。"""

    global _isolation_installed
    del tmp_path  # 存储按调用时的 settings.data_dir 解析，见 isolated_settings_store
    replaced = install_settings_isolation(monkeypatch)
    install_chat_config_isolation(monkeypatch)
    install_outbound_guard(monkeypatch)
    clear_blocked_attempts()
    _isolation_installed = True
    return replaced


def reset_isolation() -> None:
    """用例结束复位「本轮是否装过隔离」标志（供 scope 自证用例在两种运行下都成立）。"""

    global _isolation_installed
    _isolation_installed = False
