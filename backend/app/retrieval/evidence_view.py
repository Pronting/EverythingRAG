"""证据视图（EvidenceView）：评分与引用的最小对象（PRD RAG-REL-001 §5.3/§5.4）。

一个视图 = 原始 block（或明确关联的一组原文片段）+ 可重复构造的评分文本 + 校验哈希。
本模块只做「原文切片 + 固定模板 + 哈希」，绝不改写、摘要或补写事实：

* ``evidence_text`` 是实际进入答案上下文与来源面板的原文，逐字来自原始 block；
* ``ranking_text`` 是名称/标题/原文的固定模板，可离线重复构造；
* ``payload_hash`` 绑定 ``ranking_text`` 与视图版本，随原文或模板变化而变化；
* ``candidate_id`` 由 scope、成员引用（block ID、范围、内容哈希）、视图类型与视图版本
  共同决定，跨进程稳定且同一 chunk 的不同视图不会撞 ID。

与 T03 类型层的关系：``EvidenceView`` 在结构上直接满足 ``reranker.RerankScorable``
（``candidate_id`` / ``scope_id`` / ``index_revision`` / ``ranking_text`` / ``payload_hash``），
因此 ``RerankRequest`` 可以直接吃视图对象，不需要适配层；通道信号复用
``reranker.RouteSignals``，不另建同义类型。

外呼安全：``display_name`` 只保存 basename，任何绝对路径都不进入评分文本或字典序列化。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.retrieval.markdown_structure import MIN_SEGMENT_CHARS, structure_segments
from app.retrieval.reranker import RerankValidationError, RouteSignals, ViewKind

__all__ = [
    "MEMBERSHIP_IMAGE",
    "MEMBERSHIP_PRIMARY",
    "MEMBERSHIP_SUPPLEMENT",
    "MIN_SPLIT_CHARS",
    "RELATION_TYPES",
    "VIEW_VERSION",
    "EvidenceSlice",
    "EvidenceView",
    "HeadingLookup",
    "MemberRef",
    "RetrievalScope",
    "ViewBuildReport",
    "build_candidate_views",
    "build_candidate_views_with_report",
    "build_chunk_views",
    "payload_hash_for",
    "ranking_text_for",
    "scope_id_for",
]

#: 证据视图版本（PRD §5.3）：成员范围、模板或构造规则变化必须递增；
#: 「评分文本可重复构造」的验收依赖它，档案失配检查也读这个常量。
VIEW_VERSION = "evidence-view-v1"

#: 关联类型：``primary`` 是主要原文；``supplement`` 是同 scope/同来源/同精确标题的补充原文；
#: ``image_text`` 是图片描述文字。补充内容必须显式成成员，不能隐式追加。
MEMBERSHIP_PRIMARY = "primary"
MEMBERSHIP_SUPPLEMENT = "supplement"
MEMBERSHIP_IMAGE = "image_text"
RELATION_TYPES = (MEMBERSHIP_PRIMARY, MEMBERSHIP_SUPPLEMENT, MEMBERSHIP_IMAGE)

#: 生成较小引用视图的最小分段高度（PRD §5.3 的「必要时」）：低于该值不拆分，
#: 避免候选窗口被同一 chunk 的碎片挤占，也避免猜测主题边界。
#: 与 ``markdown_structure.MIN_SEGMENT_CHARS`` 保持同一个值，避免两处规则漂移。
MIN_SPLIT_CHARS = MIN_SEGMENT_CHARS

#: 评分文本模板的固定标题行（改这里必须递增 :data:`VIEW_VERSION`）。
RANKING_NAME_LABEL = "来源"
RANKING_HEADING_LABEL = "标题"
RANKING_TEXT_LABEL = "原文"

_SHA256_HEX_RE = re.compile(r"\A[0-9a-fA-F]{64}\Z")


# --------------------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    """文本的 SHA-256（UTF-8，小写十六进制）。"""

    if not isinstance(text, str):
        raise RerankValidationError("哈希输入必须是字符串")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_key(parts: Iterable[str]) -> str:
    """多段独立值的 SHA-256。

    每个部分先写成 ``长度:内容`` 再拼接，避免 ``("ab", "c")`` 与 ``("a", "bc")``
    这类边界歧义产生相同哈希。
    """

    buffer = bytearray()
    for part in parts:
        if not isinstance(part, str):
            raise RerankValidationError("哈希输入必须是字符串")
        encoded = part.encode("utf-8")
        buffer += str(len(encoded)).encode("ascii") + b":" + encoded + b"\n"
    return hashlib.sha256(bytes(buffer)).hexdigest()


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RerankValidationError(f"{name} 必须是非空字符串")
    return value


def _require_span(start: object, end: object, name: str) -> tuple[int, int]:
    for label, value in (("start", start), ("end", end)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise RerankValidationError(f"{name}.{label} 必须是整数")
    if start < 0 or end <= start:
        raise RerankValidationError(f"{name} 必须是 start>=0 且 end>start 的半开区间")
    return start, end


def _require_sha256(value: object, name: str) -> str:
    text = _require_text(value, name)
    if not _SHA256_HEX_RE.match(text):
        raise RerankValidationError(f"{name} 必须是 64 位十六进制 SHA-256")
    return text


def display_name_for(source_file: str | None) -> str:
    """外呼可见的来源名称：只取 basename，绝不传本机绝对路径（PRD §5.3）。"""

    raw = (source_file or "").strip()
    if not raw:
        return ""
    name = Path(raw.replace("\\", "/")).name
    return name[: -len(".md")] if name.lower().endswith(".md") else name


def scope_id_for(where: Mapping[str, Any] | None) -> str:
    """由调用方已解析的 ``where`` 生成稳定的 scope 标识（PRD §3.2/R13）。

    当前代码仍使用全局知识库时显式标记 ``global``；本函数不猜测工作区隔离是否完成。
    """

    if not where:
        return "global"
    encoded = ";".join(
        f"{key}={_scope_value(value)}" for key, value in sorted(where.items(), key=lambda item: str(item[0]))
    )
    return f"where:{sha256_key([encoded])[:16]}"


def _scope_value(value: Any) -> str:
    if isinstance(value, Mapping):
        return "{" + ",".join(f"{key}={_scope_value(item)}" for key, item in sorted(value.items())) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_scope_value(item) for item in value) + "]"
    return str(value)


def index_revision_for(vectorstore: Any, override: str | None = None) -> str:
    """索引版本：显式覆盖优先，否则读向量库的 collection 名，再退回 ``unknown``。

    查询阶段只做标识与校验，不修改任何索引。
    """

    if override:
        return _require_text(override, "index_revision")
    for attribute_path in ("collection_name", "collection.name", "_collection.name"):
        value: Any = vectorstore
        for part in attribute_path.split("."):
            value = getattr(value, part, None)
            if value is None:
                break
        if isinstance(value, str) and value.strip():
            return value
    return "unknown-index"


# --------------------------------------------------------------------------------------
# scope 与候选
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalScope:
    """一次检索的资料范围与知识版本（PRD §3.2/R13）。

    调用方必须传入已经解析好的 scope；候选、补充、图片说明、诊断与配置档案都继承它。
    """

    scope_id: str
    index_revision: str

    def __post_init__(self) -> None:
        _require_text(self.scope_id, "scope_id")
        _require_text(self.index_revision, "index_revision")

    @classmethod
    def from_where(
        cls,
        where: Mapping[str, Any] | None = None,
        *,
        index_revision: str,
    ) -> RetrievalScope:
        return cls(scope_id=scope_id_for(where), index_revision=index_revision)

    @classmethod
    def for_vectorstore(
        cls,
        vectorstore: Any,
        where: Mapping[str, Any] | None = None,
        *,
        index_revision: str | None = None,
    ) -> RetrievalScope:
        return cls(
            scope_id=scope_id_for(where),
            index_revision=index_revision_for(vectorstore, index_revision),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"scope_id": self.scope_id, "index_revision": self.index_revision}


@dataclass(frozen=True)
class RetrievalCandidate:
    """待转为证据视图的逻辑候选（T02 ``collect_candidates`` 的记录化投影）。

    只保存身份与信号：正文在构造视图时才读取，保证「未读到的内容不会进入评分文本」。
    """

    candidate_id: str
    block_id: str
    scope: RetrievalScope
    route_signals: RouteSignals | None = None
    view_kind: ViewKind = ViewKind.BLOCK

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        _require_text(self.block_id, "block_id")
        if not isinstance(self.scope, RetrievalScope):
            raise RerankValidationError("scope 必须是 RetrievalScope")
        if self.route_signals is not None and not isinstance(self.route_signals, RouteSignals):
            raise RerankValidationError("route_signals 必须是 RouteSignals")
        object.__setattr__(self, "view_kind", ViewKind(self.view_kind))


# --------------------------------------------------------------------------------------
# 成员引用与证据视图
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MemberRef:
    """一个成员原文片段：block 身份 + 原文半开区间 + 内容哈希 + 关联类型。

    ``start``/``end`` 使用 Python Unicode code point 的半开区间 ``[start, end)``，
    相对**该 block 的原始正文**记录，并用原文重新切片核验。
    """

    block_id: str
    content_hash: str
    relation: str = MEMBERSHIP_PRIMARY
    start: int = 0
    end: int | None = None
    block_char_offset: int | None = None

    def __post_init__(self) -> None:
        _require_text(self.block_id, "member block_id")
        _require_sha256(self.content_hash, "member content_hash")
        if self.relation not in RELATION_TYPES:
            raise RerankValidationError(
                f"member relation 必须是 {RELATION_TYPES}，得到 {self.relation!r}"
            )
        if self.end is None:
            # 整段成员：范围即整个原始 block，content_hash 覆盖全部原文。
            if self.start != 0:
                raise RerankValidationError("整段成员（end 省略）的 start 必须为 0")
            if self.block_char_offset is not None and self.block_char_offset != 0:
                raise RerankValidationError("整段成员的 block_char_offset 必须为 0")
        else:
            start, end = _require_span(self.start, self.end, "member span")
            object.__setattr__(self, "start", start)
            object.__setattr__(self, "end", end)
        if self.block_char_offset is not None:
            if isinstance(self.block_char_offset, bool) or not isinstance(
                self.block_char_offset, int
            ):
                raise RerankValidationError("member block_char_offset 必须是整数")
            if self.block_char_offset < 0:
                raise RerankValidationError("member block_char_offset 不能为负")

    @property
    def length(self) -> int:
        """成员长度；``end is None`` 表示整段原文（长度由调用方的原始正文决定）。"""

        return 0 if self.end is None else self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "content_hash": self.content_hash,
            "relation": self.relation,
            "start": self.start,
            "end": self.end,
            "block_char_offset": self.block_char_offset,
        }


@dataclass(frozen=True)
class EvidenceSlice:
    """视图内部的一段连续原文：有序拼接即得 ``evidence_text``。"""

    block_id: str
    start: int
    end: int
    content_hash: str

    def __post_init__(self) -> None:
        _require_text(self.block_id, "slice block_id")
        start, end = _require_span(self.start, self.end, "slice span")
        _require_sha256(self.content_hash, "slice content_hash")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "start": self.start,
            "end": self.end,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class EvidenceView:
    """评分与引用的最小对象（PRD §5.3）。

    结构上满足 ``reranker.RerankScorable``：``candidate_id`` / ``scope_id`` /
    ``index_revision`` / ``ranking_text`` / ``payload_hash`` 可直接用于 ``RerankRequest``。
    """

    candidate_id: str
    scope_id: str
    index_revision: str
    primary_block_id: str
    display_name: str
    heading_path: str
    evidence_text: str
    ranking_text: str
    payload_hash: str
    slices: tuple[EvidenceSlice, ...]
    slice_texts: tuple[str, ...] = ()
    member_refs: tuple[MemberRef, ...] = ()
    route_signals: RouteSignals | None = None
    view_kind: ViewKind = ViewKind.BLOCK
    block_id: str = ""
    source_file: str = ""
    similarity: float | None = None
    match_type: str = ""
    view_version: str = VIEW_VERSION

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        _require_text(self.scope_id, "scope_id")
        _require_text(self.index_revision, "index_revision")
        _require_text(self.primary_block_id, "primary_block_id")
        _require_text(self.evidence_text, "evidence_text")
        _require_text(self.ranking_text, "ranking_text")
        _require_sha256(self.payload_hash, "payload_hash")
        _require_text(self.view_version, "view_version")
        if not isinstance(self.display_name, str):
            raise RerankValidationError("display_name 必须是字符串")
        if not isinstance(self.heading_path, str):
            raise RerankValidationError("heading_path 必须是字符串")
        # 名称只允许 basename；出现路径分隔符说明调用方传了绝对路径。
        if self.display_name and ("/" in self.display_name or "\\" in self.display_name):
            raise RerankValidationError("display_name 只能是 basename，不能包含路径")
        if not self.slices:
            raise RerankValidationError("证据视图至少需要一个原文片段")
        for item in self.slices:
            if not isinstance(item, EvidenceSlice):
                raise RerankValidationError("slices 必须是 EvidenceSlice 序列")
        for member in self.member_refs:
            if not isinstance(member, MemberRef):
                raise RerankValidationError("member_refs 必须是 MemberRef 序列")
        if self.route_signals is not None and not isinstance(
            self.route_signals, RouteSignals
        ):
            raise RerankValidationError("route_signals 必须是 RouteSignals")
        object.__setattr__(self, "view_kind", ViewKind(self.view_kind))
        object.__setattr__(self, "slices", tuple(self.slices))
        object.__setattr__(self, "member_refs", tuple(self.member_refs))
        slice_texts = tuple(self.slice_texts)
        if not slice_texts:
            raise RerankValidationError(
                "证据视图必须携带已核验的切片正文（slice_texts）"
            )
        if len(slice_texts) != len(self.slices):
            raise RerankValidationError("slice_texts 必须与 slices 一一对应")
        for piece, item in zip(slice_texts, self.slices, strict=True):
            if not isinstance(piece, str) or not piece:
                raise RerankValidationError("切片正文不能为空")
            if sha256_text(piece) != item.content_hash:
                raise RerankValidationError("切片内容哈希与原文不一致")
        object.__setattr__(self, "slice_texts", slice_texts)
        if "".join(slice_texts) != self.evidence_text:
            raise RerankValidationError("evidence_text 必须等于有序切片正文的拼接结果")
        if payload_hash_for(self.ranking_text, self.view_version, self.candidate_id) != (
            self.payload_hash
        ):
            raise RerankValidationError("payload_hash 与 ranking_text / 视图版本不一致")

    @property
    def text(self) -> str:
        """送入生成上下文与来源面板的原文（PRD §5.3/§8.3）。"""

        return self.evidence_text

    @property
    def primary_span(self) -> tuple[int, int]:
        """主片段在其原始 block 中的半开区间 ``[start, end)``。"""

        return self.slices[0].start, self.slices[0].end

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        """完整序列化；``include_text=False`` 用于不需要正文的诊断/日志路径。"""

        data: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "scope_id": self.scope_id,
            "index_revision": self.index_revision,
            "primary_block_id": self.primary_block_id,
            "display_name": self.display_name,
            "heading_path": self.heading_path,
            "payload_hash": self.payload_hash,
            "view_kind": str(self.view_kind),
            "view_version": self.view_version,
            "member_refs": [member.to_dict() for member in self.member_refs],
            "slices": [item.to_dict() for item in self.slices],
            "route_signals": self.route_signals.to_dict() if self.route_signals else None,
        }
        if include_text:
            data["evidence_text"] = self.evidence_text
            data["ranking_text"] = self.ranking_text
        return data


def payload_hash_for(ranking_text: str, view_version: str = VIEW_VERSION, candidate_id: str = "") -> str:
    """``payload_hash`` = SHA-256(视图版本, candidate_id, ranking_text)。

    绑定 candidate_id 让哈希同时表达「同一视图身份 + 该身份下的确切文本」，
    模板版本变化、原文变化或身份变化都会改变哈希（PRD §5.3/A07）。
    """

    return sha256_key([view_version, candidate_id, ranking_text])


def ranking_text_for(display_name: str, heading_path: str, evidence_text: str) -> str:
    """固定评分模板：名称 / 标题 / 原文三段，可离线重复构造（PRD §5.3）。"""

    return (
        f"{RANKING_NAME_LABEL}：{display_name}\n"
        f"{RANKING_HEADING_LABEL}：{heading_path}\n"
        f"{RANKING_TEXT_LABEL}：\n{evidence_text}"
    )


# --------------------------------------------------------------------------------------
# 构造器
# --------------------------------------------------------------------------------------


def _chunk_text(chunk: Any) -> str:
    text = getattr(chunk, "text", None)
    if not isinstance(text, str) or not text:
        raise RerankValidationError("候选正文为空，不能在证据视图里补写内容")
    return text


def _chunk_view_kind(chunk: Any) -> ViewKind:
    chunk_type = getattr(chunk, "chunk_type", "") or ""
    return ViewKind.IMAGE_TEXT if chunk_type == "image_description" else ViewKind.BLOCK


def _make_candidate_id(
    scope: RetrievalScope,
    view_kind: ViewKind,
    member_refs: Sequence[MemberRef],
    view_version: str,
) -> str:
    """稳定 candidate_id：由 scope、成员 ID、成员范围、关联类型、内容哈希与视图版本生成。"""

    key = [
        view_version,
        scope.scope_id,
        scope.index_revision,
        str(view_kind),
    ]
    for member in member_refs:
        key.extend(
            (
                member.block_id,
                str(member.start),
                str(member.end),
                member.relation,
                member.content_hash,
            )
        )
    digest = sha256_key(key)[:32]
    return f"ev1-{digest}-{member_refs[0].block_id}" if member_refs else f"ev1-{digest}"


def _build_view(
    *,
    scope: RetrievalScope,
    primary_block_id: str,
    slices: Sequence[EvidenceSlice],
    slice_texts: Sequence[str],
    member_refs: Sequence[MemberRef],
    display_name: str,
    heading_path: str,
    route_signals: RouteSignals | None,
    view_kind: ViewKind,
    source_file: str,
    similarity: float | None,
    match_type: str,
) -> EvidenceView:
    evidence_text = "".join(slice_texts)
    candidate_id = _make_candidate_id(scope, view_kind, member_refs, VIEW_VERSION)
    ranking_text = ranking_text_for(display_name, heading_path, evidence_text)
    return EvidenceView(
        candidate_id=candidate_id,
        scope_id=scope.scope_id,
        index_revision=scope.index_revision,
        primary_block_id=primary_block_id,
        display_name=display_name,
        heading_path=heading_path,
        evidence_text=evidence_text,
        ranking_text=ranking_text,
        payload_hash=payload_hash_for(ranking_text, VIEW_VERSION, candidate_id),
        slices=tuple(slices),
        slice_texts=tuple(slice_texts),
        member_refs=tuple(member_refs),
        route_signals=route_signals,
        view_kind=view_kind,
        block_id=primary_block_id,
        source_file=source_file,
        similarity=similarity,
        match_type=match_type,
    )


@dataclass(frozen=True)
class _ViewContext:
    """构造一个 block 的所有视图时共享的上下文（名称、标题、通道信号等）。"""

    block_id: str
    text: str
    display_name: str
    heading_path: str
    view_kind: ViewKind
    source_file: str
    similarity: float | None
    match_type: str
    route_signals: RouteSignals | None


class HeadingLookup:
    """按 block_id 取标题路径的权威来源（PRD §5.3）。

    真实索引里 ``heading_path`` 是块元数据字段（``BlockMetadata.heading_path`` → Chroma
    元数据，经 ``_serialize_metadata`` 直传），但候选/视图对象并非都携带它。本工具直接从
    块元数据读取标题：

    * 找不到该 block，或该 block 确实没有标题时返回 ``""``——**绝不猜测**：不从正文首行
      截取、不拼路径、不推断层级（PRD §5.3「不能用猜测修改内容」）；
    * 缺失区分三种来源，``stats()`` 分别计数，用来判断「是 ingestion 没写键，还是写了空值，
      还是调用方漏传了查表」：``missing_key``（元数据没有该键 / 未知 block）、
      ``blank_value``（键存在但为空串，或键存在而块无标题）、``lookup_absent``（调用方根本
      没传查表——这是隐性口径分叉，必须能被发现）。

    实测活动索引：4,719 块中 4,532 有标题、187 块**缺键**且没有空串（55 图片描述 + 132 无
    标题文档块）。标题缺失是数据事实，不是错误；但「有标题却取不到」必须可诊断。
    """

    #: 内部哨兵：区分「键缺失」与「键存在但值为空」。
    _MISSING = object()

    def __init__(self, headings: Mapping[str, str] | None = None) -> None:
        self._headings: dict[str, str] = {
            str(block_id): str(heading).strip()
            for block_id, heading in (headings or {}).items()
        }
        #: 构造时是否收到过原始块清单（用于区分「缺键」与「空值」）。
        self._sourced = headings is not None
        self.queried = 0
        self.resolved = 0
        self.missing_key = 0
        self.blank_value = 0
        self.lookup_absent = 0

    @classmethod
    def from_blocks(
        cls,
        blocks: Iterable[tuple[str, str, Mapping[str, Any]]],
    ) -> HeadingLookup:
        """由 ``[(block_id, text, metadata)]`` 构造（``vectorstore.list_blocks()`` 的形状）。"""

        headings: dict[str, str] = {}
        for block_id, _text, metadata in blocks:
            heading = metadata.get("heading_path") if isinstance(metadata, Mapping) else None
            headings[str(block_id)] = heading.strip() if isinstance(heading, str) else ""
        return cls(headings)

    @classmethod
    def for_vectorstore(cls, vectorstore: Any) -> HeadingLookup:
        """从向量库读取全部块的标题（只读，不修改索引）。"""

        list_blocks = getattr(vectorstore, "list_blocks", None)
        if not callable(list_blocks):
            return cls({})
        return cls.from_blocks(list_blocks())

    def __len__(self) -> int:
        return sum(1 for heading in self._headings.values() if heading)

    def lookup(self, block_id: str) -> str:
        """按 block_id 取标题并记录缺失类别。"""

        self.queried += 1
        value = self._headings.get(str(block_id), self._MISSING)
        if value is self._MISSING:
            self.missing_key += 1
            return ""
        text = str(value).strip()
        if not text:
            self.blank_value += 1
            return ""
        self.resolved += 1
        return text

    def heading_for(self, block_id: str) -> str:
        """只读查表：不改变缺失统计（批量构造时同一块会被查询多次）。"""

        value = self._headings.get(str(block_id), "")
        return str(value).strip() if value is not self._MISSING else ""

    def note_absent(self) -> None:
        """调用方未提供查表却遇到需要标题的块：记一次隐性口径分叉诊断。"""

        self.lookup_absent += 1

    @property
    def missing(self) -> int:
        """查询过但没有标题的块数（缺键 + 空值 + 未知 block）。"""

        return self.missing_key + self.blank_value

    def stats(self) -> dict[str, int]:
        return {
            "queried": self.queried,
            "resolved": self.resolved,
            "missing": self.missing,
            "missing_key": self.missing_key,
            "blank_value": self.blank_value,
            "lookup_absent": self.lookup_absent,
        }


def _resolve_heading(
    block_id: str,
    candidate: Any,
    lookup: HeadingLookup | None,
) -> str:
    """先取候选自身携带的标题，缺失时查权威来源；都缺则留空（不猜）。"""

    heading = getattr(candidate, "heading_path", None)
    if isinstance(heading, str) and heading.strip():
        return heading.strip()
    if lookup is None:
        return ""
    return lookup.lookup(block_id)


def _chunk_context(chunk: Any, heading_lookup: HeadingLookup | None = None) -> _ViewContext:
    text = _chunk_text(chunk)
    block_id = _require_text(getattr(chunk, "block_id", ""), "chunk block_id")
    route_signals = getattr(chunk, "route_signals", None)
    if route_signals is not None and not isinstance(route_signals, RouteSignals):
        raise RerankValidationError("route_signals 必须是 RouteSignals")
    return _ViewContext(
        block_id=block_id,
        text=text,
        display_name=display_name_for(getattr(chunk, "source_file", "")),
        heading_path=_resolve_heading(block_id, chunk, heading_lookup),
        view_kind=_chunk_view_kind(chunk),
        source_file=str(getattr(chunk, "source_file", "") or ""),
        similarity=_optional_number(getattr(chunk, "similarity", None)),
        match_type=str(getattr(chunk, "match_type", "") or ""),
        route_signals=route_signals,
    )


def _segment_view(
    scope: RetrievalScope,
    context: _ViewContext,
    start: int,
    end: int,
) -> EvidenceView:
    """把原文区间 ``[start, end)`` 构造成一个可核验的引用视图。"""

    piece = context.text[start:end]
    piece_hash = sha256_text(piece)
    return _build_view(
        scope=scope,
        primary_block_id=context.block_id,
        slices=(
            EvidenceSlice(
                block_id=context.block_id,
                start=start,
                end=end,
                content_hash=piece_hash,
            ),
        ),
        slice_texts=(piece,),
        member_refs=(
            MemberRef(
                block_id=context.block_id,
                content_hash=piece_hash,
                relation=MEMBERSHIP_PRIMARY,
                start=start,
                end=end,
                block_char_offset=start,
            ),
        ),
        display_name=context.display_name,
        heading_path=context.heading_path,
        route_signals=context.route_signals,
        view_kind=context.view_kind,
        source_file=context.source_file,
        similarity=context.similarity,
        match_type=context.match_type,
    )


def build_chunk_views(
    chunk: Any,
    scope: RetrievalScope,
    *,
    include_structure_views: bool = True,
    min_split_chars: int = MIN_SPLIT_CHARS,
    heading_lookup: HeadingLookup | None = None,
) -> tuple[EvidenceView, ...]:
    """由一个检索结果构造证据视图：主视图在前，其后是较小的结构化引用视图。

    * 主视图覆盖整个 block，原文逐字来自 ``chunk.text``；
    * 结构化引用视图只在可确认的结构边界处切分，切分范围是原文的连续区间，
      拼接后与主视图正文完全相同——绝不改写、摘要或补写（PRD §5.3）；
    * 图片文字块是原子证据，不生成结构化引用视图（PRD §5.5）；
    * ``heading_lookup`` 用于补全候选对象未携带的标题：真实索引的标题存在块元数据的
      ``heading_path`` 字段里，调用方（如 ChatService）应传
      ``HeadingLookup.for_vectorstore(store)``，让 ``ranking_text`` 带上标题这一维判别信号。
      查不到时留空并计入 ``lookup.stats()``，绝不猜标题。
    """

    if not isinstance(scope, RetrievalScope):
        raise RerankValidationError("scope 必须是 RetrievalScope")
    context = _chunk_context(chunk, heading_lookup)
    views = [_segment_view(scope, context, 0, len(context.text))]

    if (
        not include_structure_views
        or context.view_kind is ViewKind.IMAGE_TEXT
        or len(context.text) < max(1, min_split_chars)
    ):
        return tuple(views)

    segments = structure_segments(context.text)
    if len(segments) < 2 or any(
        end - start < min_split_chars for start, end in segments
    ):
        # 结构边界不可靠，或某个分段过短：保留完整分块，不生成拆碎后的引用视图。
        return tuple(views)

    views.extend(_segment_view(scope, context, start, end) for start, end in segments)
    return tuple(views)


def _optional_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


@dataclass(frozen=True)
class _CandidateChunk:
    """把候选记录投影成 ``build_chunk_views`` 需要的形状（内部使用）。"""

    block_id: str
    text: str
    chunk_type: str = "text"
    source_file: str = ""
    heading_path: str = ""
    similarity: float | None = None
    match_type: str = ""
    route_signals: RouteSignals | None = None


@dataclass
class ViewBuildReport:
    """一次「候选 → 视图」构造的标题诊断（可变累加器，构造过程中填充）。"""

    chunks: int = 0
    views: int = 0
    from_candidate: int = 0
    from_lookup: int = 0
    heading_lookup_absent: int = 0
    lookup_provided: bool = True

    @property
    def heading_missing(self) -> int:
        """既没有候选自带标题、查表也没有结果的块数。"""

        return max(0, self.chunks - self.from_candidate - self.from_lookup)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks": self.chunks,
            "views": self.views,
            "from_candidate": self.from_candidate,
            "from_lookup": self.from_lookup,
            "heading_missing": self.heading_missing,
            "heading_lookup_absent": self.heading_lookup_absent,
            "lookup_provided": self.lookup_provided,
        }


def build_candidate_views_with_report(
    candidates: Iterable[Any],
    scope: RetrievalScope,
    *,
    block_texts: Mapping[str, str] | None = None,
    chunk_factory: Callable[[Any], Any] | None = None,
    include_structure_views: bool = True,
    min_split_chars: int = MIN_SPLIT_CHARS,
    heading_lookup: HeadingLookup | None = None,
) -> tuple[tuple[EvidenceView, ...], ViewBuildReport]:
    """``build_candidate_views`` 的诊断版本：同时返回标题补齐情况。

    「候选有 block_id 但调用方没传 ``heading_lookup``」不再静默：计入
    ``report.heading_lookup_absent``，让线上/评测的口径分叉能被发现。
    """

    report = ViewBuildReport(lookup_provided=heading_lookup is not None)
    views = build_candidate_views(
        candidates,
        scope,
        block_texts=block_texts,
        chunk_factory=chunk_factory,
        include_structure_views=include_structure_views,
        min_split_chars=min_split_chars,
        heading_lookup=heading_lookup,
        report=report,
    )
    return views, report


def build_candidate_views(
    candidates: Iterable[Any],
    scope: RetrievalScope,
    *,
    block_texts: Mapping[str, str] | None = None,
    chunk_factory: Callable[[Any], Any] | None = None,
    include_structure_views: bool = True,
    min_split_chars: int = MIN_SPLIT_CHARS,
    heading_lookup: HeadingLookup | None = None,
    report: ViewBuildReport | None = None,
) -> tuple[EvidenceView, ...]:
    """把候选记录批量转成证据视图（在线与评测共用同一入口）。

    * ``chunk_factory`` 负责把候选记录还原成带 ``text`` 的对象（默认直接取候选本身）；
    * ``block_texts`` 提供兜底正文，便于候选只携带 ``block_id`` 的调用方；
    * ``heading_lookup`` 在候选记录未携带 ``heading_path`` 时从块元数据补齐标题
      （真实索引的标题就存在元数据里），缺失留空并计入 ``lookup.stats()``；
    * ``report`` 非空时记录标题补齐情况，``heading_lookup_absent`` 专门暴露
      「候选有 block_id 但调用方漏传查表」的隐性口径分叉（用
      :func:`build_candidate_views_with_report` 可直接拿到报告）。
    """

    views: list[EvidenceView] = []
    for candidate in candidates:
        chunk = chunk_factory(candidate) if chunk_factory is not None else candidate
        text = getattr(chunk, "text", None)
        if not isinstance(text, str) or not text:
            block_id = str(
                getattr(candidate, "block_id", "") or getattr(candidate, "candidate_id", "")
            )
            if not block_texts or block_id not in block_texts:
                raise RerankValidationError("候选缺少正文，不能构造证据视图")
            chunk = _CandidateChunk(
                block_id=block_id,
                text=block_texts[block_id],
                chunk_type=str(getattr(candidate, "chunk_type", "") or "text"),
                source_file=str(getattr(candidate, "source_file", "") or ""),
                heading_path=_candidate_heading(candidate, block_id, heading_lookup),
                similarity=_optional_number(getattr(candidate, "similarity", None)),
                match_type=str(getattr(candidate, "match_type", "") or ""),
                route_signals=getattr(candidate, "route_signals", None),
            )
        built = build_chunk_views(
            chunk,
            scope,
            include_structure_views=include_structure_views,
            min_split_chars=min_split_chars,
            heading_lookup=heading_lookup,
        )
        if report is not None:
            report.chunks += 1
            report.views += len(built)
            _record_heading_source(report, candidate, chunk, heading_lookup)
        views.extend(built)
    return tuple(views)


def _record_heading_source(
    report: ViewBuildReport,
    candidate: Any,
    chunk: Any,
    lookup: HeadingLookup | None,
) -> None:
    """统计这一个块的标题来自候选自身/构建出的 chunk，还是查表，或根本没有来源。"""

    for source in (getattr(candidate, "heading_path", None), getattr(chunk, "heading_path", None)):
        if isinstance(source, str) and source.strip():
            report.from_candidate += 1
            return
    block_id = str(getattr(chunk, "block_id", "") or getattr(candidate, "block_id", ""))
    if lookup is None:
        # 有 block_id 却没人能查表：隐性口径分叉，必须可发现（不是静默留空）。
        report.heading_lookup_absent += 1
        return
    if block_id and lookup.heading_for(block_id):
        report.from_lookup += 1


def _candidate_heading(
    candidate: Any,
    block_id: str,
    lookup: HeadingLookup | None,
) -> str:
    """候选记录自身的标题，缺失时按 block_id 查表（不计入缺失统计，避免重复计数）。"""

    heading = getattr(candidate, "heading_path", None)
    if isinstance(heading, str) and heading.strip():
        return heading.strip()
    if lookup is None:
        return ""
    return lookup.heading_for(block_id)
