"""Deterministic conversational-query normalisation tests."""

from __future__ import annotations

from app.retrieval.query_transform import (
    build_query_variants,
    extract_identifier_definition_subjects,
    normalize_query,
)


def test_conversational_query_keeps_entities_and_removes_scaffolding() -> None:
    query = (
        "好像有用一些厂商的分表对比，比如说京东、淘宝等对比，和公司的对比，"
        "我想问一下这些对比的详情是什么样的"
    )
    variants = build_query_variants(query)

    assert variants[0].kind == "original"
    assert variants[0].text == query
    assert variants[1].kind == "normalized"
    normalized = variants[1].text
    assert "好像" not in normalized
    assert "有用一些" not in normalized
    assert "我想问一下" not in normalized
    assert "是什么样的" not in normalized
    assert normalized == "厂商分表对比,京东、淘宝,公司对比详情"
    for entity in ("厂商", "分表", "京东", "淘宝", "公司", "对比", "详情"):
        assert entity in normalized


def test_identifiers_survive_normalisation_and_original_is_always_kept() -> None:
    query = "我想了解一下 SPR-2024-0001、RocketMQ 和 Qwen3-Embedding-8B 是什么样的？"
    variants = build_query_variants(query)
    assert variants[0].text == query
    assert len(variants) == 2
    for entity in ("SPR-2024-0001", "RocketMQ", "Qwen3-Embedding-8B"):
        assert entity in variants[1].text


def test_already_concise_query_has_no_duplicate_variant() -> None:
    variants = build_query_variants("购物车分表方案对比")
    assert [(variant.kind, variant.text) for variant in variants] == [
        ("original", "购物车分表方案对比")
    ]


def test_punctuation_only_normalisation_does_not_create_duplicate_route() -> None:
    query = "简单介绍一下warebase， 有没有什么事故出现在warebase身上"
    variants = build_query_variants(query)

    assert [(variant.kind, variant.text) for variant in variants] == [("original", query)]


def test_extracts_only_narrow_identifier_definition_questions() -> None:
    assert extract_identifier_definition_subjects("什么是adb") == ("adb",)
    assert extract_identifier_definition_subjects("Warebase 是什么？") == ("Warebase",)
    assert extract_identifier_definition_subjects("请问一下 AAB 是什么？") == ("AAB",)
    assert extract_identifier_definition_subjects("什么是首次触点归因") == ()
    assert (
        extract_identifier_definition_subjects("简单介绍Warebase，再说说它出现过什么事故")
        == ()
    )
    assert extract_identifier_definition_subjects("APISIX 发版事故的根本原因是什么？") == ()
    assert (
        extract_identifier_definition_subjects("eBay、SHEIN 的未登录购物车方案分别是什么？")
        == ()
    )


def test_empty_query_stays_empty() -> None:
    assert normalize_query("   ") == ""
    assert build_query_variants("   ") == ()


def test_normalize_topic_and_method_scaffolding_without_losing_subject() -> None:
    assert normalize_query("对于风控，团队是怎么优化99分位的？") == "风控,优化99分位"


def test_nested_usage_question_keeps_domain_and_metric():
    query = "应该是风控对于99分位，它的优化是怎么样的，以及99分位是如何在风控服务中存在的，它的意义何在"
    variants = build_query_variants(query)
    assert variants[0].text == query
    assert variants[1].text == "风控 99分位,优化,99分位 风控服务"


def test_scaffolding_removal_applies_beyond_percentiles_without_changing_facts():
    query = "应该是RocketMQ对于重试，它的优化是怎么样的，以及死信队列是如何在支付服务中存在的"
    normalized = normalize_query(query)
    for term in ("RocketMQ", "重试", "优化", "死信队列", "支付服务"):
        assert term in normalized
    assert "样的" not in normalized
    for query in ("不是风控，是支付服务的P95", "火星服务中P99.9的意义何在", "存在性检测失败"):
        assert build_query_variants(query)[0].text == query
    assert "火星" in normalize_query("火星服务中P99.9的意义何在")
