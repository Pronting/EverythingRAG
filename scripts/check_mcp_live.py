import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx
import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client


async def verify(url: str):
    token = os.environ["EVERYTHING_RAG_MCP_TOKEN"]
    authorization = {"Authorization": f"Bearer {token}"}
    checks = []
    async with httpx.AsyncClient(timeout=60) as http:
        for method in ("GET", "POST", "DELETE"):
            response = await http.request(method, url)
            assert response.status_code == 401, (method, response.status_code)
        for headers in ({"Host": "external.invalid"}, {"Origin": "https://external.invalid"}):
            response = await http.post(url, headers=authorization | headers)
            assert response.status_code == 403
        checks.append("http_auth_host_origin")
    for mode in ("auto", "legacy"):
        async with httpx2.AsyncClient(headers=authorization, timeout=60) as http:
            transport = streamable_http_client(url, http_client=http)
            async with Client(transport, mode=mode, read_timeout_seconds=60) as client:
                discovered = await client.list_tools()
                assert {t.name for t in discovered.tools} == {
                    "get_knowledge_overview", "lookup_catalog", "search_knowledge", "read_document",
                }
                assert all(t.output_schema and t.annotations.read_only_hint for t in discovered.tools)

                async def call(name, arguments):
                    result = await client.call_tool(name, arguments)
                    assert not result.is_error, (name, result.content)
                    data = result.structured_content
                    assert isinstance(data, dict)
                    assert json.loads(result.content[0].text) == data
                    encoded = result.model_dump_json(by_alias=True)
                    assert len(encoded.encode()) <= 256 * 1024
                    assert token not in encoded
                    return data

                overview = await call("get_knowledge_overview", {"limit": 20})
                assert overview["index_status"] == "ready"
                assert overview["file_count"] > 0
                doc = overview["documents"][0]
                catalog = await call("lookup_catalog", {"query": doc["document_name"], "limit": 1})
                assert any(d["document_id"] == doc["document_id"] for d in catalog["documents"])
                first = await call("read_document", {"document_id": doc["document_id"], "limit": 600})
                assert first["text"] and first["content_origin"] == "indexed_chunks"
                if first["next_offset"] is not None:
                    second = await call("read_document", {
                        "document_id": doc["document_id"], "offset": first["next_offset"],
                        "limit": 600, "expected_document_revision": first["document_revision"],
                    })
                    combined = await call("read_document", {
                        "document_id": doc["document_id"], "limit": 1200,
                    })
                    assert first["text"] + second["text"] == combined["text"]
                for name, arguments, code in [
                    ("read_document", {"document_id": doc["document_id"], "offset": 1},
                     "DOCUMENT_REVISION_REQUIRED"),
                    ("read_document", {"document_id": doc["document_id"],
                                       "expected_document_revision": "0" * 64}, "STALE_DOCUMENT"),
                    ("lookup_catalog", {"query": "MCP", "cursor": "invalid"}, "INVALID_CURSOR"),
                    ("search_knowledge", {"query": " "}, "INVALID_ARGUMENTS"),
                    ("search_knowledge", {"query": "MCP", "top_k": 21}, "INVALID_ARGUMENTS"),
                    ("search_knowledge", {"query": "MCP", "document_scope": ["unknown"]},
                     "DOCUMENT_NOT_FOUND"),
                ]:
                    result = await client.call_tool(name, arguments)
                    assert result.is_error and result.content[0].text == code, (code, result.content)
                query = doc["headings"][0] if doc["headings"] else doc["document_name"]
                search = await call("search_knowledge", {
                    "query": query, "document_scope": [doc["document_id"]], "top_k": 3,
                })
                assert search["chunks"], search["status"]
                assert all(c["document_id"] == doc["document_id"] for c in search["chunks"])
                assert all(c["document_revision"] == first["document_revision"]
                           for c in search["chunks"])
                checks.append({"mode": mode, "tools": len(discovered.tools),
                               "files": overview["file_count"], "blocks": overview["chunk_count"],
                               "search_hits": len(search["chunks"]),
                               "rerank_status": search["rerank_status"],
                               "pagination": True, "error_checks": 6})
    return {"checks": checks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:9999/mcp/")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = asyncio.run(verify(arguments.url))
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if arguments.output:
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded)
