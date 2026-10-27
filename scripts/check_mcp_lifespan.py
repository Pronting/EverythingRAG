import argparse
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enabled", action="store_true")
    args = parser.parse_args()
    directory = ROOT / ".tmp_probe" / "mcp-lifespan" / uuid4().hex
    directory.mkdir(parents=True)
    token = uuid4().hex + uuid4().hex
    os.environ["EVERYTHING_RAG_DATA_DIR"] = str(directory)
    os.environ["EVERYTHING_RAG_MCP_ENABLED"] = str(args.enabled).lower()
    os.environ["EVERYTHING_RAG_MCP_TOKEN"] = token
    os.environ["EVERYTHING_RAG_EMBED_BASE_URL"] = ""
    os.environ["EVERYTHING_RAG_EMBED_MODEL"] = ""
    os.environ["EVERYTHING_RAG_EMBED_API_KEY"] = ""

    from app.main import app, mcp_mount

    previous = None
    for _ in range(2):
        with TestClient(app, base_url="http://127.0.0.1:9999") as client:
            if not args.enabled:
                for path in ("/mcp", "/mcp/"):
                    for method in ("GET", "POST", "DELETE"):
                        assert client.request(method, path).status_code == 404
                continue
            assert mcp_mount.runtime is not None and mcp_mount.runtime is not previous
            previous = mcp_mount.runtime
            for path in ("/mcp", "/mcp/"):
                assert client.post(path).status_code == 401
            headers = {"Authorization": f"Bearer {token}",
                       "Accept": "application/json, text/event-stream"}
            response = client.post("/mcp/", headers=headers, json={
                "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                    "protocolVersion": "2025-11-25", "capabilities": {},
                    "clientInfo": {"name": "everything-rag-acceptance", "version": "1"},
                },
            })
            assert response.status_code == 200
            assert response.json()["result"]["serverInfo"]["name"] == "everything_rag"
            response = client.post("/mcp/", headers=headers, json={
                "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
            })
            assert response.status_code == 200
            assert len(response.json()["result"]["tools"]) == 4
            response = client.post("/mcp/", headers=headers, json={
                "jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
                    "name": "get_knowledge_overview", "arguments": {},
                },
            })
            assert response.status_code == 200
            result = response.json()["result"]
            assert result["isError"] and result["content"][0]["text"] == "EMBEDDING_NOT_CONFIGURED"
        assert mcp_mount.runtime is None
    print(json.dumps({"enabled": args.enabled, "lifespan_cycles": 2, "passed": True}))


if __name__ == "__main__":
    main()
