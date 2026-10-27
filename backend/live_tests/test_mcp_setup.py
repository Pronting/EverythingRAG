import asyncio
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from uuid import uuid4

import tomlkit
from ruamel.yaml import YAML

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mcp"))

from manage import NAME, ROOT, TOOLS, config_updates, private_write, settings, verify


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / ".tmp_probe" / "mcp-onboarding" / f"test space {uuid4().hex}"
        self.directory.mkdir(parents=True)
        self.token = settings()["EVERYTHING_RAG_MCP_TOKEN"]

    def test_real_configuration_preserved_and_repeated_install_stable(self):
        sources = {
            Path(".codex/config.toml"): Path(os.environ.get(
                "CODEX_HOME", str(Path.home() / ".codex"),
            )) / "config.toml",
            Path(".claude.json"): Path(os.environ["CLAUDE_CONFIG_DIR"]) / ".claude.json"
            if os.environ.get("CLAUDE_CONFIG_DIR") else Path.home() / ".claude.json",
            Path(".dsh/cordis.patch.yml"): Path(os.environ.get(
                "DSH_HOME", str(Path.home() / ".dsh"),
            )) / "cordis.patch.yml",
        }
        original = {}
        for relative, source in sources.items():
            if source.exists():
                destination = self.directory / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                original[relative] = source.read_bytes()
        backups = self.directory / "backups"
        clients = ["codex", "claude", "dsh", "generic"]
        updates = config_updates(clients, self.token, self.directory)
        for path, content in updates:
            private_write(path, content, backups)
        for relative, content in original.items():
            source = sources[relative]
            self.assertTrue(source.read_bytes() == content)
            path = self.directory / relative
            if relative.suffix == ".toml":
                before = tomlkit.parse(content.decode("utf-8-sig")).unwrap()
                after = tomlkit.parse(path.read_text(encoding="utf-8")).unwrap()
                before.get("mcp_servers", {}).pop(NAME, None)
                after.get("mcp_servers", {}).pop(NAME, None)
                self.assertTrue(before == after)
            elif relative.suffix == ".json":
                before = json.loads(content.decode("utf-8-sig"))
                after = json.loads(path.read_text(encoding="utf-8"))
                before.get("mcpServers", {}).pop(NAME, None)
                after.get("mcpServers", {}).pop(NAME, None)
                before.setdefault("mcpServers", {})
                self.assertTrue(before == after)
            else:
                yaml = YAML()
                before = yaml.load(content.decode("utf-8-sig")) or []
                after = yaml.load(path.read_text(encoding="utf-8"))
                self.assertEqual(len(after), len(before) + 1)
                self.assertTrue(all(str(a) == str(b) for a, b in zip(before, after, strict=False)))
            if content != path.read_bytes():
                self.assertTrue(any(file.read_bytes() == content for file in backups.iterdir()))
        snapshots = {path: path.read_bytes() for path, _ in updates}
        backup_count = len(list(backups.glob("*")))
        for path, content in config_updates(clients, self.token, self.directory):
            private_write(path, content, backups)
            self.assertTrue(path.read_bytes() == snapshots[path])
        self.assertEqual(len(list(backups.glob("*"))), backup_count)

    def test_native_codex_and_claude_read_configuration(self):
        for path, content in config_updates(["codex", "claude"], self.token, self.directory):
            private_write(path, content, self.directory / "backups")
        for name, environment in (
            ("codex", {"CODEX_HOME": str(self.directory / ".codex")}),
            ("claude", {"CLAUDE_CONFIG_DIR": str(self.directory)}),
        ):
            executable = shutil.which(name)
            self.assertIsNotNone(executable, f"需要安装 {name} 执行客户端检查")
            result = subprocess.run(
                [executable, "mcp", "get", NAME], env=os.environ | environment,
                capture_output=True, timeout=90, check=False,
            )
            self.assertEqual(result.returncode, 0, f"{name} 未能读取生成的配置")
            self.assertIn(NAME.encode(), result.stdout)
            if name == "claude":
                self.assertIn(b"Connected", result.stdout)

    def test_live_mcp_connection(self):
        result = asyncio.run(verify(self.token))
        self.assertEqual(result["tools"], len(TOOLS))
        self.assertGreater(result["file_count"], 0)

    def test_native_dsh_reads_patch(self):
        for path, content in config_updates(["dsh"], self.token, self.directory):
            private_write(path, content, self.directory / "backups")
        executable = shutil.which("dsh")
        self.assertIsNotNone(executable, "需要安装 DSH 执行配置检查")
        result = subprocess.run(
            [executable, "web", "--patch", str(self.directory / ".dsh" / "cordis.patch.yml"),
             "--dump-config"], capture_output=True, timeout=90, check=False,
        )
        self.assertEqual(result.returncode, 0, "DSH 未能读取生成的配置")
        self.assertIn(NAME.encode(), result.stdout)


if __name__ == "__main__":
    unittest.main()
