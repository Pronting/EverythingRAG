"""EverythingRagStart.bat 的更新感知与运行时版本门禁静态契约。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START_BAT = ROOT / "EverythingRagStart.bat"


def test_start_bat_uses_windows_line_endings() -> None:
    raw = START_BAT.read_bytes()
    assert b"\r\n" in raw
    assert b"\n" not in raw.replace(b"\r\n", b"")


def test_start_bat_uses_backend_and_frontend_stamps() -> None:
    script = START_BAT.read_text(encoding="utf-8")
    assert "--check-stamp" in script
    assert "--write-stamp" in script
    assert 'if exist "frontend\\dist\\index.html" exit /b 0' not in script
    assert "frontend\\src" in script
    assert "frontend\\public" in script
    assert "frontend\\package-lock.json" in script
    assert "backend\\pyproject.toml" in script


def test_start_bat_checks_required_runtime_versions() -> None:
    script = START_BAT.read_text(encoding="utf-8")
    assert 'set "PYTHONUTF8=1"' in script
    assert "sys.version_info >= (3, 11)" in script
    assert "major>=18" in script
    assert ":venv_python_unsupported" in script
    assert ":node_version_unsupported" in script


def test_start_bat_verifies_direct_backend_dependencies() -> None:
    script = START_BAT.read_text(encoding="utf-8")
    assert "charset_normalizer" in script
    assert "multipart" in script
    assert "pydantic_settings" in script
    assert "-m pip check" in script
