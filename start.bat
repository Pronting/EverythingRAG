@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PYTHON=backend\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo [错误] 未找到虚拟环境: %PYTHON%
    echo.
    echo 请先安装后端依赖（在项目根目录执行）:
    echo   cd backend
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -e ".[dev]"
    echo.
    pause
    exit /b 1
)

echo ================================================
echo   Everything RAG — 个人知识第二大脑
echo   正在启动本地服务，随后会自动打开浏览器...
echo   关闭本窗口或按 Ctrl+C 可停止服务
echo ================================================
echo.

"%PYTHON%" scripts\run.py

echo.
echo 服务已停止。
pause
