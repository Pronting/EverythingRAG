@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "VENV_PY=backend\.venv\Scripts\python.exe"
set "BACKEND_STAMP=backend\.venv\.everything-rag-pyproject.stamp"
set "FRONTEND_DEPS_STAMP=frontend\node_modules\.everything-rag-deps.stamp"
set "FRONTEND_BUILD_STAMP=frontend\dist\.everything-rag-build.stamp"

call :ensure_backend
if errorlevel 1 goto setup_failed

call :ensure_frontend
if errorlevel 1 goto setup_failed

echo ================================================
echo   Everything RAG — 个人知识第二大脑
echo   正在启动本地服务，服务就绪后会自动打开浏览器...
echo   关闭本窗口或按 Ctrl+C 可停止服务
echo ================================================
echo.

"%VENV_PY%" scripts\run.py %*
set "APP_EXIT=%ERRORLEVEL%"

echo.
echo 服务已停止。
pause
exit /b %APP_EXIT%

:ensure_backend
if exist "%VENV_PY%" goto check_backend_python

echo [首次运行] 正在创建 Python 虚拟环境...
where py >nul 2>nul
if errorlevel 1 goto try_python
py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
if errorlevel 1 goto try_python
py -3 -m venv "backend\.venv"
if errorlevel 1 goto python_setup_failed
goto check_backend_python

:try_python
where python >nul 2>nul
if errorlevel 1 goto python_missing
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
if errorlevel 1 goto python_missing
python -m venv "backend\.venv"
if errorlevel 1 goto python_setup_failed

:check_backend_python
"%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
if errorlevel 1 goto venv_python_unsupported

set "BACKEND_INSTALL=0"
"%VENV_PY%" scripts\run.py --check-stamp "%BACKEND_STAMP%" --stamp-input "backend\pyproject.toml" >nul 2>nul
if errorlevel 1 set "BACKEND_INSTALL=1"
"%VENV_PY%" -c "import aiosqlite, charset_normalizer, chromadb, fastapi, httpx, markdown_it, multipart, openai, PIL, pydantic, pydantic_settings, uvicorn, xxhash" >nul 2>nul
if errorlevel 1 set "BACKEND_INSTALL=1"
"%VENV_PY%" -m pip check >nul 2>nul
if errorlevel 1 set "BACKEND_INSTALL=1"
if "%BACKEND_INSTALL%"=="0" exit /b 0

echo [准备环境] 后端依赖有变化，正在同步安装，这可能需要几分钟...
"%VENV_PY%" -m pip install -e "backend[dev]"
if errorlevel 1 goto backend_install_failed
"%VENV_PY%" -c "import aiosqlite, charset_normalizer, chromadb, fastapi, httpx, markdown_it, multipart, openai, PIL, pydantic, pydantic_settings, uvicorn, xxhash" >nul 2>nul
if errorlevel 1 goto backend_install_failed
"%VENV_PY%" -m pip check >nul 2>nul
if errorlevel 1 goto backend_install_failed
"%VENV_PY%" scripts\run.py --write-stamp "%BACKEND_STAMP%" --stamp-input "backend\pyproject.toml" >nul
if errorlevel 1 goto stamp_write_failed
exit /b 0

:ensure_frontend
if not exist "frontend\dist\index.html" goto frontend_needs_build
"%VENV_PY%" scripts\run.py --check-stamp "%FRONTEND_BUILD_STAMP%" ^
  --stamp-input "frontend\src" ^
  --stamp-input "frontend\public" ^
  --stamp-input "frontend\index.html" ^
  --stamp-input "frontend\package.json" ^
  --stamp-input "frontend\package-lock.json" ^
  --stamp-input "frontend\tsconfig.json" ^
  --stamp-input "frontend\vite.config.ts" ^
  --stamp-input "frontend\.env" ^
  --stamp-input "frontend\.env.local" ^
  --stamp-input "frontend\.env.production" ^
  --stamp-input "frontend\.env.production.local" >nul 2>nul
if not errorlevel 1 exit /b 0

:frontend_needs_build
where node >nul 2>nul
if errorlevel 1 goto node_missing
node -e "const major=Number(process.versions.node.split('.')[0]); process.exit(major>=18?0:1)" >nul 2>nul
if errorlevel 1 goto node_version_unsupported
where npm >nul 2>nul
if errorlevel 1 goto npm_missing

set "FRONTEND_INSTALL=0"
if not exist "frontend\node_modules\.package-lock.json" set "FRONTEND_INSTALL=1"
"%VENV_PY%" scripts\run.py --check-stamp "%FRONTEND_DEPS_STAMP%" ^
  --stamp-input "frontend\package.json" ^
  --stamp-input "frontend\package-lock.json" >nul 2>nul
if errorlevel 1 set "FRONTEND_INSTALL=1"

if "%FRONTEND_INSTALL%"=="0" goto build_frontend
echo [准备环境] 前端依赖有变化，正在同步安装...
call npm --prefix frontend ci
if errorlevel 1 goto frontend_install_failed
"%VENV_PY%" scripts\run.py --write-stamp "%FRONTEND_DEPS_STAMP%" ^
  --stamp-input "frontend\package.json" ^
  --stamp-input "frontend\package-lock.json" >nul
if errorlevel 1 goto stamp_write_failed

:build_frontend
echo [准备环境] 前端源码有变化，正在构建...
call npm --prefix frontend run build
if errorlevel 1 goto frontend_build_failed
if not exist "frontend\dist\index.html" goto frontend_build_failed
"%VENV_PY%" scripts\run.py --write-stamp "%FRONTEND_BUILD_STAMP%" ^
  --stamp-input "frontend\src" ^
  --stamp-input "frontend\public" ^
  --stamp-input "frontend\index.html" ^
  --stamp-input "frontend\package.json" ^
  --stamp-input "frontend\package-lock.json" ^
  --stamp-input "frontend\tsconfig.json" ^
  --stamp-input "frontend\vite.config.ts" ^
  --stamp-input "frontend\.env" ^
  --stamp-input "frontend\.env.local" ^
  --stamp-input "frontend\.env.production" ^
  --stamp-input "frontend\.env.production.local" >nul
if errorlevel 1 goto stamp_write_failed
exit /b 0

:python_missing
echo [错误] 未找到 Python 3.11 或更高版本。
echo 请先安装 Python 3.11+，并勾选“Add Python to PATH”。
exit /b 1

:venv_python_unsupported
echo [错误] backend\.venv 不是可用的 Python 3.11+ 虚拟环境。
echo 请移走该虚拟环境后重新运行，或使用 Python 3.11+ 重新创建。
exit /b 1

:python_setup_failed
echo [错误] Python 虚拟环境创建失败。
exit /b 1

:backend_install_failed
echo [错误] 后端依赖安装或校验失败，请检查上方提示和网络连接。
exit /b 1

:node_missing
echo [错误] 前端需要重新构建，但未找到 Node.js。
echo 请安装 Node.js 18+ 后重新运行本脚本。
exit /b 1

:node_version_unsupported
echo [错误] 当前 Node.js 版本低于 18，无法可靠构建前端。
echo 请升级到 Node.js 18 或更高版本后重试。
exit /b 1

:npm_missing
echo [错误] 已找到 Node.js，但未找到 npm。
echo 请修复 Node.js/npm 安装后重试。
exit /b 1

:frontend_install_failed
echo [错误] 前端依赖安装失败，请检查上方提示和网络连接。
exit /b 1

:frontend_build_failed
echo [错误] 前端构建失败，请检查上方错误信息。
exit /b 1

:stamp_write_failed
echo [错误] 环境已准备，但无法写入启动 stamp；请检查项目目录权限。
exit /b 1

:setup_failed
echo.
echo 项目未能启动。
pause
exit /b 1
