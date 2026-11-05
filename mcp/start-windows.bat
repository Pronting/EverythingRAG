@echo off
setlocal
title Everything RAG MCP Server
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" -Action start
set "mcp_exit_code=%ERRORLEVEL%"
if not "%mcp_exit_code%"=="0" (
    echo MCP startup failed. See the error above and mcp/INSTALL.md.
    pause
    exit /b %mcp_exit_code%
)
echo.
echo Status: READY - MCP connection verified; server running in the background.
echo Address: http://127.0.0.1:9999/mcp/
echo Tool and document counts are shown in the check result above.
echo.
echo Closing this window will keep the MCP server running.
echo Press any key to close this window.
pause >nul
exit /b 0
