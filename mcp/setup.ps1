param(
    [ValidateSet('auto', 'codex', 'claude', 'dsh', 'all', 'generic')]
    [string]$Client = 'auto',
    [ValidateSet('install', 'start', 'check')]
    [string]$Action = 'install'
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot 'backend\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Backend environment missing. Run EverythingRagStart.bat once to install the project.'
}
if (Get-Command uv -ErrorAction SilentlyContinue) {
    & uv pip install --python $python -r (Join-Path $PSScriptRoot 'requirements.txt')
} else {
    & $python -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
}
if ($LASTEXITCODE -ne 0) { throw 'MCP setup dependency installation failed.' }
& $python (Join-Path $PSScriptRoot 'manage.py') $Action --client $Client
exit $LASTEXITCODE
