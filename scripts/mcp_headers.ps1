$ErrorActionPreference = 'Stop'
$mcpToken = [Environment]::GetEnvironmentVariable('EVERYTHING_RAG_MCP_TOKEN', 'User')
if ([string]::IsNullOrWhiteSpace($mcpToken)) {
    $mcpToken = $env:EVERYTHING_RAG_MCP_TOKEN
}
if ([string]::IsNullOrWhiteSpace($mcpToken) -or $mcpToken.Length -lt 32) {
    throw 'MCP_TOKEN_NOT_CONFIGURED'
}
@{ Authorization = "Bearer $mcpToken" } | ConvertTo-Json -Compress
