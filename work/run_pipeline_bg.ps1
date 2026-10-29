param(
    [string]$LogName = "pipeline"
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$transcript = Join-Path $logDir "$LogName.log"
$env:PYTHONPATH = $root

function Write-Step {
    param([string]$Message)
    $line = "[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $Message
    Write-Output $line
}

Start-Transcript -Path $transcript -Force | Out-Null
try {
    Write-Step "grid start"
    & (Join-Path $PSScriptRoot "run_lambda_grid.ps1")
    if ($LASTEXITCODE -ne 0) { throw "lambda grid failed" }
    Write-Step "grid done"

    Write-Step "lambda selection start"
    & $python (Join-Path $PSScriptRoot "select_lambda_grid.py")
    if ($LASTEXITCODE -ne 0) { throw "lambda selection failed" }
    Write-Step "lambda selection done"

    Write-Step "core 5x5 start"
    & (Join-Path $PSScriptRoot "run_split_jobs.ps1") -Stage main
    if ($LASTEXITCODE -ne 0) { throw "main stage failed" }
    Write-Step "core 5x5 done"
}
catch {
    Write-Step ("FAILED: " + $_.Exception.Message)
    Stop-Transcript | Out-Null
    exit 1
}
Write-Step "pipeline segment complete"
Stop-Transcript | Out-Null
