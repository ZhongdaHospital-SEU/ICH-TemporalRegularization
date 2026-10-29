param(
    [string]$LogName = "finalize"
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
$logDir = Join-Path $PSScriptRoot "logs"
$env:PYTHONPATH = $root

function Write-Step {
    param([string]$Message)
    Write-Output ("[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $Message)
}

function Get-TaggedPython {
    param([string]$Tag)
    @(
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object { $_.CommandLine -match "--tag\s+$([regex]::Escape($Tag))\b" }
    ).Count
}

Start-Transcript -Path (Join-Path $logDir "$LogName.log") -Force | Out-Null
try {
    $coreManifest = Join-Path $root "results\run_manifest_core.json"
    $baselineManifest = Join-Path $root "results\run_manifest_baselines.json"

    Write-Step "waiting for core"
    while (-not (Test-Path -LiteralPath $coreManifest)) {
        if ((Get-TaggedPython -Tag "core") -eq 0) {
            throw "core process disappeared without writing its manifest"
        }
        Start-Sleep -Seconds 60
    }
    Write-Step "core manifest present"

    Write-Step "waiting for baselines"
    while (-not (Test-Path -LiteralPath $baselineManifest)) {
        if ((Get-TaggedPython -Tag "baselines") -eq 0) {
            throw "baselines process disappeared without writing its manifest"
        }
        Start-Sleep -Seconds 60
    }
    Write-Step "baselines manifest present"

    & (Join-Path $PSScriptRoot "run_split_jobs.ps1") -Stage secondary
    if ($LASTEXITCODE -ne 0) { throw "secondary stage failed" }
    Write-Step "secondary done"

    & $python (Join-Path $PSScriptRoot "assemble_split_run.py")
    if ($LASTEXITCODE -ne 0) { throw "assembly failed" }
    Write-Step "assembly done"

    & $python (Join-Path $PSScriptRoot "validate_formal_run.py")
    if ($LASTEXITCODE -ne 0) { throw "validation failed" }
    Write-Step "validation done"
}
catch {
    Write-Step ("FAILED: " + $_.Exception.Message)
    Stop-Transcript | Out-Null
    exit 1
}
Write-Step "pipeline complete"
Stop-Transcript | Out-Null
