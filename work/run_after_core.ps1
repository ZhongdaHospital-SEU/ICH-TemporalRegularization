param(
    [string]$LogName = "after_core"
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
$logDir = Join-Path $PSScriptRoot "logs"
$env:PYTHONPATH = $root
$transcript = Join-Path $logDir "$LogName.log"

function Write-Step {
    param([string]$Message)
    $line = "[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $Message
    Write-Output $line
}

Start-Transcript -Path $transcript -Force | Out-Null
try {
    $coreManifest = Join-Path $root "results\run_manifest_core.json"
    $baselineManifest = Join-Path $root "results\run_manifest_baselines.json"
    Write-Step "waiting for core and baseline manifests"
    while (-not ((Test-Path -LiteralPath $coreManifest) -and (Test-Path -LiteralPath $baselineManifest))) {
        $driverAlive = @(
            Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
                Where-Object { $_.CommandLine -like "*run_pipeline_bg.ps1*" }
        ).Count
        if ($driverAlive -eq 0) {
            Write-Step "the pipeline driver is gone before both manifests exist"
            throw "main stage did not complete"
        }
        Start-Sleep -Seconds 120
    }
    Write-Step "both manifests present; starting secondary stage"
    & (Join-Path $PSScriptRoot "run_split_jobs.ps1") -Stage secondary
    if ($LASTEXITCODE -ne 0) { throw "secondary stage failed" }
    Write-Step "secondary stage done"

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
Write-Step "post-core pipeline complete"
Stop-Transcript | Out-Null
