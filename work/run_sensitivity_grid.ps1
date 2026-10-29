$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = (& (Get-Command py).Source -3 -c "import sys; print(sys.executable)").Trim()
$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$env:ICH_TORCH_THREADS = "2"
$env:ICH_XGB_THREADS = "2"

$running = @()
foreach ($dataset in @("mimic4", "eicu")) {
    $auditPath = Join-Path $root "results\sensitivity\$dataset\scenario_audit.json"
    Remove-Item -LiteralPath $auditPath -Force -ErrorAction SilentlyContinue
    $arguments = @(
        "-m", "src.ich_dynamic.sensitivity_analysis",
        "--analysis", "all",
        "--dataset", $dataset,
        "--epochs", "25",
        "--folds", "5",
        "--cv-repeats", "5",
        "--replicates", "2000",
        "--workers", "2",
        "--worker-threads", "2"
    )
    $process = Start-Process -FilePath $python -ArgumentList $arguments `
        -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $logDir "sensitivity_$dataset.out.log") `
        -RedirectStandardError (Join-Path $logDir "sensitivity_$dataset.err.log")
    $running += [pscustomobject]@{
        Dataset = $dataset
        Process = $process
    }
    Write-Output "started sensitivity/$dataset pid=$($process.Id)"
}

$failed = 0
foreach ($job in $running) {
    $job.Process.WaitForExit()
    $job.Process.Refresh()
    $exitCode = $job.Process.ExitCode
    $auditPath = Join-Path $root "results\sensitivity\$($job.Dataset)\scenario_audit.json"
    $hasOutput = Test-Path -LiteralPath $auditPath
    $exitLabel = if ($null -eq $exitCode) { "unavailable" } else { $exitCode }
    Write-Output "$($job.Dataset) exit=$exitLabel audit=$hasOutput"
    if (($null -ne $exitCode -and $exitCode -ne 0) -or -not $hasOutput) {
        $failed += 1
    }
}
if ($failed -gt 0) {
    throw "$failed sensitivity process(es) failed"
}
Write-Output "sensitivity grid complete"
