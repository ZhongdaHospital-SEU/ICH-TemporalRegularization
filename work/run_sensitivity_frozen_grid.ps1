$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = (& (Get-Command py).Source -3 -c "import sys; print(sys.executable)").Trim()
$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$env:ICH_TORCH_THREADS = "1"
$env:ICH_XGB_THREADS = "1"
$env:PYTHONUNBUFFERED = "1"

$manifestPath = Join-Path $root "results\run_manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw "Canonical validated primary manifest is required: $manifestPath"
}

$planJson = & $python -m work.run_sensitivity_frozen --list-supported
if ($LASTEXITCODE -ne 0) {
    throw "Sensitivity support preflight failed with exit code $LASTEXITCODE"
}
$plan = $planJson | ConvertFrom-Json

$jobs = foreach ($dataset in @("mimic4", "mimic3", "eicu")) {
    foreach ($entry in $plan.datasets.$dataset) {
        if (-not $entry.status.model_run_allowed) {
            continue
        }
        [pscustomobject]@{
            Dataset = $dataset
            Scenario = $entry.scenario.scenario_id
        }
    }
}

$maxParallel = 14
$running = @()
$failed = 0
foreach ($job in $jobs) {
    while ($running.Count -ge $maxParallel) {
        Start-Sleep -Seconds 2
        $finished = @($running | Where-Object { $_.Process.HasExited })
        foreach ($entry in $finished) {
            $entry.Process.Refresh()
            $entry.Process.WaitForExit()
            $exitCode = $entry.Process.ExitCode
            Write-Output "$($entry.Name) exit=$exitCode"
            if ($exitCode -ne 0) {
                $failed += 1
            }
        }
        $running = @($running | Where-Object { -not $_.Process.HasExited })
    }

    $tag = "$($job.Dataset)_$($job.Scenario)"
    $arguments = @(
        "-m", "work.run_sensitivity_frozen",
        "--dataset", $job.Dataset,
        "--scenario", $job.Scenario,
        "--epochs", "25",
        "--folds", "5",
        "--cv-repeats", "5",
        "--replicates", "2000",
        "--worker-threads", "1"
    )
    $process = Start-Process -FilePath $python -ArgumentList $arguments `
        -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $logDir "sensitivity_frozen_$tag.out.log") `
        -RedirectStandardError (Join-Path $logDir "sensitivity_frozen_$tag.err.log")
    $running += [pscustomobject]@{
        Name = $tag
        Process = $process
    }
    Write-Output "started $tag pid=$($process.Id)"
}

foreach ($entry in $running) {
    $entry.Process.WaitForExit()
    $entry.Process.Refresh()
    $exitCode = $entry.Process.ExitCode
    Write-Output "$($entry.Name) exit=$exitCode"
    if ($exitCode -ne 0) {
        $failed += 1
    }
}
if ($failed -gt 0) {
    throw "$failed frozen-lambda sensitivity process(es) failed"
}

foreach ($dataset in @("mimic4", "mimic3", "eicu")) {
    $recordDir = Join-Path $root "results\sensitivity\$dataset\records"
    $records = foreach ($entry in $plan.datasets.$dataset) {
        $scenario = $entry.scenario.scenario_id
        $path = Join-Path $recordDir "$scenario.json"
        if (Test-Path -LiteralPath $path) {
            $record = Get-Content -Raw -LiteralPath $path | ConvertFrom-Json
            $result = $record.result
            $result | Add-Member -NotePropertyName "fitting_protocol" `
                -NotePropertyValue $record.fitting_protocol
            $result | Add-Member -NotePropertyName "run_signature" `
                -NotePropertyValue $record.run_signature
            $result
        } elseif ($entry.status.model_run_allowed) {
            throw "Missing sensitivity record: $path"
        } else {
            [pscustomobject]@{
                scenario = $entry.scenario
                status = $entry.status
                n = $null
                outcomes = $null
                artifact = $null
                weighting = $null
                preflight_skipped = $true
            }
        }
    }
    $auditPath = Join-Path $root "results\sensitivity\$dataset\scenario_audit.json"
    $records | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $auditPath -Encoding UTF8
}

Write-Output "frozen-lambda sensitivity grid complete"
