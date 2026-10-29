$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = (& (Get-Command py).Source -3 -c "import sys; print(sys.executable)").Trim()
$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$env:ICH_TORCH_THREADS = "2"
$env:ICH_XGB_THREADS = "2"
$config = Get-Content -Raw (Join-Path $root "configs\study.json") |
    ConvertFrom-Json
$settings = foreach ($lambdaMonotone in $config.lambda_grid) {
    [pscustomobject]@{
        Monotone = [double]$lambdaMonotone
        Smooth = [double]$lambdaMonotone * [double]$config.lambda_smooth_ratio
    }
}

$failed = 0
# The grid is a screening pass: one CV repeat per weight. Six concurrent
# processes keep about 12 threads busy and leave memory headroom on a
# 16-core / 16 GB host.
$batchSize = 6
for ($start = 0; $start -lt $settings.Count; $start += $batchSize) {
    $batch = $settings[$start..([Math]::Min($start + $batchSize - 1, $settings.Count - 1))]
    $running = @()
    foreach ($setting in $batch) {
        $lambdaMonotone = $setting.Monotone.ToString("0.######", [cultureinfo]::InvariantCulture)
        $lambdaSmooth = $setting.Smooth.ToString("0.######", [cultureinfo]::InvariantCulture)
        $monotoneTag = $lambdaMonotone.Replace(".", "p")
        $smoothTag = $lambdaSmooth.Replace(".", "p")
        $tag = "lambda_m${monotoneTag}_s${smoothTag}"
        $arguments = @(
            "-m", "src.ich_dynamic.train_evaluate",
            "--models", "dggru_tr",
            "--epochs", "25",
            "--folds", "5",
            "--cv-repeats", "1",
            "--replicates", "200",
            "--skip-external",
            "--fixed-lambda",
            "--lambda-monotone", $lambdaMonotone,
            "--lambda-smooth", $lambdaSmooth,
            "--tag", $tag,
            "--workers", "1",
            "--worker-threads", "2",
            "--metric-workers", "1"
        )
        $metricPath = Join-Path $root "results\metrics_$tag.json"
        $manifestPath = Join-Path $root "results\run_manifest_$tag.json"
        Remove-Item -LiteralPath $metricPath -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $manifestPath -Force -ErrorAction SilentlyContinue
        $process = Start-Process -FilePath $python -ArgumentList $arguments `
            -WorkingDirectory $root -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $logDir "$tag.out.log") `
            -RedirectStandardError (Join-Path $logDir "$tag.err.log")
        $running += [pscustomobject]@{
            Name = $tag
            Process = $process
        }
        Write-Output "started $tag pid=$($process.Id)"
    }

    foreach ($job in $running) {
        $job.Process.WaitForExit()
        $job.Process.Refresh()
        $exitCode = $job.Process.ExitCode
        $metricPath = Join-Path $root "results\metrics_$($job.Name).json"
        $hasOutput = Test-Path -LiteralPath $metricPath
        $exitLabel = if ($null -eq $exitCode) { "unavailable" } else { $exitCode }
        Write-Output "$($job.Name) exit=$exitLabel output=$hasOutput"
        if (($null -ne $exitCode -and $exitCode -ne 0) -or -not $hasOutput) {
            $failed += 1
        }
    }
}
if ($failed -gt 0) {
    throw "$failed lambda-grid process(es) failed"
}
Write-Output "lambda grid complete"
