$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = (& (Get-Command py).Source -3 -c "import sys; print(sys.executable)").Trim()
$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$env:ICH_TORCH_THREADS = "1"
$env:ICH_XGB_THREADS = "1"
$env:PYTHONUNBUFFERED = "1"

$config = Get-Content -Raw (Join-Path $root "configs\study.json") |
    ConvertFrom-Json
$configHash = (Get-FileHash -LiteralPath (Join-Path $root "configs\study.json") `
    -Algorithm SHA256).Hash.ToLowerInvariant()

function Test-CurrentLambdaArtifact {
    param(
        [string]$MetricPath,
        [string]$ManifestPath,
        [double]$Monotone,
        [double]$Smooth
    )
    if (-not (Test-Path -LiteralPath $MetricPath) -or
        -not (Test-Path -LiteralPath $ManifestPath)) {
        return $false
    }
    try {
        $manifest = Get-Content -Raw -LiteralPath $ManifestPath |
            ConvertFrom-Json
    } catch {
        return $false
    }
    if (($manifest.models -join ",") -ne "dggru_tr" -or
        [int]$manifest.folds -ne 5 -or
        [int]$manifest.cv_repeats -ne 5 -or
        (($manifest.model_seeds -join ",") -ne "0,1,2,3,4") -or
        [math]::Abs([double]$manifest.lambda_monotone - $Monotone) -gt 1e-12 -or
        [math]::Abs([double]$manifest.lambda_smooth - $Smooth) -gt 1e-12 -or
        [string]$manifest.config_sha256 -ne $configHash) {
        return $false
    }
    foreach ($property in $manifest.data_sha256.PSObject.Properties) {
        $path = Join-Path $root (Join-Path "data\processed" $property.Name)
        if (-not (Test-Path -LiteralPath $path)) {
            return $false
        }
        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne [string]$property.Value) {
            return $false
        }
    }
    foreach ($property in $manifest.source_sha256.PSObject.Properties) {
        $path = Join-Path $root (Join-Path "src\ich_dynamic" $property.Name)
        if (-not (Test-Path -LiteralPath $path)) {
            return $false
        }
        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne [string]$property.Value) {
            return $false
        }
    }
    return $true
}

$settings = foreach ($lambdaMonotone in $config.lambda_grid) {
    [pscustomobject]@{
        Monotone = [double]$lambdaMonotone
        Smooth = [double]$lambdaMonotone * [double]$config.lambda_smooth_ratio
    }
}

$running = @()
foreach ($setting in $settings) {
    $lambdaMonotone = $setting.Monotone.ToString("0.######", [cultureinfo]::InvariantCulture)
    $lambdaSmooth = $setting.Smooth.ToString("0.######", [cultureinfo]::InvariantCulture)
    $monotoneTag = $lambdaMonotone.Replace(".", "p")
    $smoothTag = $lambdaSmooth.Replace(".", "p")
    $tag = "lambda_m${monotoneTag}_s${smoothTag}"
    $metricPath = Join-Path $root "results\metrics_$tag.json"
    $manifestPath = Join-Path $root "results\run_manifest_$tag.json"
    if (Test-CurrentLambdaArtifact `
        -MetricPath $metricPath `
        -ManifestPath $manifestPath `
        -Monotone $setting.Monotone `
        -Smooth $setting.Smooth) {
        Write-Output "reuse $tag"
        continue
    }

    Remove-Item -LiteralPath $metricPath -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $manifestPath -Force -ErrorAction SilentlyContinue
    $arguments = @(
        "-m", "src.ich_dynamic.train_evaluate",
        "--models", "dggru_tr",
        "--epochs", "25",
        "--folds", "5",
        "--cv-repeats", "5",
        "--replicates", "2000",
        "--skip-external",
        "--fixed-lambda",
        "--lambda-monotone", $lambdaMonotone,
        "--lambda-smooth", $lambdaSmooth,
        "--tag", $tag,
        "--workers", "1",
        "--worker-threads", "1",
        "--metric-workers", "1"
    )
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

$failed = 0
foreach ($job in $running) {
    $job.Process.WaitForExit()
    $job.Process.Refresh()
    $exitCode = $job.Process.ExitCode
    $metricPath = Join-Path $root "results\metrics_$($job.Name).json"
    $manifestPath = Join-Path $root "results\run_manifest_$($job.Name).json"
    $hasOutput = (Test-Path -LiteralPath $metricPath) -and
        (Test-Path -LiteralPath $manifestPath)
    $exitLabel = if ($null -eq $exitCode) { "unavailable" } else { $exitCode }
    Write-Output "$($job.Name) exit=$exitLabel output=$hasOutput"
    if (($null -ne $exitCode -and $exitCode -ne 0) -or -not $hasOutput) {
        $failed += 1
    }
}
if ($failed -gt 0) {
    throw "$failed lambda-grid process(es) failed"
}
Write-Output "lambda grid complete"
