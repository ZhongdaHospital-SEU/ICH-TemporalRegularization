param(
    [ValidateSet("grid", "core", "baselines", "main", "secondary", "all")]
    [string]$Stage = "all"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$env:PYTHONPATH = $root
$env:ICH_TORCH_THREADS = "2"
$env:ICH_XGB_THREADS = "2"

function Start-TrainingJob {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    $stdout = Join-Path $logDir "$Name.out.log"
    $stderr = Join-Path $logDir "$Name.err.log"
    Remove-Item -LiteralPath $stdout -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stderr -Force -ErrorAction SilentlyContinue
    # Start-Process joins the array with spaces without preserving element
    # boundaries. Quote any argument containing whitespace so paths survive.
    $quotedArguments = @(
        foreach ($argument in $Arguments) {
            if ($argument -match '\s') {
                '"' + $argument.Replace('"', '\"') + '"'
            }
            else {
                $argument
            }
        }
    )
    $process = Start-Process -FilePath $python -ArgumentList $quotedArguments `
        -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    # Host output only: emitting this through the pipeline would make the
    # function return an array and break WaitForExit on the caller side.
    Write-Host "started $Name pid=$($process.Id)"
    return ,$process
}

function Wait-TrainingJob {
    param(
        [string]$Name,
        $Process
    )
    $Process.WaitForExit()
    $Process.Refresh()
    $exitCode = $Process.ExitCode
    Write-Output "$Name exit=$exitCode"
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode"
    }
}

if ($Stage -in @("grid", "all")) {
    & (Join-Path $PSScriptRoot "run_lambda_grid.ps1")
    & $python (Join-Path $PSScriptRoot "select_lambda_grid.py")
    if ($LASTEXITCODE -ne 0) {
        throw "Lambda-grid selection failed"
    }
}

$selectionPath = Join-Path $root "results\lambda_grid_selection.json"
if (-not (Test-Path -LiteralPath $selectionPath)) {
    throw "Missing $selectionPath; run the grid stage first."
}
$selection = Get-Content -Raw $selectionPath | ConvertFrom-Json
$lambdaMonotone = ([double]$selection.selected.lambda_monotone).ToString(
    "0.######",
    [cultureinfo]::InvariantCulture
)
$lambdaSmooth = ([double]$selection.selected.lambda_smooth).ToString(
    "0.######",
    [cultureinfo]::InvariantCulture
)

if ($Stage -in @("core", "main", "all")) {
    $coreArgs = @(
        "-u", "-m", "src.ich_dynamic.train_evaluate",
        "--models", "xgb_summary,gru,dggru,dggru_tr",
        "--epochs", "25",
        "--folds", "5",
        "--cv-repeats", "5",
        "--replicates", "2000",
        "--fixed-lambda",
        "--lambda-monotone", $lambdaMonotone,
        "--lambda-smooth", $lambdaSmooth,
        "--tag", "core",
        "--workers", "5",
        "--worker-threads", "3",
        "--metric-workers", "5"
    )
    # The core four-model set is the primary estimate. Run it first with the
    # full CPU allocation, then run the additional baselines.
    $core = Start-TrainingJob -Name "core_5x5" -Arguments $coreArgs
    Wait-TrainingJob -Name "core_5x5" -Process $core
}

if ($Stage -in @("baselines", "main", "all")) {
    # Kept as its own stage so that baselines can start after core has already
    # been launched by another driver.
    $baselineArgs = @(
        "-u", "-m", "src.ich_dynamic.train_evaluate",
        "--models", "xgb_static,gcs_slope,grud,transformer",
        "--epochs", "25",
        "--folds", "5",
        "--cv-repeats", "5",
        "--replicates", "2000",
        "--tag", "baselines",
        "--workers", "5",
        "--worker-threads", "3",
        "--metric-workers", "5"
    )
    $baselines = Start-TrainingJob -Name "baselines_5x5" -Arguments $baselineArgs
    Wait-TrainingJob -Name "baselines_5x5" -Process $baselines
}

if ($Stage -in @("secondary", "all")) {
    $coreManifest = Join-Path $root "results\run_manifest_core.json"
    if (-not (Test-Path -LiteralPath $coreManifest)) {
        throw "Missing $coreManifest; run the main stage first."
    }
    $stackArgs = @(
        "-u", "-m", "src.ich_dynamic.train_evaluate",
        "--models", "nested_stack",
        "--epochs", "25",
        "--folds", "5",
        "--cv-repeats", "5",
        "--replicates", "2000",
        "--freeze-from", $coreManifest,
        "--tag", "stack",
        "--workers", "3",
        "--worker-threads", "2",
        "--metric-workers", "2"
    )
    $ablationModels = @(
        "dggru_tr_treverse",
        "dggru_tr_tpermute",
        "dggru_laplace",
        "dggru_tr_gcsmask",
        "dggru_tr_noseq",
        "dggru_tr_static",
        "dggru_tr_traj"
    ) -join ","
    $ablationArgs = @(
        "-u", "-m", "src.ich_dynamic.train_evaluate",
        "--models", $ablationModels,
        "--epochs", "25",
        "--folds", "5",
        "--cv-repeats", "1",
        "--replicates", "2000",
        "--freeze-from", $coreManifest,
        "--tag", "ablation",
        "--workers", "3",
        "--worker-threads", "2",
        "--metric-workers", "2"
    )
    $stack = Start-TrainingJob -Name "stack_5x5" -Arguments $stackArgs
    $ablation = Start-TrainingJob -Name "ablation_1x5" -Arguments $ablationArgs
    Wait-TrainingJob -Name "stack_5x5" -Process $stack
    Wait-TrainingJob -Name "ablation_1x5" -Process $ablation
}

Write-Output "stage $Stage complete"
