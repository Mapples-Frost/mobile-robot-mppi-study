param(
    [ValidateSet("Shadow", "Live")]
    [string]$Mode = "Shadow",
    [ValidateSet("B11FullProposed")]
    [string]$Algorithm = "B11FullProposed",
    [ValidateSet("Generic", "HumanLeg", "ComplexScene")]
    [string]$ObstacleProfile = "Generic",
    [double]$GoalX = 1.0,
    [double]$GoalY = 0.0,
    [double]$BoundaryMinX = -0.75,
    [double]$BoundaryMaxX = 3.50,
    [double]$BoundaryMinY = -1.50,
    [double]$BoundaryMaxY = 1.50,
    [double]$MaxLinear = 0.06,
    [double]$MaxReverse = 0.10,
    [double]$MaxAngular = 0.25,
    [double]$DurationSeconds = 60.0,
    [double]$PeriodSeconds = 0.35,
    [int]$WarmupCycles = 3,
    [switch]$AutoReturnHome,
    [int]$Seed = 20260731,
    [string]$StaticBackgroundMap = "",
    [string]$ConfirmMotion = "",
    [string]$RobotHost = "192.168.31.200",
    [string]$RobotUser = "eaibot",
    [string]$SshKey = "C:\Users\lenovo\AppData\Local\Temp\codex_eaibot_inspect_8e04e093b32f43d8910d73a0025e7c5d\id_ed25519"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $projectRoot ".venv-cuda\Scripts\python.exe"
$runner = Join-Path $PSScriptRoot "real_robot_operator.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "CUDA Python not found: $python"
}
if (-not (Test-Path -LiteralPath $SshKey)) {
    throw "SSH key not found: $SshKey"
}
if ($Mode -eq "Live" -and $ConfirmMotion -ne "I_HAVE_PHYSICAL_ESTOP") {
    throw "Live mode requires -ConfirmMotion I_HAVE_PHYSICAL_ESTOP"
}
if ($MaxLinear -le 0.0 -or $MaxLinear -gt 0.70) {
    throw "MaxLinear must be in (0, 0.70]"
}
if ($MaxReverse -lt 0.0 -or $MaxReverse -gt 0.70) {
    throw "MaxReverse must be in [0, 0.70]"
}
if ($MaxAngular -le 0.0 -or $MaxAngular -gt 3.00) {
    throw "MaxAngular must be in (0, 3.00]"
}
if ($BoundaryMinX -ge $BoundaryMaxX -or $BoundaryMinY -ge $BoundaryMaxY) {
    throw "Invalid boundary"
}

$competing = Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and
    $_.Name -match '^python(?:w)?\.exe$' -and
    $_.CommandLine -match "real_robot_operator.py|real_robot_shadow.py"
}
if ($competing) {
    $ids = ($competing | Select-Object -ExpandProperty ProcessId) -join ","
    throw "Another real-robot planner is active: $ids"
}

$sshBase = @(
    "-i", $SshKey,
    "-o", "BatchMode=yes",
    "-o", "IdentitiesOnly=yes",
    "-o", "ConnectTimeout=6",
    "$RobotUser@$RobotHost"
)

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runRoot = Join-Path $PSScriptRoot "runs"
$output = Join-Path $runRoot ("{0}_{1}" -f $Mode.ToLowerInvariant(), $stamp)
$sessionStarted = $false

$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:CUDA_VISIBLE_DEVICES = "0"

if ($Mode -eq "Live") {
    $commandWatchdogS = 0.60
    # The robot's yocs smoother is configured for 0.6 m/s^2 deceleration.
    # Cover one command-watchdog interval, ideal braking distance, and a fixed
    # 0.30 m body/model margin before reaching the hard-stop boundary.
    $frontStopM = [Math]::Max(
        0.45,
        0.30 + ($MaxLinear * $commandWatchdogS) + (($MaxLinear * $MaxLinear) / (2.0 * 0.60))
    )
    $frontSlowM = [Math]::Max(0.80, $frontStopM + 0.50)
    $sessionTimeout = [Math]::Min(330.0, $DurationSeconds + 30.0)
    $remoteArgs = @(
        "--allow-reverse",
        "--max-linear-mps", $MaxLinear.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--max-angular-radps", $MaxAngular.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--max-reverse-mps", $MaxReverse.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--command-watchdog-s", $commandWatchdogS.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--sensor-watchdog-s", "0.40",
        "--front-stop-m", $frontStopM.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--front-slow-m", $frontSlowM.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--session-timeout-s", $sessionTimeout.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--boundary-min-x", $BoundaryMinX.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--boundary-max-x", $BoundaryMaxX.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--boundary-min-y", $BoundaryMinY.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--boundary-max-y", $BoundaryMaxY.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--boundary-margin-m", "0.25",
        "--boundary-prediction-s", "0.75"
    )
    $remoteCommand = (
        "/home/eaibot/deployments/rl_mppi_reverse_full_stack_current/start_control_session.sh " +
        ($remoteArgs -join " ")
    )
    & ssh.exe @sshBase $remoteCommand
    if ($LASTEXITCODE -ne 0) {
        throw "Robot control session failed to start"
    }
    $sessionStarted = $true
}

Push-Location $projectRoot
try {
    $runnerArgs = @(
        $runner,
        "--mode", $Mode,
        "--algorithm", $Algorithm,
        "--obstacle-profile", $ObstacleProfile,
        "--seed", $Seed,
        "--goal-x", $GoalX,
        "--goal-y", $GoalY,
        "--boundary-min-x", $BoundaryMinX,
        "--boundary-max-x", $BoundaryMaxX,
        "--boundary-min-y", $BoundaryMinY,
        "--boundary-max-y", $BoundaryMaxY,
        "--max-linear-mps", $MaxLinear,
        "--max-reverse-mps", $MaxReverse,
        "--max-angular-radps", $MaxAngular,
        "--duration-s", $DurationSeconds,
        "--period-s", $PeriodSeconds,
        "--warmup-cycles", $WarmupCycles,
        "--output", $output
    )
    if ($StaticBackgroundMap) {
        if (-not (Test-Path -LiteralPath $StaticBackgroundMap)) {
            throw "Static background map not found: $StaticBackgroundMap"
        }
        $runnerArgs += @(
            "--static-background-map", (Resolve-Path -LiteralPath $StaticBackgroundMap).Path
        )
    }
    if ($AutoReturnHome) {
        $runnerArgs += "--auto-return-home"
    }
    if ($Mode -eq "Live") {
        $runnerArgs += @(
            "--confirm-motion", "I_HAVE_PHYSICAL_ESTOP"
        )
    }
    & $python @runnerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Full-stack runner failed with exit code $LASTEXITCODE"
    }
    Write-Output "RUN_OUTPUT=$output"
}
finally {
    Pop-Location
    if ($sessionStarted) {
        & ssh.exe @sshBase `
            "/home/eaibot/deployments/rl_mppi_reverse_full_stack_current/stop_control_session.sh"
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Automatic robot-side stop returned a failure; use Stop-Robot.ps1 immediately."
        }
    }
}
