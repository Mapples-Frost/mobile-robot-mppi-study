param(
    [double]$GoalX = 3.0,
    [double]$GoalY = 0.0,
    [double]$DurationS = 30.0,
    [double]$MaxVMps = 0.50,
    [double]$MaxReverseVMps = 0.30,
    [double]$MaxOmegaRadps = 0.60,
    [switch]$FullProposed,
    [switch]$TraditionalMPPI,
    [switch]$EnableActorGuidance,
    [switch]$EnableHSSReliability,
    [switch]$EnableResidualLearning,
    [switch]$EnableChangeAwarePrediction,
    [switch]$EnableProbabilisticRisk,
    [switch]$EnableAR1Sampling,
    [switch]$EnableForwardPassage,
    [switch]$ForwardPassageV3,
    [switch]$DisableResidualLearning,
    [switch]$UntilGoal,
    [switch]$Shadow
)

$ErrorActionPreference = 'Stop'
if ($MaxVMps -le 0.0 -or $MaxVMps -gt 0.50) {
    throw 'MaxVMps must be in (0, 0.50].'
}
if ($MaxReverseVMps -le 0.0 -or $MaxReverseVMps -gt 0.30) {
    throw 'MaxReverseVMps must be in (0, 0.30].'
}
if ($MaxOmegaRadps -le 0.0 -or $MaxOmegaRadps -gt 0.60) {
    throw 'MaxOmegaRadps must be in (0, 0.60].'
}
if ($UntilGoal -and $Shadow) {
    throw 'UntilGoal requires an armed run and cannot be combined with Shadow.'
}
if ($FullProposed -and $TraditionalMPPI) {
    throw 'FullProposed and TraditionalMPPI are exclusive.'
}
$explicitEnhancement = (
    $EnableActorGuidance -or
    $EnableHSSReliability -or
    $EnableResidualLearning -or
    $EnableChangeAwarePrediction -or
    $EnableProbabilisticRisk -or
    $EnableAR1Sampling -or
    $EnableForwardPassage -or
    $ForwardPassageV3
)
if ($TraditionalMPPI -and $explicitEnhancement) {
    throw 'TraditionalMPPI cannot be combined with enhancement switches.'
}
if ($EnableResidualLearning -and $DisableResidualLearning) {
    throw 'Residual learning cannot be both enabled and disabled.'
}
$actorEffective = $FullProposed -or $EnableActorGuidance
$hssEffective = $FullProposed -or $EnableHSSReliability
$predictionEffective = $FullProposed -or $EnableChangeAwarePrediction
$riskEffective = $FullProposed -or $EnableProbabilisticRisk
$passageEffective = $FullProposed -or $EnableForwardPassage -or $ForwardPassageV3
if ($hssEffective -and -not $actorEffective) {
    throw 'HSS reliability requires Actor guidance.'
}
if ($riskEffective -and -not $predictionEffective) {
    throw 'Probabilistic risk requires change-aware obstacle prediction.'
}
if ($passageEffective -and -not $riskEffective) {
    throw 'Forward Passage requires probabilistic risk.'
}
$runDurationS = if ($UntilGoal) { 180.0 } else { $DurationS }
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $projectRoot '.venv-cuda\Scripts\python.exe'
$key = Join-Path $projectRoot 'codex_tmp\pi5_deployment_20260803\pi5_codex_ed25519'
$weightRoot = Join-Path $projectRoot 'codex_tmp\pi5_release_staging_v2'
$lidarConfig = Join-Path $projectRoot 'deploy\raspberry_pi5_scout\livox_adapter_pi5_calibrated.yaml'
$piHost = '10.141.194.219'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$token = 'rlmppi-' + [guid]::NewGuid().ToString('N')
$remoteOut = "/home/pi/rlmppi_pi5/experiments/remote_cuda_${stamp}"
$localOut = Join-Path $projectRoot "codex_tmp\real_robot_cuda_runs\$stamp"
$allowArm = if ($Shadow) { 0 } else { 1 }
$gatewayDuration = [math]::Ceiling($runDurationS + 15.0)

foreach ($required in @($python, $key, $weightRoot, $lidarConfig)) {
    if (-not (Test-Path $required)) {
        throw "Required deployment path is missing: $required"
    }
}
New-Item -ItemType Directory -Path (Split-Path $localOut) -Force | Out-Null

$sshCommon = @(
    '-i', $key,
    '-o', 'HostKeyAlias=raspberrypi.local',
    '-o', 'BatchMode=yes',
    '-o', 'ConnectTimeout=5',
    '-o', 'StrictHostKeyChecking=yes'
)
$active = & ssh @sshCommon "pi@$piHost" "pgrep -af '[r]un_remote_pi_gateway|[r]lmppi_livox_udp_bridge' || true"
if ($active) {
    throw "Another real-robot gateway is active. Refusing concurrent control: $active"
}

$remoteCommand = "test ! -e '$remoteOut' && env " +
    "TOKEN='$token' OUT='$remoteOut' DURATION_S='$gatewayDuration' " +
    "ALLOW_ARM='$allowArm' MAX_V_MPS='$MaxVMps' " +
    "MAX_REVERSE_V_MPS='$MaxReverseVMps' " +
    "MAX_OMEGA_RADPS='$MaxOmegaRadps' " +
    "SCAN_STRIDE='6' " +
    "/home/pi/rlmppi_pi5/current/deploy/raspberry_pi5_scout/run_remote_pi_gateway.sh"
$gatewaySsh = Start-Process -FilePath ssh.exe -ArgumentList ($sshCommon + @("pi@$piHost", $remoteCommand)) -WindowStyle Hidden -PassThru

$runnerArgs = @(
    '-m', 'deploy.raspberry_pi5_scout.run_remote_cuda_full',
    '--pi-host', $piHost,
    '--token', $token,
    '--weight-root', $weightRoot,
    '--lidar-config', $lidarConfig,
    '--output', $localOut,
    '--duration-s', $runDurationS,
    '--accumulation-s', 0.04,
    '--goal-x', $GoalX,
    '--goal-y', $GoalY,
    '--max-v-mps', $MaxVMps,
    '--max-reverse-v-mps', $MaxReverseVMps,
    '--max-omega-radps', $MaxOmegaRadps,
    '--human-leg-mode',
    '--goal-stop-radius-m', 0.25,
    '--warmup-cycles', 3
)
if (-not $Shadow) {
    $runnerArgs += '--publish'
}
if ($FullProposed) {
    $runnerArgs += '--full-proposed'
}
if ($TraditionalMPPI) {
    $runnerArgs += '--traditional-mppi'
}
if ($EnableActorGuidance) {
    $runnerArgs += '--enable-actor-guidance'
}
if ($EnableHSSReliability) {
    $runnerArgs += '--enable-hss-reliability'
}
if ($EnableResidualLearning) {
    $runnerArgs += '--enable-residual-learning'
}
if ($EnableChangeAwarePrediction) {
    $runnerArgs += '--enable-change-aware-prediction'
}
if ($EnableProbabilisticRisk) {
    $runnerArgs += '--enable-probabilistic-risk'
}
if ($EnableAR1Sampling) {
    $runnerArgs += '--enable-ar1-sampling'
}
if ($EnableForwardPassage -or $ForwardPassageV3) {
    $runnerArgs += '--enable-forward-passage'
}
if ($DisableResidualLearning) {
    $runnerArgs += '--disable-residual-learning'
}
if ($UntilGoal) {
    $runnerArgs += '--until-goal'
}

$requestedProfile = if ($FullProposed) {
    if ($DisableResidualLearning) {
        'Full Proposed (residual disabled only)'
    } else {
        'Full Proposed'
    }
} elseif ($TraditionalMPPI -or -not $explicitEnhancement) {
    'Traditional MPPI'
} else {
    'Custom ablation'
}
Write-Host "Requested algorithm profile: $requestedProfile"

$runnerExit = 1
$originalLocation = Get-Location
try {
    Set-Location $projectRoot
    Start-Sleep -Seconds 4
    & $python @runnerArgs
    $runnerExit = $LASTEXITCODE
}
finally {
    Set-Location $originalLocation
    if ($null -ne $gatewaySsh -and -not $gatewaySsh.HasExited) {
        # Stop only the one-time gateway started by this invocation.  The
        # remote wrapper's TERM trap then reaps its own Livox bridge.  This
        # also runs after Ctrl+C or a Python failure, preventing the next
        # experiment from being rejected as a concurrent controller.
        $cleanupPattern = "[r]un_remote_pi_gateway.py --token $token"
        & ssh @sshCommon "pi@$piHost" "pkill -TERM -f '$cleanupPattern' || true" 2>$null
        $gatewaySsh.WaitForExit(5000) | Out-Null
        if (-not $gatewaySsh.HasExited) {
            Stop-Process -Id $gatewaySsh.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
if ($runnerExit -ne 0) {
    throw "CUDA Full Proposed runner failed with exit code $runnerExit"
}

Write-Host "MPPI algorithm-profile run completed."
Write-Host "Local evidence:  $localOut"
Write-Host "Pi evidence:     $remoteOut"
