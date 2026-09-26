param(
    [switch]$ValidateOnly
)

# Native wait: no extra Python process exists during measured learning/timing.
$ErrorActionPreference = 'Stop'
$studyRoot = '\\wsl.localhost\Ubuntu-20.04\home\mapples\projects\mobile-robot-mppi-study'
$linuxRoot = '/home/mapples/projects/mobile-robot-mppi-study'
$resultRoot = Join-Path $studyRoot 'research_artifacts/bohn2021_reproduction_2026-09-17/results/latency_tree_2026-09-26'
$scriptRoot = Join-Path $studyRoot 'experiments/bohn2021_reproduction'
$statePath = Join-Path $resultRoot 'native_pipeline_v2_status.json'
$registrationPath = Join-Path $resultRoot 'native_pipeline_v2_registration.json'
$legacyPython = '/home/mapples/.local/share/bohn2021-python37/bin/python'
$modernPython = $linuxRoot + '/.venv/bin/python'

function Read-Json([string]$path) {
    return Get-Content -LiteralPath $path -Raw | ConvertFrom-Json -AsHashtable
}

function Digest([string]$path) {
    return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Save-Json([string]$path, $value) {
    $temporary = $path + '.tmp'
    [IO.File]::WriteAllText($temporary, ($value | ConvertTo-Json -Depth 50) + [Environment]::NewLine,
                           [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $path -Force
}

function Assert-SourceHashes($hashes) {
    foreach ($relative in $hashes.Keys) {
        $actual = Digest (Join-Path $studyRoot $relative)
        if ($actual -ne $hashes[$relative]) {
            throw ('Registered pipeline source changed: ' + $relative)
        }
    }
}

function Matching-TrainingProcess([long]$processNumber) {
    $path = '\\wsl.localhost\Ubuntu-20.04\proc\' + $processNumber + '\cmdline'
    try {
        $commandText = [Text.Encoding]::UTF8.GetString([IO.File]::ReadAllBytes($path))
        return $commandText.Contains('latency_tree_run.py')
    } catch [IO.FileNotFoundException] {
        return $false
    } catch [IO.DirectoryNotFoundException] {
        return $false
    }
}

$programs = @(
    'latency_tree_evaluation_spec.py', 'latency_tree_evaluate.py',
    'latency_tree_baselines.py', 'latency_tree_fixed_train.py',
    'latency_tree_evaluation_audit.py', 'latency_tree_effect.py',
    'latency_tree_effect_review.py', 'latency_tree_evaluation_checks.py',
    'latency_tree_training_delivery.py', 'latency_tree_training_delivery_review.py',
    'latency_tree_native_pipeline.ps1'
)
$sourceHashes = @{}
foreach ($name in $programs) {
    $relative = 'experiments/bohn2021_reproduction/' + $name
    $sourceHashes[$relative] = Digest (Join-Path $studyRoot $relative)
}
if (-not (Test-Path -LiteralPath $registrationPath)) {
    throw 'Native pipeline registration must be frozen and reviewed before launch'
}
$registration = Read-Json $registrationPath
Assert-SourceHashes $registration.source_hashes
if ($registration.source_hashes.Count -ne $sourceHashes.Count) {
    throw 'Unexpected pipeline source set'
}
foreach ($name in $sourceHashes.Keys) {
    if ($registration.source_hashes[$name] -ne $sourceHashes[$name]) { throw ('Source mismatch: ' + $name) }
}
if ($ValidateOnly) {
    [ordered]@{validated=$true;source_count=$sourceHashes.Count;training_touched=$false;python_started=$false} |
        ConvertTo-Json
    exit 0
}
if (Test-Path -LiteralPath $statePath) {
    throw 'Existing native controller state requires inspection; never auto-restart'
}
$state = [ordered]@{
    pid=$PID;active=$true;complete=$false;stage='waiting_for_training_exit'
    started_utc=[DateTime]::UtcNow.ToString('o');training_pid=1694525
    completed_stages=@();registration_sha256=(Digest $registrationPath)
    test_accessed=$false;goal_complete=$false
}
Save-Json $statePath $state

function Execute-Stage([string]$label, [string]$interpreter, [string]$program, [string[]]$arguments) {
    Assert-SourceHashes $registration.source_hashes
    $state.stage = $label
    Save-Json $statePath $state
    $log = Join-Path $resultRoot ('native_' + $label + '_' + [DateTime]::UtcNow.Ticks + '.log')
    $scriptPath = $linuxRoot + '/experiments/bohn2021_reproduction/' + $program
    & wsl.exe -d Ubuntu-20.04 --cd $linuxRoot -- $interpreter -u $scriptPath @arguments *> $log
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw ('Stage failed: ' + $label + '; exit=' + $code + '; log=' + $log) }
    $state.completed_stages += [ordered]@{stage=$label;exit_code=$code;log=$log;ended_utc=[DateTime]::UtcNow.ToString('o')}
    Save-Json $statePath $state
}

try {
    while ($true) {
        $training = Read-Json (Join-Path $resultRoot 'status.json')
        if ($training.pid -ne 1694525) { throw 'Unexpected training process identity' }
        $live = Matching-TrainingProcess $training.pid
        if (-not $training.active) {
            if (-not $training.complete) { throw ('Training stopped incomplete: ' + ($training | ConvertTo-Json -Compress)) }
            if (-not $live) { break }
        } elseif (-not $live) {
            # One re-read avoids mistaking an ordinary state-file transition for a crash.
            Start-Sleep -Seconds 10
            $training = Read-Json (Join-Path $resultRoot 'status.json')
            if ($training.active -and -not (Matching-TrainingProcess $training.pid)) {
                throw 'Training status is active but its process disappeared; preserve for recovery audit'
            }
        }
        Start-Sleep -Seconds 30
    }
    Execute-Stage 'evaluation_checks' $modernPython 'latency_tree_evaluation_checks.py' @()
    Execute-Stage 'training_delivery_checks' $modernPython 'latency_tree_training_delivery.py' @('--check')
    Execute-Stage 'register_evaluation' $modernPython 'latency_tree_evaluation_spec.py' @('--register')
    Execute-Stage 'audit_training_traces' $modernPython 'latency_tree_audit.py' @('--phase','train')
    Execute-Stage 'audit_learning_selection' $modernPython 'latency_tree_learning_audit.py' @()
    Execute-Stage 'freeze_fitted_policies' $modernPython 'latency_tree_evaluation_spec.py' @('--freeze-policies')
    Execute-Stage 'training_delivery' $modernPython 'latency_tree_training_delivery.py' @()
    Execute-Stage 'training_delivery_review' $modernPython 'latency_tree_training_delivery_review.py' @()
    Execute-Stage 'evaluation_smoke' $legacyPython 'latency_tree_evaluate.py' @('--split','smoke','--stage','smoke')
    Execute-Stage 'audit_evaluation_smoke' $modernPython 'latency_tree_evaluation_audit.py' @('--split','smoke','--stage','smoke')
    Execute-Stage 'validation_initial' $legacyPython 'latency_tree_evaluate.py' @('--split','validation','--stage','initial')
    Execute-Stage 'audit_validation_initial' $modernPython 'latency_tree_evaluation_audit.py' @('--split','validation','--stage','initial')
    Execute-Stage 'nominate_fixed_baselines' $modernPython 'latency_tree_baselines.py' @('--mode','nominate')
    Execute-Stage 'complete_fixed_baselines' $modernPython 'latency_tree_baselines.py' @('--mode','complete')
    Execute-Stage 'validation_supplement' $legacyPython 'latency_tree_evaluate.py' @('--split','validation','--stage','supplement')
    Execute-Stage 'audit_validation_full' $modernPython 'latency_tree_evaluation_audit.py' @('--split','validation','--stage','full')
    Execute-Stage 'timing_validation' $legacyPython 'latency_tree_evaluate.py' @('--split','validation','--stage','timing')
    Execute-Stage 'audit_timing_validation' $modernPython 'latency_tree_evaluation_audit.py' @('--split','validation','--stage','timing')
    Execute-Stage 'validation_effect' $modernPython 'latency_tree_effect.py' @('--split','validation')
    Execute-Stage 'validation_review' $modernPython 'latency_tree_effect_review.py' @('--split','validation')
    $gate = Read-Json (Join-Path $resultRoot 'validation_delivery/effect_gate.json')
    $review = Read-Json (Join-Path $resultRoot 'validation_delivery/independent_review.json')
    if (-not $review.passed) { throw 'Independent validation review failed' }
    if ($gate.effect_passed -ne $review.effect_passed) { throw 'Validation effect and review disagree' }
    if ($gate.effect_passed) {
        Execute-Stage 'freeze_confirmation' $modernPython 'latency_tree_effect.py' @('--freeze-confirmation')
        $state.test_accessed = $true
        Save-Json $statePath $state
        Execute-Stage 'confirmation_initial' $legacyPython 'latency_tree_evaluate.py' @('--split','test','--stage','initial')
        Execute-Stage 'audit_confirmation_initial' $modernPython 'latency_tree_evaluation_audit.py' @('--split','test','--stage','initial')
        Execute-Stage 'timing_confirmation' $legacyPython 'latency_tree_evaluate.py' @('--split','test','--stage','timing')
        Execute-Stage 'audit_timing_confirmation' $modernPython 'latency_tree_evaluation_audit.py' @('--split','test','--stage','timing')
        Execute-Stage 'confirmation_effect' $modernPython 'latency_tree_effect.py' @('--split','test')
        Execute-Stage 'confirmation_review' $modernPython 'latency_tree_effect_review.py' @('--split','test')
        $finalReview = Read-Json (Join-Path $resultRoot 'test_delivery/independent_review.json')
        if (-not $finalReview.passed) { throw 'Independent confirmation review failed' }
        $state.independent_effect_passed = $finalReview.effect_passed
        $state.stage = 'confirmation_review_complete_delivery_pending'
    } else {
        $state.independent_effect_passed = $false
        $state.stage = 'negative_validation_test_sealed'
    }
    $state.active = $false
    $state.complete = $true
    $state.ended_utc = [DateTime]::UtcNow.ToString('o')
    Save-Json $statePath $state
} catch {
    $state.active = $false
    $state.complete = $false
    $state.exception = $_.Exception.ToString()
    $state.ended_utc = [DateTime]::UtcNow.ToString('o')
    Save-Json $statePath $state
    throw
}
