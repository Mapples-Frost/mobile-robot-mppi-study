param([switch]$ValidateOnly)

# Post-evaluation accounting. Waits natively so timing remains serial.
$ErrorActionPreference = 'Stop'
$studyRoot = '\\wsl.localhost\Ubuntu-20.04\home\mapples\projects\mobile-robot-mppi-study'
$linuxRoot = '/home/mapples/projects/mobile-robot-mppi-study'
$resultRoot = Join-Path $studyRoot 'research_artifacts/bohn2021_reproduction_2026-09-17/results/latency_tree_2026-09-26'
$modernPython = $linuxRoot + '/.venv/bin/python'
$registrationPath = Join-Path $resultRoot 'budget_finish_registration.json'
$statePath = Join-Path $resultRoot 'budget_finish_status.json'
$upstreamStatePath = Join-Path $resultRoot 'native_pipeline_v2_status.json'

function Read-Json([string]$path) {
    Get-Content -LiteralPath $path -Raw | ConvertFrom-Json -AsHashtable
}
function Digest([string]$path) {
    (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Save-Json([string]$path, $value) {
    $temporary = $path + '.tmp'
    [IO.File]::WriteAllText($temporary, ($value | ConvertTo-Json -Depth 50) + [Environment]::NewLine,
                           [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $path -Force
}
function Verify-Sources($hashes) {
    foreach($relative in $hashes.Keys) {
        if((Digest (Join-Path $studyRoot $relative)) -ne $hashes[$relative]) {
            throw ('Frozen budget source changed: ' + $relative)
        }
    }
}
function Upstream-IsLive([long]$processNumber) {
    $process = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $processNumber)
    return $null -ne $process -and $process.CommandLine.Contains('latency_tree_native_pipeline.ps1')
}
$names = @('latency_tree_final_budget.py', 'latency_tree_final_budget_review.py',
           'latency_tree_budget_finish.ps1')
if(-not (Test-Path -LiteralPath $registrationPath)) {
    throw 'Budget controller requires separately frozen source registration'
}
$registration = Read-Json $registrationPath
if($registration.source_hashes.Count -ne $names.Count) { throw 'Budget source count mismatch' }
foreach($name in $names) {
    if(-not $registration.source_hashes.ContainsKey('experiments/bohn2021_reproduction/' + $name)) {
        throw ('Budget source missing: ' + $name)
    }
}
Verify-Sources $registration.source_hashes
$upstreamRegistration = Join-Path $resultRoot 'native_pipeline_v2_registration.json'
if((Digest $upstreamRegistration) -ne $registration.upstream_registration_sha256) {
    throw 'Upstream execution registration changed; inspect before continuing'
}
if($ValidateOnly) {
    [ordered]@{validated=$true;source_count=$names.Count;python_started=$false;upstream_touched=$false} |
        ConvertTo-Json
    exit 0
}
if(Test-Path -LiteralPath $statePath) { throw 'Existing budget controller state requires inspection' }
$state = [ordered]@{
    pid=$PID; active=$true; complete=$false; stage='waiting_for_pipeline_exit'
    upstream_pid=26188; started_utc=[DateTime]::UtcNow.ToString('o')
    completed_stages=@(); registration_sha256=(Digest $registrationPath)
    goal_complete=$false
}
Save-Json $statePath $state
function Execute-BudgetStage([string]$label, [string]$program, [string[]]$arguments) {
    Verify-Sources $registration.source_hashes
    $state.stage=$label
    Save-Json $statePath $state
    $log=Join-Path $resultRoot ('budget_' + $label + '_' + [DateTime]::UtcNow.Ticks + '.log')
    $programPath=$linuxRoot + '/experiments/bohn2021_reproduction/' + $program
    & wsl.exe -d Ubuntu-20.04 --cd $linuxRoot -- $modernPython -u $programPath @arguments *> $log
    $code=$LASTEXITCODE
    if($code -ne 0) { throw ('Budget stage failed: ' + $label + '; code=' + $code + '; log=' + $log) }
    $state.completed_stages += [ordered]@{stage=$label;exit_code=$code;log=$log;ended_utc=[DateTime]::UtcNow.ToString('o')}
    Save-Json $statePath $state
}
try {
    while($true) {
        $upstream=Read-Json $upstreamStatePath
        if($upstream.pid -ne 26188) { throw 'Upstream PID changed; inspect new controller before proceeding' }
        $live=Upstream-IsLive $upstream.pid
        if(-not $upstream.active) {
            if(-not $upstream.complete) { throw ('Upstream pipeline incomplete: ' + $upstream.stage) }
            if(-not $live) { break }
        } elseif(-not $live) {
            Start-Sleep -Seconds 10
            $upstream=Read-Json $upstreamStatePath
            if($upstream.active -and -not (Upstream-IsLive $upstream.pid)) {
                throw 'Upstream status active but matching process disappeared; inspect'
            }
        }
        Start-Sleep -Seconds 45
    }
    if($upstream.stage -notin @('negative_validation_test_sealed','confirmation_review_complete_delivery_pending')) {
        throw ('Unexpected completed upstream stage: ' + $upstream.stage)
    }
    Execute-BudgetStage 'checks' 'latency_tree_final_budget.py' @('--check')
    Execute-BudgetStage 'register' 'latency_tree_final_budget.py' @('--register')
    Execute-BudgetStage 'ledger' 'latency_tree_final_budget.py' @()
    Execute-BudgetStage 'review' 'latency_tree_final_budget_review.py' @()
    $review=Read-Json (Join-Path $resultRoot 'final_budget/independent_review.json')
    if(-not $review.passed) { throw 'Independent budget review did not pass' }
    $state.stage='budget_complete_final_delivery_pending'
    $state.complete_budget=$review.complete_budget
    $state.active=$false
    $state.complete=$true
    $state.ended_utc=[DateTime]::UtcNow.ToString('o')
    Save-Json $statePath $state
} catch {
    $state.active=$false
    $state.complete=$false
    $state.exception=$_.Exception.ToString()
    $state.ended_utc=[DateTime]::UtcNow.ToString('o')
    Save-Json $statePath $state
    throw
}

