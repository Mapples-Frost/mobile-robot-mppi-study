<#
.SYNOPSIS
  Run the frozen segment-half-thickness A/B screen (12 paired seeds, 24 episodes).

.DESCRIPTION
  Sequential, one episode at a time, matching the project's existing execution
  discipline. Refuses to overwrite any existing evidence. Runs the control arm
  (bit-exact historical sampler) and the treatment arm (ar1:2.0) for each seed,
  then applies the frozen decision rule.

  Protocol: configs/research/segment_half_thickness_ab_development_v1.yaml
  Evidence : research_artifacts/segment_half_thickness_ab_v1/

.EXAMPLE
  .\scripts\run_segment_half_thickness_ab.ps1
  .\scripts\run_segment_half_thickness_ab.ps1 -Resume        # skip completed episodes
  .\scripts\run_segment_half_thickness_ab.ps1 -AnalyzeOnly   # just re-apply the decision rule
#>

param(
    [switch]$Resume,
    [switch]$AnalyzeOnly
)

$ErrorActionPreference = 'Stop'

$repo    = Split-Path -Parent $PSScriptRoot
$python  = Join-Path $repo '.venv\Scripts\python.exe'
$root    = Join-Path $repo 'research_artifacts\segment_half_thickness_ab_v1'
$mapName = 'chapter1'
$seeds   = 791101301..791101312
$arms    = @('control', 'treatment')

if (-not (Test-Path $python)) { throw "venv interpreter not found: $python" }

if (-not $AnalyzeOnly) {

    # Refuse to start if any Python process from this checkout is already in
    # flight. This catches both ``-m`` and direct-script invocations and avoids
    # contaminating planner timing with a viewer or another experiment.
    $running = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python' -and
        $_.CommandLine -like "*$repo*"
    })
    if ($running.Count -gt 0) {
        throw "A project Python process is already running (pid $($running[0].ProcessId))."
    }

    New-Item -ItemType Directory -Force -Path $root | Out-Null

    $total = $seeds.Count * $arms.Count
    $done  = 0
    $started = Get-Date

    foreach ($seed in $seeds) {
        foreach ($arm in $arms) {
            $done++
            $output = Join-Path $root ("seed{0}\{1}" -f $seed, $arm)
            $stdout = Join-Path $root ("seed{0}.{1}.stdout.log" -f $seed, $arm)
            $stderr = Join-Path $root ("seed{0}.{1}.stderr.log" -f $seed, $arm)

            if (Test-Path (Join-Path $output 'metrics.json')) {
                if ($Resume) {
                    Write-Host ("[{0}/{1}] skip  seed {2} {3} (already complete)" -f $done, $total, $seed, $arm)
                    continue
                }
                throw "Evidence already exists at $output. Re-run with -Resume, or move it aside."
            }
            if ((Test-Path $output) -and (Get-ChildItem $output | Measure-Object).Count -gt 0) {
                throw "Partial evidence at $output. Inspect and move it aside before re-running."
            }

            Write-Host ("[{0}/{1}] run   seed {2} {3}" -f $done, $total, $seed, $arm) -NoNewline
            $t0 = Get-Date

            & $python -m experiments.dynamic_uncertainty.run_segment_half_thickness_ab `
                --map $mapName --seed $seed --arm $arm --output $output `
                1> $stdout 2> $stderr

            if ($LASTEXITCODE -ne 0) {
                Write-Host ''
                Write-Host "--- stderr ---" -ForegroundColor Red
                Get-Content $stderr -Tail 30 | Write-Host
                throw "Episode failed: seed $seed arm $arm (exit $LASTEXITCODE). Evidence retained."
            }

            $metrics = Get-Content (Join-Path $output 'metrics.json') -Raw | ConvertFrom-Json
            Write-Host ("  ->  success={0} collision={1} steps={2} dist={3:N2}  [{4:N0}s]" -f `
                $metrics.success, $metrics.collision, $metrics.steps, `
                $metrics.final_goal_distance, ((Get-Date) - $t0).TotalSeconds)
        }
    }

    Write-Host ''
    Write-Host ("All {0} episodes complete in {1:N1} min." -f $total, ((Get-Date) - $started).TotalMinutes)
    Write-Host ''
}

& $python -m experiments.dynamic_uncertainty.analyze_segment_half_thickness_ab --root $root
