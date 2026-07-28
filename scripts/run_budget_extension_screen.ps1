<#
.SYNOPSIS
  Run the chapter-1 budget-extension screen: 3 continuation episodes, 1200 -> 2050 steps.

.DESCRIPTION
  Each episode reuses the exact seed and configuration of the corresponding
  segment_half_thickness_ab_v1 treatment episode, changing only
  experiment.max_steps. The analyzer verifies that steps 0-1199 reproduce the
  reference artifact exactly and voids the comparison if they do not.

  Seeds 791101302 / 791101305 / 791101310 are the front_obstacle_slow mode,
  still gaining 1.10-1.76 m of route in their final 30 s at the 1200-step cutoff.
  The near_body_hard_stop seeds (301, 307, 311) are excluded: they are arrested
  at v ~ 0.0004 m/s with no legal escape, so extra time cannot help them.

  Protocol: configs/research/budget_extension_screen_v1.yaml
  Evidence : research_artifacts/budget_extension_screen_v1/

.EXAMPLE
  .\scripts\run_budget_extension_screen.ps1
  .\scripts\run_budget_extension_screen.ps1 -Resume
  .\scripts\run_budget_extension_screen.ps1 -AnalyzeOnly
#>

param(
    [switch]$Resume,
    [switch]$AnalyzeOnly
)

$ErrorActionPreference = 'Stop'

$repo   = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
$root   = Join-Path $repo 'research_artifacts\budget_extension_screen_v1'
$seeds  = @(791101302, 791101305, 791101310)

if (-not (Test-Path $python)) { throw "venv interpreter not found: $python" }

if (-not $AnalyzeOnly) {

    $running = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python' -and $_.CommandLine -like "*$repo*"
    })
    if ($running.Count -gt 0) {
        throw "A project Python process is already running (pid $($running[0].ProcessId))."
    }

    New-Item -ItemType Directory -Force -Path $root | Out-Null

    $done = 0
    $started = Get-Date

    foreach ($seed in $seeds) {
        $done++
        $output = Join-Path $root ("seed{0}" -f $seed)
        $stdout = Join-Path $root ("seed{0}.stdout.log" -f $seed)
        $stderr = Join-Path $root ("seed{0}.stderr.log" -f $seed)

        if (Test-Path (Join-Path $output 'metrics.json')) {
            if ($Resume) {
                Write-Host ("[{0}/{1}] skip  seed {2} (already complete)" -f $done, $seeds.Count, $seed)
                continue
            }
            throw "Evidence already exists at $output. Re-run with -Resume, or move it aside."
        }
        if ((Test-Path $output) -and (Get-ChildItem $output | Measure-Object).Count -gt 0) {
            throw "Partial evidence at $output. Inspect and move it aside before re-running."
        }

        Write-Host ("[{0}/{1}] run   seed {2}  (max_steps 2050)" -f $done, $seeds.Count, $seed) -NoNewline
        $t0 = Get-Date

        & $python -m experiments.dynamic_uncertainty.run_budget_extension_screen `
            --seed $seed --output $output 1> $stdout 2> $stderr

        if ($LASTEXITCODE -ne 0) {
            Write-Host ''
            Write-Host '--- stderr ---' -ForegroundColor Red
            Get-Content $stderr -Tail 30 | Write-Host
            throw "Episode failed: seed $seed (exit $LASTEXITCODE). Evidence retained."
        }

        $m = Get-Content (Join-Path $output 'metrics.json') -Raw | ConvertFrom-Json
        Write-Host ("  ->  success={0} collision={1} steps={2} dist={3:N2}  [{4:N0}s]" -f `
            $m.success, $m.collision, $m.steps, $m.final_goal_distance, `
            ((Get-Date) - $t0).TotalSeconds)
    }

    Write-Host ''
    Write-Host ("All {0} episodes complete in {1:N1} min." -f $seeds.Count, `
        ((Get-Date) - $started).TotalMinutes)
    Write-Host ''
}

& $python -m experiments.dynamic_uncertainty.analyze_budget_extension_screen --root $root
