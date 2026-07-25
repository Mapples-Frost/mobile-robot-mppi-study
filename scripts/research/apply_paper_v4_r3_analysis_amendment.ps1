param(
    [string]$Mirror = "C:\Research\execution_mirror\paper_v4_r2_20260725_1820",
    [string]$Python = "C:\Research\venv\Scripts\python.exe",
    [string]$PatchArchiveSha256 = ""
)

$ErrorActionPreference = "Stop"
$patchRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$amendmentRelative = "docs\experiments\dynamic_uncertainty\SINGLE_DYNAMIC_OBSTACLE_PAPER_V4_REVISION3_ANALYSIS_AMENDMENT.json"
$sourceAmendment = Join-Path $patchRoot $amendmentRelative
$registry = Join-Path $Mirror "configs\seeds\single_dynamic_obstacle_paper_v4_sealed.yaml"
$qualification = Join-Path $Mirror "research_artifacts\single_dynamic_obstacle_paper_v4_qualification\qualification_report.json"
$formalOutput = "C:\Research\single_dynamic_obstacle_paper_v4_formal"
$backup = "C:\Research\bootstrap\paper_v4_r2_pre_revision3_analysis_backup"

if (-not (Test-Path -LiteralPath $Mirror)) {
    throw "R2 mirror missing: $Mirror"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Research Python missing: $Python"
}
if (Test-Path -LiteralPath $formalOutput) {
    throw "Formal output already exists; Revision 3 cannot be applied after formal start: $formalOutput"
}
if (Test-Path -LiteralPath $backup) {
    throw "Revision-3 backup already exists; refusing a second application: $backup"
}
if (-not (Test-Path -LiteralPath $sourceAmendment)) {
    throw "Patch amendment manifest missing: $sourceAmendment"
}

$amendment = Get-Content -LiteralPath $sourceAmendment -Raw | ConvertFrom-Json
if ($amendment.formal_experiment_started -ne $false -or $amendment.formal_outcomes_exist -ne $false) {
    throw "Patch is not marked pre-formal"
}
$registryHash = (Get-FileHash -LiteralPath $registry -Algorithm SHA256).Hash.ToLower()
if ($registryHash -ne $amendment.sealed_registry.sha256) {
    throw "Sealed registry changed: $registryHash"
}
$qualificationReport = Get-Content -LiteralPath $qualification -Raw | ConvertFrom-Json
if (
    $qualificationReport.status -ne "pass" -or
    $qualificationReport.formal_registry_authorized -ne $true -or
    $qualificationReport.seed_blocks -ne 16 -or
    $qualificationReport.episode_jobs -ne 112 -or
    $qualificationReport.missing.Count -ne 0 -or
    $qualificationReport.integrity_failures.Count -ne 0
) {
    throw "Passing Revision-2 qualification contract is not present"
}

foreach ($property in $amendment.superseded_revision_2_hashes.PSObject.Properties) {
    $relative = $property.Name.Replace("/", "\")
    $path = Join-Path $Mirror $relative
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $property.Value) {
        throw "Unexpected R2 source before patch: $relative actual=$actual"
    }
}

New-Item -ItemType Directory -Force -Path $backup | Out-Null
foreach ($property in $amendment.superseded_revision_2_hashes.PSObject.Properties) {
    $relative = $property.Name.Replace("/", "\")
    $source = Join-Path $Mirror $relative
    $destination = Join-Path $backup $relative
    New-Item -ItemType Directory -Force -Path (Split-Path $destination) | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
}

foreach ($property in $amendment.revision_3_hashes.PSObject.Properties) {
    $relative = $property.Name.Replace("/", "\")
    $source = Join-Path $patchRoot $relative
    $destination = Join-Path $Mirror $relative
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Revision-3 patch file missing: $source"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $destination) | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Force
    $actual = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $property.Value) {
        throw "Revision-3 hash mismatch after copy: $relative actual=$actual"
    }
}

$destinationAmendment = Join-Path $Mirror $amendmentRelative
New-Item -ItemType Directory -Force -Path (Split-Path $destinationAmendment) | Out-Null
Copy-Item -LiteralPath $sourceAmendment -Destination $destinationAmendment -Force

$env:PYTHONPATH = "$Mirror\src;$Mirror"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:PYTHONHASHSEED = "0"
$env:CUDA_VISIBLE_DEVICES = ""

$runner = Join-Path $Mirror "experiments\dynamic_uncertainty\run_single_dynamic_obstacle_paper_v4.py"
$analyzer = Join-Path $Mirror "experiments\dynamic_uncertainty\analyze_single_dynamic_obstacle_paper_v4.py"
& $Python -m py_compile $runner $analyzer
if ($LASTEXITCODE -ne 0) {
    throw "Revision-3 Python compile failed"
}

Push-Location $Mirror
try {
    $preflightText = (& $Python $runner preflight | Out-String)
    if ($LASTEXITCODE -ne 0) {
        throw "Revision-3 preflight failed: exit code=$LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
$preflight = $preflightText | ConvertFrom-Json
if (
    $preflight.status -ne "preflight_pass" -or
    $preflight.component_construction.status -ne "pass" -or
    $preflight.factor_separability.status -ne "pass" -or
    $preflight.qualification_episode_jobs -ne 112
) {
    throw "Revision-3 preflight contract failed"
}

$preflightPath = Join-Path $Mirror "research_artifacts\single_dynamic_obstacle_paper_v4_qualification\revision3_preformal_preflight.json"
$preflight | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $preflightPath -Encoding UTF8
$record = [ordered]@{
    schema_version = 1
    status = "revision3_analysis_amendment_applied_preformal"
    applied_at = (Get-Date).ToString("o")
    mirror = $Mirror
    patch_archive_sha256 = $PatchArchiveSha256.ToLower()
    sealed_registry_sha256 = $registryHash
    qualification_status = $qualificationReport.status
    qualification_episode_jobs = $qualificationReport.episode_jobs
    preflight_status = $preflight.status
    preflight_manifest_sha256 = $preflight.execution_manifest.manifest_sha256
    preflight_runtime_sha256 = $preflight.execution_manifest.runtime_sha256
    formal_experiment_started = $false
    backup = $backup
}
$recordPath = Join-Path $Mirror "research_artifacts\single_dynamic_obstacle_paper_v4_qualification\revision3_application_record.json"
$record | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $recordPath -Encoding UTF8

$finalRegistryHash = (Get-FileHash -LiteralPath $registry -Algorithm SHA256).Hash.ToLower()
if ($finalRegistryHash -ne $registryHash) {
    throw "Registry changed while applying Revision 3"
}

Write-Host "REVISION 3 ANALYSIS AMENDMENT PASS" -ForegroundColor Green
$record | ConvertTo-Json -Depth 10
