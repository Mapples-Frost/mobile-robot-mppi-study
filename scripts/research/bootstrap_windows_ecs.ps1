#Requires -RunAsAdministrator

[CmdletBinding()]
param(
    [string]$ResearchRoot = "C:\Research",
    [string]$PythonUrl = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe",
    [string]$PythonSha256 = "D8DEDE5005564B408BA50317108B765ED9C3C510342A598F9FD42681CBE0648B"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$bootstrapRoot = Join-Path $ResearchRoot "bootstrap"
$cacheRoot = Join-Path $bootstrapRoot "cache"
$logRoot = Join-Path $ResearchRoot "logs"
$artifactRoot = Join-Path $ResearchRoot "artifacts"
$mirrorRoot = Join-Path $ResearchRoot "execution_mirror"
$pythonRoot = "C:\Python310"
$pythonExe = Join-Path $pythonRoot "python.exe"
$venvRoot = Join-Path $ResearchRoot "venv"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"

foreach ($path in @($ResearchRoot, $bootstrapRoot, $cacheRoot, $logRoot, $artifactRoot, $mirrorRoot)) {
    New-Item -ItemType Directory -Force -Path $path | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$transcript = Join-Path $logRoot "bootstrap_$timestamp.log"
Start-Transcript -Path $transcript -Force | Out-Null

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable $($Arguments -join ' ')"
    }
}

try {
    $scriptHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $PSCommandPath).Hash
    Write-Host "Bootstrap script SHA256: $scriptHash"

    if (-not (Test-Path -LiteralPath $pythonExe)) {
        $installer = Join-Path $cacheRoot "python-3.10.11-amd64.exe"
        if (-not (Test-Path -LiteralPath $installer)) {
            Invoke-WebRequest -Uri $PythonUrl -OutFile $installer
        }
        $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $installer).Hash
        if ($actualHash -ne $PythonSha256) {
            throw "Python installer SHA256 mismatch: expected $PythonSha256, got $actualHash"
        }
        $installArgs = @(
            "/quiet",
            "InstallAllUsers=1",
            "PrependPath=0",
            "Include_launcher=0",
            "Include_test=0",
            "Include_doc=0",
            "TargetDir=$pythonRoot"
        )
        $process = Start-Process -FilePath $installer -ArgumentList $installArgs -Wait -PassThru
        if ($process.ExitCode -notin @(0, 3010)) {
            throw "Python installer failed with exit code $($process.ExitCode)"
        }
    }

    Invoke-NativeChecked $pythonExe "--version"

    if (-not (Test-Path -LiteralPath $venvPython)) {
        Invoke-NativeChecked $pythonExe "-m" "venv" $venvRoot
    }

    Invoke-NativeChecked $venvPython "-m" "pip" "install" "--disable-pip-version-check" `
        "pip==25.1.1" "setuptools==75.8.2" "wheel==0.45.1"
    Invoke-NativeChecked $venvPython "-m" "pip" "install" "--disable-pip-version-check" `
        "--index-url" "https://download.pytorch.org/whl/cpu" "torch==2.7.0+cpu"
    Invoke-NativeChecked $venvPython "-m" "pip" "install" "--disable-pip-version-check" `
        "numpy==1.26.4" `
        "PyYAML==6.0.3" `
        "mujoco==3.2.3" `
        "scipy==1.15.3" `
        "scikit-learn==1.7.2" `
        "matplotlib==3.7.5" `
        "pytest==8.4.2" `
        "psutil==6.1.1"

    # Keep formal jobs single-threaded internally; worker-level parallelism is
    # assigned later by the qualification scheduler. Do not set global thread
    # environment variables here because the isolated timing cohort records
    # and owns its execution environment explicitly.
    try {
        Invoke-NativeChecked "powercfg.exe" "/S" "SCHEME_MIN"
    }
    catch {
        Write-Warning "High-performance power plan could not be selected: $($_.Exception.Message)"
    }

    $verifyCode = @'
import json
import platform
import sys
import mujoco
import numpy
import psutil
import scipy
import sklearn
import torch
import yaml

payload = {
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "logical_cpu_count": psutil.cpu_count(logical=True),
    "physical_cpu_count": psutil.cpu_count(logical=False),
    "memory_bytes": psutil.virtual_memory().total,
    "versions": {
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "mujoco": mujoco.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pyyaml": yaml.__version__,
        "psutil": psutil.__version__,
    },
}
print(json.dumps(payload, indent=2, sort_keys=True))
'@
    # Windows PowerShell 5.1 uses legacy native-argument quoting. Passing the
    # multiline program through `python -c` strips the embedded JSON-key
    # quotes, turning e.g. `"python"` into the undefined name `python`.
    # Execute an auditable file instead so the verification source is
    # byte-preserved and retained with the bootstrap artifacts.
    $verifyScript = Join-Path $bootstrapRoot "verify_python_inventory.py"
    $verifyCode | Out-File -LiteralPath $verifyScript -Encoding utf8
    $inventoryJson = & $venvPython $verifyScript
    if ($LASTEXITCODE -ne 0) {
        throw "Python import and inventory verification failed with exit code $LASTEXITCODE"
    }
    $inventoryJson | Out-File -LiteralPath (Join-Path $bootstrapRoot "python_inventory.json") -Encoding utf8

    & $venvPython "-m" "pip" "freeze" | Sort-Object |
        Out-File -LiteralPath (Join-Path $bootstrapRoot "pip_freeze.txt") -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        throw "pip freeze failed with exit code $LASTEXITCODE"
    }

    @(
        "Win32_ComputerSystem"
        "Win32_Processor"
        "Win32_OperatingSystem"
    ) | ForEach-Object { Get-CimInstance -ClassName $_ } |
        Select-Object PSComputerName, __CLASS, Manufacturer, Model, Name, Caption,
            NumberOfCores, NumberOfLogicalProcessors, TotalPhysicalMemory, Version, BuildNumber |
        ConvertTo-Json -Depth 4 |
        Out-File -LiteralPath (Join-Path $bootstrapRoot "windows_inventory.json") -Encoding utf8

    $status = [ordered]@{
        status = "pass"
        completed_at = (Get-Date).ToString("o")
        script_sha256 = $scriptHash
        python_installer_sha256 = $PythonSha256
        python_executable = $venvPython
        research_root = $ResearchRoot
        formal_experiment_started = $false
    }
    $status | ConvertTo-Json |
        Out-File -LiteralPath (Join-Path $bootstrapRoot "bootstrap_status.json") -Encoding utf8

    Write-Host "BOOTSTRAP PASS"
    Write-Host "Python: $venvPython"
    Write-Host "Inventory: $(Join-Path $bootstrapRoot 'python_inventory.json')"
}
finally {
    Stop-Transcript | Out-Null
}
