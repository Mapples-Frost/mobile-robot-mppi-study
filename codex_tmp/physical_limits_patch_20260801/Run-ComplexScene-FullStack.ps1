param(
    [ValidateSet("Shadow", "Live")]
    [string]$Mode = "Shadow",
    [double]$GoalX = 2.0,
    [double]$GoalY = 0.0,
    [double]$BoundaryMinX = -0.75,
    [double]$BoundaryMaxX = 3.50,
    [double]$BoundaryMinY = -1.50,
    [double]$BoundaryMaxY = 1.50,
    [double]$MaxLinearMps = 0.10,
    [double]$MaxReverseMps = 0.10,
    [double]$MaxAngularRadps = 0.35,
    [double]$DurationSeconds = 60.0,
    [double]$PeriodSeconds = 0.35,
    [int]$WarmupCycles = 8,
    [string]$StaticBackgroundMap = (Join-Path $PSScriptRoot "calibration\static_background_20260731T105310Z_final\map_v1\static_background_map.npz"),
    [string]$ConfirmMotion = ""
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "Run-FullStack.ps1"

& $runner `
    -Mode $Mode `
    -Algorithm "B11FullProposed" `
    -ObstacleProfile "ComplexScene" `
    -GoalX $GoalX `
    -GoalY $GoalY `
    -BoundaryMinX $BoundaryMinX `
    -BoundaryMaxX $BoundaryMaxX `
    -BoundaryMinY $BoundaryMinY `
    -BoundaryMaxY $BoundaryMaxY `
    -MaxLinear $MaxLinearMps `
    -MaxReverse $MaxReverseMps `
    -MaxAngular $MaxAngularRadps `
    -DurationSeconds $DurationSeconds `
    -PeriodSeconds $PeriodSeconds `
    -WarmupCycles $WarmupCycles `
    -StaticBackgroundMap $StaticBackgroundMap `
    -ConfirmMotion $ConfirmMotion

exit $LASTEXITCODE
