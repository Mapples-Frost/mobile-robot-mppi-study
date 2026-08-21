param(
    [string]$RobotHost = "192.168.31.200",
    [string]$RobotUser = "eaibot",
    [string]$SshKey = "C:\Users\lenovo\AppData\Local\Temp\codex_eaibot_inspect_8e04e093b32f43d8910d73a0025e7c5d\id_ed25519"
)

$ErrorActionPreference = "Stop"
$remoteDir = "/home/eaibot/deployments/rl_mppi_reverse_full_stack_20260802_responsive_vehicle_v17"
$files = @(
    "reverse_safety_contract.py",
    "safe_command_gateway_v2.py",
    "capture_static_background_rotation.py",
    "capture_static_background.sh",
    "build_static_background_map.py",
    "start_control_session.sh",
    "stop_control_session.sh",
    "status_control_stack.sh"
)

if (-not (Test-Path -LiteralPath $SshKey)) {
    throw "SSH key not found: $SshKey"
}

$sshBase = @(
    "-i", $SshKey,
    "-o", "BatchMode=yes",
    "-o", "IdentitiesOnly=yes",
    "-o", "ConnectTimeout=6",
    "$RobotUser@$RobotHost"
)

& ssh.exe @sshBase "mkdir -p '$remoteDir'"
if ($LASTEXITCODE -ne 0) {
    throw "Could not create remote deployment directory"
}

foreach ($file in $files) {
    $local = Join-Path $PSScriptRoot $file
    & scp.exe -i $SshKey -o BatchMode=yes -o IdentitiesOnly=yes `
        $local "$RobotUser@${RobotHost}:$remoteDir/$file"
    if ($LASTEXITCODE -ne 0) {
        throw "Upload failed: $file"
    }
}

$verify = @"
set -e
chmod 0755 '$remoteDir/'*.sh '$remoteDir/safe_command_gateway_v2.py' '$remoteDir/capture_static_background_rotation.py' '$remoteDir/build_static_background_map.py'
python -m py_compile '$remoteDir/reverse_safety_contract.py' '$remoteDir/safe_command_gateway_v2.py' '$remoteDir/capture_static_background_rotation.py' '$remoteDir/build_static_background_map.py'
cd '$remoteDir'
sha256sum reverse_safety_contract.py safe_command_gateway_v2.py capture_static_background_rotation.py capture_static_background.sh build_static_background_map.py start_control_session.sh stop_control_session.sh status_control_stack.sh > DEPLOYMENT.SHA256
ln -sfn '$remoteDir' /home/eaibot/deployments/rl_mppi_reverse_full_stack_current
echo DEPLOYED_DIR='$remoteDir'
cat DEPLOYMENT.SHA256
"@
& ssh.exe @sshBase $verify
if ($LASTEXITCODE -ne 0) {
    throw "Remote compile or hash verification failed"
}
