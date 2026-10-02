[CmdletBinding()]
param(
    [int]$Camera = 0,
    [double]$Seconds = 3,
    [ValidateSet('auto', 'dshow', 'msmf', 'avfoundation')][string]$Api = 'auto',
    [switch]$Preview
)
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    $cameraArgs = @('run', '--no-managed-python', '--no-sync', 'posture-lab', 'camera-check',
        '--camera', $Camera, '--seconds', $Seconds, '--api', $Api)
    if ($Preview) { $cameraArgs += '--preview' }
    & uv @cameraArgs
    if ($LASTEXITCODE -ne 0) { throw "Camera check exited with code $LASTEXITCODE." }
} finally {
    Pop-Location
}
