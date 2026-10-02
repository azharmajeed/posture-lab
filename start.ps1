[CmdletBinding()]
param(
    [string]$Config = 'configs/laptop.json',
    [ValidateSet('notification', 'voice', 'roast')][string]$Reminder,
    [ValidateSet('auto', 'laptop', 'left-monitor')][string]$Profile = 'auto',
    [switch]$Background,
    [switch]$NoSync
)

$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    $uvPath = (Get-Command uv -ErrorAction Stop).Source
    $configPath = (Resolve-Path -LiteralPath $Config).Path
    $uvArgs = @('run', '--no-managed-python')
    if ($NoSync) {
        $uvArgs += '--no-sync'
    } else {
        $uvArgs += @('--frozen', '--extra', 'local', '--extra', 'voice', '--extra', 'dev')
    }

    # Prepare once, then reuse WAVs across sessions. No synthesis on the reminder path.
    $settings = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    if ($settings.reminders.tts_engine -ne 'system' -and
        (Test-Path -LiteralPath 'models/voices/en_US-lessac-medium.onnx') -and
        (Test-Path -LiteralPath 'models/voices/hi_IN-pratham-medium.onnx')) {
        & $uvPath @uvArgs posture-lab prepare-voices --config $configPath
        if ($LASTEXITCODE -ne 0) { throw 'Could not prepare voice audio; see the error above.' }
    }

    $runArgs = $uvArgs + @('posture-lab', 'run', '--config', $configPath, '--profile', $Profile)
    if ($Reminder) { $runArgs += @('--reminder', $Reminder) }
    if ($Background) {
        $runDirectory = Join-Path $PSScriptRoot '.run'
        New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
        # Paths can contain spaces; Start-Process joins ArgumentList into one command line.
        $quotedArgs = $runArgs | ForEach-Object { '"' + $_ + '"' }
        $process = Start-Process -FilePath $uvPath -ArgumentList $quotedArgs -WorkingDirectory $PSScriptRoot `
            -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $runDirectory 'posture.stdout.log') `
            -RedirectStandardError (Join-Path $runDirectory 'posture.stderr.log')
        Set-Content -LiteralPath (Join-Path $runDirectory 'posture.pid') -Value $process.Id
        Write-Host "Posture Lab started (PID $($process.Id)). Q closes the camera preview."
    } else {
        & $uvPath @runArgs
        if ($LASTEXITCODE -ne 0) { throw "Posture Lab exited with code $LASTEXITCODE." }
    }
} finally {
    Pop-Location
}
