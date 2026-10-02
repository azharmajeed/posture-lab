[CmdletBinding()]
param(
    [ValidateSet('auto', 'laptop', 'left-monitor')][string]$Profile = 'auto',
    [switch]$Background,
    [switch]$NoSync
)
& (Join-Path $PSScriptRoot 'start.ps1') -Config 'configs/hindi-roast.json' -Reminder roast `
    -Profile $Profile -Background:$Background -NoSync:$NoSync
