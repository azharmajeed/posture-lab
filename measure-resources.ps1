param(
    [ValidateRange(2, 60)]
    [int]$Seconds = 10,
    [int]$RootProcessId = 0
)

$ErrorActionPreference = 'Stop'
if ($RootProcessId -eq 0) {
    $taskPidFile = Join-Path $PSScriptRoot '.run\posture.pid'
    if (-not (Test-Path -LiteralPath $taskPidFile)) {
        throw 'Start the app first, or pass -RootProcessId for its launcher.'
    }
    $RootProcessId = [int](Get-Content -LiteralPath $taskPidFile -Raw)
}
$taskRoot = Get-CimInstance Win32_Process -Filter "ProcessId = $RootProcessId"
if ($null -eq $taskRoot -or $taskRoot.CommandLine -notmatch 'posture[-_]lab') {
    throw 'The saved process is not the running posture app. Pass its current launcher PID.'
}

$taskProcesses = @(Get-CimInstance Win32_Process)
$taskProcessIds = [System.Collections.Generic.HashSet[int]]::new()
[void]$taskProcessIds.Add($RootProcessId)
do {
    $taskAdded = $false
    foreach ($taskProcess in $taskProcesses) {
        if ($taskProcessIds.Contains([int]$taskProcess.ParentProcessId)) {
            if ($taskProcessIds.Add([int]$taskProcess.ProcessId)) { $taskAdded = $true }
        }
    }
} while ($taskAdded)

$taskLogicalCpus = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
function Read-AppResources {
    $taskCpuSeconds = 0.0
    $taskWorkingBytes = 0.0
    $taskPrivateBytes = 0.0
    $taskFound = 0
    foreach ($taskProcessId in $taskProcessIds) {
        $taskLive = Get-Process -Id $taskProcessId -ErrorAction SilentlyContinue
        if ($null -eq $taskLive) { continue }
        $taskCpuSeconds += $taskLive.TotalProcessorTime.TotalSeconds
        $taskWorkingBytes += $taskLive.WorkingSet64
        $taskPrivateBytes += $taskLive.PrivateMemorySize64
        $taskFound++
    }
    if ($taskFound -eq 0) { throw 'The posture app exited during measurement.' }
    [pscustomobject]@{
        CpuSeconds = $taskCpuSeconds
        WorkingMiB = $taskWorkingBytes / 1MB
        PrivateMiB = $taskPrivateBytes / 1MB
        ProcessCount = $taskFound
    }
}

$taskSamples = [System.Collections.Generic.List[object]]::new()
$taskClock = [System.Diagnostics.Stopwatch]::StartNew()
$taskPrevious = Read-AppResources
$taskPreviousTime = $taskClock.Elapsed.TotalSeconds
for ($taskSampleIndex = 0; $taskSampleIndex -lt $Seconds; $taskSampleIndex++) {
    Start-Sleep -Milliseconds 1000
    $taskCurrent = Read-AppResources
    $taskCurrentTime = $taskClock.Elapsed.TotalSeconds
    $taskCpuPercent = 100 * ($taskCurrent.CpuSeconds - $taskPrevious.CpuSeconds) /
        ($taskCurrentTime - $taskPreviousTime) / $taskLogicalCpus
    if ($taskCpuPercent -lt 0) { throw 'The process tree changed; rerun the measurement.' }
    $taskSamples.Add([pscustomobject]@{
        CpuPercent = $taskCpuPercent
        WorkingMiB = $taskCurrent.WorkingMiB
        PrivateMiB = $taskCurrent.PrivateMiB
    })
    $taskPrevious = $taskCurrent
    $taskPreviousTime = $taskCurrentTime
}
$taskReport = [ordered]@{
    root_pid = $RootProcessId
    processes_measured = $taskPrevious.ProcessCount
    logical_processors = $taskLogicalCpus
    sample_seconds = [math]::Round($taskClock.Elapsed.TotalSeconds, 1)
    cpu_average_percent_of_machine = [math]::Round(
        ($taskSamples | Measure-Object CpuPercent -Average).Average, 1)
    cpu_peak_percent_of_machine = [math]::Round(
        ($taskSamples | Measure-Object CpuPercent -Maximum).Maximum, 1)
    resident_ram_average_mib = [math]::Round(
        ($taskSamples | Measure-Object WorkingMiB -Average).Average, 1)
    resident_ram_peak_mib = [math]::Round(
        ($taskSamples | Measure-Object WorkingMiB -Maximum).Maximum, 1)
    private_commit_average_mib = [math]::Round(
        ($taskSamples | Measure-Object PrivateMiB -Average).Average, 1)
    note = 'CPU is normalized over all logical processors. Resident RAM may include shared pages. Speech can create brief CPU/RAM increases.'
}
$taskReport | ConvertTo-Json
