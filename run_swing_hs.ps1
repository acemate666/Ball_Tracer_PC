# 本文件必须存成 UTF-8 带 BOM：ASUS 上只有 Windows PowerShell 5.1、ANSI 码页 936，无 BOM 就按 GBK 读（同 run_tracker.ps1）。
<#
100 fps 静态挥拍识别（swing_hs/run_swing_hs.py）：车不动、臂挥拍，4 台相机只开「黑标包络 + 两块 tag」的 ROI、
100 fps 同步触发，在线认拍上黑标 + 车位，全程录 rosbag。tracker 不能同时开（相机独占）。

用法（ASUS，仓库根）：
    .\run_swing_hs.ps1 -Car v05 -Side fore                 正手，录到 Ctrl+C（或建 tracker_output\.stop_swing_hs）
    .\run_swing_hs.ps1 -Car v05 -Side both -PlanOnly       只开相机定车位、出 ROI 规划和 roi_*.jpg
    .\run_swing_hs.ps1 -Car v05 -Side fore -Probe 20       自检 20 s
其余参数原样转给 run_swing_hs.py（如 '--j1-range' -60 45、'--car-pose' 0.1 2.5 0；双横线参数要加引号）。

发 /pc_car_loc（同 tracker 契约）+ /swing_hs/racket_marker + /swing_hs/status；挥拍时不起 bot_center（臂用面板 / swing_replay 挥）。
rosbag 录全部话题，不连球拍 IMU。
#>
param(
    [Parameter(Mandatory = $true, HelpMessage = "Pick the car: v04 or v05")]
    [ValidateSet('v04', 'v05')]
    [string]$Car,
    [Parameter(Mandatory = $true, HelpMessage = "Pick the side: fore, back, or both")]
    [ValidateSet('fore', 'back', 'both')]
    [string]$Side,
    [double]$Duration = 1800,
    [double]$Probe = 0,
    [switch]$PlanOnly,
    [double]$ExposureUs = 0,
    [switch]$NoPatches,
    [switch]$NoRos,
    [string]$RecordRaw = '',
    # 其余参数原样转给 run_swing_hs.py
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Extra = @()
)

$ErrorActionPreference = 'Stop'
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $utf8NoBom
[Console]::OutputEncoding = $utf8NoBom
$OutputEncoding = $utf8NoBom
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
try {
    & chcp.com 65001 > $null
} catch {
}

$python = Join-Path $PSScriptRoot '.venv_ros2\Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw "找不到 $python（相机 / 识别跑在 .venv_ros2 里；ROS 桥和 rosbag 走 ros2\run_ros2.bat 自带的 ROS python）"
}
if (-not (Test-Path Env:MVS_MVIMPORT_DIR)) {
    $env:MVS_MVIMPORT_DIR = 'C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport'
}
if (-not (Test-Path $env:MVS_MVIMPORT_DIR)) {
    throw "MVS MvImport directory not found: $($env:MVS_MVIMPORT_DIR)"
}

$pyArgs = @((Join-Path $PSScriptRoot 'swing_hs\run_swing_hs.py'), '--car', $Car, '--side', $Side, '--duration', $Duration)
if ($Probe -gt 0) { $pyArgs += @('--probe', $Probe) }
if ($PlanOnly) { $pyArgs += '--plan-only' }
if ($ExposureUs -gt 0) { $pyArgs += @('--exposure-us', $ExposureUs) }
if ($NoPatches) { $pyArgs += '--no-patches' }
if ($NoRos) { $pyArgs += '--no-ros' }
if (-not [string]::IsNullOrWhiteSpace($RecordRaw)) { $pyArgs += @('--record-raw', $RecordRaw) }
$pyArgs += $Extra

Push-Location $PSScriptRoot
try {
    & $python @pyArgs
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
