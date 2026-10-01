<#
在笔记本上一键跑 ASUS 采集机上的 tracker，停止后自动把本场数据整目录拉回本机 tracker_output\。

等价于手动：
    ssh asus
    (.venv_ros2) PS D:\robot\Ball_Tracer_PC> .\run_tracker_with_predict.ps1 -Car v05 -Duration 500 -ExposureUs 2000

用法（除下面几个本脚本自己的参数外，其余原样转给 ASUS 上的 run_tracker_with_predict.ps1；
-Car 必填、没有默认值，理由同 run_tracker.ps1）：
    .\run_tracker_asus.ps1 -Car v05 -Duration 500 -ExposureUs 2000
    .\run_tracker_asus.ps1 -PullOnly                                    只拉 ASUS 上最新一场
    .\run_tracker_asus.ps1 -PullOnly -Session tracker_20261001_234800   拉指定场（拉取中断后续传也用它）
    -AsusHost asus@10.50.65.130   不在 ASUS 局域网时走 ZeroTier（见 tennis-man memory/asus-ssh-access.md）

流程：
  1. 记下 ASUS tracker_output\ 里已有的场次；ASUS 上已有 tracker 在跑就不起第二个。
  2. ssh -t 进 ASUS：cd 到仓库、激活 .venv_ros2、跑 run_tracker_with_predict.ps1。本地终端此时就是
     远端伪终端，Ctrl+C 原样送过去：run_tracker.py 收到 SIGINT 照常收尾（写 JSON、关 bag、后处理出
     HTML 等），收尾完远端 shell 才退、ssh 才返回（2026-10-01 用模拟 tracker 实测）。
     只按一次——收尾阶段再按会把后处理子进程一起打断。跑满 -Duration 自己停也一样。
  3. 前后清单一比，新出现的 tracker_YYYYMMDD_HHMMSS\ 整个目录用 sftp 拉到本机：小文件先拉、mp4 最后；
     本地已齐的文件跳过、拉了一半的续传（get -a）；拉完逐个文件比字节数。

注意：
  - ASUS 上有 tracker 在跑时不拉（先等它退）：ASUS 走 Wi-Fi，几 GB 的拷贝会挤 /pc_car_loc。
    拉 mp4 时想马上开下一场：Ctrl+C 掉拉取，之后 -PullOnly -Session <场次> 续传。
  - ssh 中途断线会把远端 tracker 一起带走（和手动 ssh 一样），那一场数据不全。
  - 一场 500 s 约 7 GB（mp4 占大头）；1001 实测 ASUS→笔记本约 17 MB/s，拉完约 7 分钟。
  - 依赖 ~/.ssh/config 的 Host asus 免密登录；ASUS 换了 IP 改那里。
#>
param(
    [switch]$PullOnly,
    [string[]]$Session = @(),
    [string]$AsusHost = 'asus',
    [string]$RemoteRepo = 'D:\robot\Ball_Tracer_PC',
    [string]$LocalRoot = ''
)

$ErrorActionPreference = 'Stop'
$trackerArgs = @($args)
if ([string]::IsNullOrWhiteSpace($LocalRoot)) {
    $LocalRoot = Join-Path $PSScriptRoot 'tracker_output'
}
$LocalRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($LocalRoot)
# 远端路径只做字符串拼接：本机 Join-Path 遇到本机没有的盘符会直接报错
$remoteOutput = $RemoteRepo.TrimEnd('\') + '\tracker_output'
# Windows 版 sftp-server 的路径写法：/D:/robot/...
$sftpOutput = '/' + ($remoteOutput -replace '\\', '/')
$sessionPattern = '^tracker_\d{8}_\d{6}$'
$sshExe = (Get-Command ssh.exe -CommandType Application | Select-Object -First 1).Source
$sftpExe = (Get-Command sftp.exe -CommandType Application | Select-Object -First 1).Source

function ConvertTo-PsLiteral([string]$Text) {
    return "'" + $Text.Replace("'", "''") + "'"
}

# 把本地收到的参数还原成远端命令行：$args 里参数名是 '-Xxx' 字符串，值保留原类型
function ConvertTo-PsArgText([object[]]$Items) {
    $tokens = New-Object System.Collections.Generic.List[string]
    foreach ($a in $Items) {
        if ($a -is [bool]) {
            $tok = if ($a) { '$true' } else { '$false' }
        } elseif ($a -is [string] -and $a -match '^-[A-Za-z_]\w*:?$') {
            $tok = $a
        } elseif ($a -is [string]) {
            $tok = ConvertTo-PsLiteral $a
        } elseif ($a -is [ValueType]) {
            $tok = [Convert]::ToString($a, [Globalization.CultureInfo]::InvariantCulture)
        } else {
            throw "Unsupported tracker argument '$a' ($($a.GetType().FullName))"
        }
        # -NoVideo:$false 在 $args 里被拆成 '-NoVideo:' 和 $false 两项，拼回一个
        if ($tokens.Count -gt 0 -and $tokens[$tokens.Count - 1].EndsWith(':')) {
            $tokens[$tokens.Count - 1] = $tokens[$tokens.Count - 1] + $tok
        } else {
            $tokens.Add($tok)
        }
    }
    return ($tokens -join ' ')
}

# 子进程直接继承本机控制台、不经 PowerShell 管道：ssh -t 才拿得到真终端（Ctrl+C 才送得到远端），
# sftp 才显示进度条
function Invoke-Console([string]$Exe, [string[]]$ArgList) {
    $quoted = foreach ($a in $ArgList) {
        if ($a -match '[\s"]') { '"' + $a.Replace('"', '\"') + '"' } else { $a }
    }
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Exe
    $psi.Arguments = ($quoted -join ' ')
    $psi.UseShellExecute = $false
    $proc = [System.Diagnostics.Process]::Start($psi)
    $proc.WaitForExit()
    return $proc.ExitCode
}

# 在 ASUS 上跑一段 PowerShell，取回 stdout 各行。脚本体走 -EncodedCommand，免掉多层转义；
# 远端出错只经 'ERR|' 行报回（不让 PowerShell 往 stderr 吐 CLIXML），输出统一 UTF-8
function Invoke-AsusPs([string]$Body) {
    $script = @(
        '$ErrorActionPreference = ''Stop'''
        '$ProgressPreference = ''SilentlyContinue'''
        'try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false } catch { }'
        'try {'
        $Body
        '} catch { ''ERR|'' + $_.Exception.Message; exit 3 }'
    ) -join "`n"
    $b64 = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($script))
    $prevEncoding = [Console]::OutputEncoding
    try {
        try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false } catch { }
        $lines = @(& $sshExe -o BatchMode=yes -o ConnectTimeout=10 -o LogLevel=ERROR $AsusHost "powershell -NoProfile -NonInteractive -EncodedCommand $b64")
        $rc = $LASTEXITCODE
    } finally {
        try { [Console]::OutputEncoding = $prevEncoding } catch { }
    }
    $err = @($lines | Where-Object { $_ -like 'ERR|*' })
    if ($err.Count -gt 0) { throw ('ASUS: ' + $err[0].Substring(4)) }
    if ($rc -ne 0) { throw "ssh $AsusHost failed (exit $rc)" }
    return $lines
}

# ASUS 上已有的场次 + 正在跑的 tracker 进程
function Get-AsusState {
    $body = @'
$out = __OUT__
if (Test-Path -LiteralPath $out) {
    Get-ChildItem -LiteralPath $out -Directory |
        Where-Object { $_.Name -match '^tracker_\d{8}_\d{6}$' } |
        ForEach-Object { 'S|' + $_.Name }
}
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -like '*run_tracker.py*' } |
    ForEach-Object { 'P|' + $_.ProcessId }
'@
    $lines = @(Invoke-AsusPs ($body.Replace('__OUT__', (ConvertTo-PsLiteral $remoteOutput))))
    return [pscustomobject]@{
        Sessions = @($lines | Where-Object { $_ -like 'S|*' } | ForEach-Object { $_.Substring(2) } | Sort-Object)
        Pids     = @($lines | Where-Object { $_ -like 'P|*' } | ForEach-Object { $_.Substring(2) })
    }
}

# ASUS 上有 tracker 在跑就等它退：ASUS 走 Wi-Fi，几 GB 的拷贝会挤 /pc_car_loc
function Wait-AsusIdle {
    $state = Get-AsusState
    if ($state.Pids.Count -gt 0) {
        Write-Host ("A tracker is running on ASUS (pid {0}); waiting for it to exit before pulling, so the copy does not crowd /pc_car_loc on Wi-Fi. Ctrl+C to give up." -f ($state.Pids -join ', ')) -ForegroundColor Yellow
        while ($state.Pids.Count -gt 0) {
            Start-Sleep -Seconds 10
            $state = Get-AsusState
        }
    }
    return $state
}

function Get-AsusSessionFiles([string]$Name) {
    $body = @'
$dir = __DIR__
if (-not (Test-Path -LiteralPath $dir -PathType Container)) { 'ERR|no such session: ' + $dir; exit 3 }
$root = (Get-Item -LiteralPath $dir).FullName.TrimEnd('\')
Get-ChildItem -LiteralPath $root -Recurse -File -Force |
    ForEach-Object { 'F|' + $_.Length + '|' + $_.LastWriteTimeUtc.Ticks + '|' + $_.FullName.Substring($root.Length + 1) }
'@
    $lines = @(Invoke-AsusPs ($body.Replace('__DIR__', (ConvertTo-PsLiteral ($remoteOutput + '\' + $Name)))))
    foreach ($line in $lines) {
        if ($line -match '^F\|(\d+)\|(\d+)\|(.+)$') {
            [pscustomobject]@{ Length = [int64]$Matches[1]; MtimeUtc = [datetime]::new([int64]$Matches[2], 'Utc'); Rel = $Matches[3] }
        }
    }
}

# 把 ASUS 上一场整目录拉到 $LocalRoot\<场次>\；返回是否逐文件字节数对齐
function Sync-AsusSession([string]$Name) {
    $files = @(Get-AsusSessionFiles $Name | Sort-Object Length)
    $localDir = Join-Path $LocalRoot $Name
    if ($files.Count -eq 0) {
        Write-Warning "$Name has no files on ASUS; nothing to pull."
        return $false
    }
    $totalBytes = [int64]($files | Measure-Object Length -Sum).Sum
    $todo = @()
    $need = [int64]0
    foreach ($f in $files) {
        $local = Join-Path $localDir $f.Rel
        $have = [int64]-1
        $remoteNewer = $false
        if (Test-Path -LiteralPath $local -PathType Leaf) {
            $item = Get-Item -LiteralPath $local
            $have = $item.Length
            # 本地副本落盘之后远端又改过（如收尾报告重生成）= 内容已不同，不能当半截续传
            $remoteNewer = $f.MtimeUtc -gt $item.LastWriteTimeUtc.AddSeconds(2)
        }
        if ($have -eq $f.Length -and -not $remoteNewer) { continue }
        # 本地比远端短且是远端定稿之后写的 = 上次拉了一半，续传；其余一律从头拉
        $resume = ($have -ge 0 -and $have -lt $f.Length -and -not $remoteNewer)
        $need += $(if ($resume) { $f.Length - $have } else { $f.Length })
        $todo += [pscustomobject]@{ Rel = $f.Rel; Local = $local; Resume = $resume }
    }
    if ($todo.Count -eq 0) {
        Write-Host ("{0}: already complete locally ({1} files, {2:N2} GB) -> {3}" -f $Name, $files.Count, ($totalBytes / 1GB), $localDir) -ForegroundColor Green
        return $true
    }

    $drive = New-Object System.IO.DriveInfo ([System.IO.Path]::GetPathRoot($localDir))
    if ($drive.AvailableFreeSpace -lt $need + 2GB) {
        throw ("Not enough space on {0}: need {1:N1} GB (+2 GB margin), free {2:N1} GB" -f $drive.Name, ($need / 1GB), ($drive.AvailableFreeSpace / 1GB))
    }
    foreach ($t in $todo) { $null = New-Item -ItemType Directory -Force -Path (Split-Path -Parent $t.Local) }

    # sftp 批处理：'@' 不回显命令；不加 '-' 前缀，任一文件失败即中止，后面按字节数核对
    $batchLines = @('@progress')
    foreach ($t in $todo) {
        $cmd = if ($t.Resume) { '@get -a' } else { '@get' }
        $batchLines += ('{0} "{1}/{2}/{3}" "{4}"' -f $cmd, $sftpOutput, $Name, ($t.Rel -replace '\\', '/'), ($t.Local -replace '\\', '/'))
    }
    $batch = Join-Path ([System.IO.Path]::GetTempPath()) ("run_tracker_asus_{0}.sftp" -f $Name)
    # 只用 LF（CRLF 的 \r 会被 sftp 当成路径的一部分）、无 BOM
    [System.IO.File]::WriteAllText($batch, ($batchLines -join "`n") + "`n", (New-Object System.Text.UTF8Encoding $false))

    Write-Host ''
    Write-Host ("Pulling {0}: {1} of {2} files, {3:N2} GB -> {4}" -f $Name, $todo.Count, $files.Count, ($need / 1GB), $localDir) -ForegroundColor Cyan
    Write-Host ("  (if interrupted, resume with: {0} -PullOnly -Session {1})" -f $PSCommandPath, $Name)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $rc = Invoke-Console $sftpExe @('-o', 'ConnectTimeout=10', '-o', 'LogLevel=ERROR', '-b', $batch, $AsusHost)
    } finally {
        Remove-Item -LiteralPath $batch -Force -ErrorAction SilentlyContinue
    }
    $sw.Stop()

    $short = @($files | Where-Object {
        $p = Join-Path $localDir $_.Rel
        -not (Test-Path -LiteralPath $p -PathType Leaf) -or (Get-Item -LiteralPath $p).Length -ne $_.Length
    })
    if ($short.Count -gt 0) {
        Write-Host ("INCOMPLETE {0}: {1} file(s) missing or short after sftp (exit {2}):" -f $Name, $short.Count, $rc) -ForegroundColor Red
        foreach ($s in $short) { Write-Host ("  {0}" -f $s.Rel) }
        Write-Host ("Resume with: {0} -PullOnly -Session {1}" -f $PSCommandPath, $Name)
        return $false
    }
    $secs = [Math]::Max($sw.Elapsed.TotalSeconds, 0.001)
    Write-Host ("OK {0}: {1} files, {2:N2} GB; pulled {3:N2} GB in {4:N0} s ({5:N1} MB/s) -> {6}" -f $Name, $files.Count, ($totalBytes / 1GB), ($need / 1GB), $secs, ($need / 1MB / $secs), $localDir) -ForegroundColor Green
    $html = Join-Path $localDir ($Name + '.html')
    if (Test-Path -LiteralPath $html) { Write-Host "Report: $html" }
    return $true
}

$null = New-Item -ItemType Directory -Force -Path $LocalRoot

if ($PullOnly) {
    if ($trackerArgs.Count -gt 0) { throw "-PullOnly does not run the tracker; unexpected arguments: $trackerArgs" }
    foreach ($n in $Session) {
        if ($n -notmatch $sessionPattern) { throw "Bad -Session '$n' (expected tracker_YYYYMMDD_HHMMSS)" }
    }
    $state = Wait-AsusIdle
    $names = @($Session)
    if ($names.Count -eq 0) {
        if ($state.Sessions.Count -eq 0) { throw "No tracker_* session under $remoteOutput on ASUS" }
        $names = @($state.Sessions[-1])
    }
    $allOk = $true
    foreach ($n in $names) { if (-not (Sync-AsusSession $n)) { $allOk = $false } }
    if ($allOk) { exit 0 } else { exit 1 }
}

if (-not ($trackerArgs | Where-Object { $_ -is [string] -and $_ -match '^-Car:?$' })) {
    throw 'Missing -Car v04|v05. There is deliberately no default car: a wrong AprilTag layout silently shifts car localization.'
}
$argText = ConvertTo-PsArgText $trackerArgs

$before = Get-AsusState
if ($before.Pids.Count -gt 0) {
    throw ("A tracker is already running on ASUS (pid {0}); not starting a second one." -f ($before.Pids -join ', '))
}

# ConPTY 一开场就清屏，本地先打的字会被冲掉，所以提示由远端在清屏之后打印
$banner = "[run_tracker_asus] {0}> .\run_tracker_with_predict.ps1 {1}`n[run_tracker_asus] Ctrl+C once to stop. After the tracker exits, the new session is pulled to {2}" -f $RemoteRepo, $argText, $LocalRoot
$runBody = @(
    'Set-Location -LiteralPath ' + (ConvertTo-PsLiteral $RemoteRepo) + ' -ErrorAction Stop'
    'Write-Host ' + (ConvertTo-PsLiteral $banner) + ' -ForegroundColor Cyan'
    "if (Test-Path -LiteralPath '.venv_ros2\Scripts\Activate.ps1') { . '.\.venv_ros2\Scripts\Activate.ps1' }"
    "& '.\run_tracker_with_predict.ps1' $argText"
    'exit $LASTEXITCODE'
) -join "`n"
$b64 = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($runBody))
$rc = Invoke-Console $sshExe @('-t', '-o', 'LogLevel=ERROR', $AsusHost, "powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand $b64")

Write-Host ''
Write-Host ("Tracker on ASUS exited (code {0})." -f $rc)
if ($rc -eq 255) {
    Write-Warning 'The ssh connection was lost; the tracker on ASUS most likely died with it, so this session is probably incomplete. If ASUS is unreachable now, pull later with -PullOnly.'
}
$after = Wait-AsusIdle
$new = @($after.Sessions | Where-Object { $before.Sessions -notcontains $_ })
if ($new.Count -eq 0) {
    Write-Host "No new session under $remoteOutput - nothing to pull."
    exit $rc
}
$allOk = $true
foreach ($n in $new) { if (-not (Sync-AsusSession $n)) { $allOk = $false } }
if ($allOk) { exit 0 } else { exit 1 }
