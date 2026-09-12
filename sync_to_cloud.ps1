<#
  把 tracker_output 下的 session 目录同步到阿里云数据盘 (/data/tracker_output)。
  只上传、从不删除本地或远端。已完整上传的 session 会跳过,可反复运行。

  用法:  .\sync_to_cloud.ps1              上传所有缺失的 session
         .\sync_to_cloud.ps1 -Latest 3    只传最近 3 个
         .\sync_to_cloud.ps1 -WhatIf      只看要传什么,不实际传

  实现上的坑(改之前先看):
    ssh.exe / scp.exe 在没有控制台的脚本里会继承一个永不关闭的 stdin 句柄而挂死,
    加 -n 反而更糟(Windows 版 OpenSSH 的 -n 有 bug)。唯一可靠的办法是用
    Start-Process 把 stdin 显式指向一个空文件。别改回 "& ssh.exe ..." 的写法。
#>
param(
    [int]$Latest = 0,
    [switch]$WhatIf
)
$ErrorActionPreference = 'Stop'
trap { Write-Output ('未捕获异常: ' + $_.Exception.Message + ' @ ' + $_.InvocationInfo.ScriptLineNumber); exit 9 }
$LocalRoot  = 'D:\robot\Ball_Tracer_PC\tracker_output'
$RemoteRoot = '/data/tracker_output'
$Target     = 'root@47.119.152.170'
# 计划任务以 SYSTEM 身份运行,所以: (1) 用绝对路径, (2) 密钥必须放在 ACL 只给 SYSTEM/
# Administrators 的位置 —— 放在用户家目录下 ssh 会判定 'permissions too open' 而拒绝使用。
$KeyFile    = 'C:\ProgramData\ballnas\id_ed25519'
$SshOpts    = '-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30'

$NullIn = Join-Path $env:TEMP 'ssh_null_stdin.txt'
if (-not (Test-Path $NullIn)) { New-Item -ItemType File -Path $NullIn -Force | Out-Null }

function Invoke-Native([string]$Exe, [string]$ArgLine) {
    $o = [IO.Path]::GetTempFileName()
    $e = [IO.Path]::GetTempFileName()
    try {
        $p = Start-Process -FilePath $Exe -ArgumentList $ArgLine -NoNewWindow -Wait -PassThru `
             -RedirectStandardInput $NullIn -RedirectStandardOutput $o -RedirectStandardError $e
        return [pscustomobject]@{
            ExitCode = $p.ExitCode
            Out      = @(Get-Content $o -EA SilentlyContinue)
            Err      = @(Get-Content $e -EA SilentlyContinue)
        }
    } finally { Remove-Item $o, $e -Force -EA SilentlyContinue }
}

function Invoke-Remote([string]$Command) {
    Invoke-Native 'ssh.exe' ('-i "' + $KeyFile + '" ' + $SshOpts + ' ' + $Target + ' "' + $Command + '"')
}

function Get-RemoteSizes {
    # 远端命令里不要出现双引号或管道: 经 ssh 传递后会被远端 shell 重新解析。
    # 直接用 du 原生的 "字节数<TAB>路径" 输出,在这边解析。
    $r = Invoke-Remote ('mkdir -p ' + $RemoteRoot + '; du -sb ' + $RemoteRoot + '/*/ 2>/dev/null')
    # 必须用 Write-Warning 而不是 Write-Output: 函数里任何 Write-Output 都会混进返回值,
    # 让调用方拿到 [字符串, 哈希表] 数组而不是哈希表。
    if ($r.ExitCode -ne 0 -and $r.Err) { Write-Warning ("读取远端清单失败(exit " + $r.ExitCode + "): " + ($r.Err -join ' ')) }
    $map = @{}
    foreach ($line in $r.Out) {
        if ("$line" -match '^(\d+)\s+.*/([^/]+)/?\s*$') { $map[$matches[2]] = [int64]$matches[1] }
    }
    return $map
}

$dirs = Get-ChildItem $LocalRoot -Directory | Sort-Object Name -Descending
if ($Latest -gt 0) { $dirs = $dirs | Select-Object -First $Latest }
Write-Output "本地 session: $($dirs.Count) 个"

$remote = Get-RemoteSizes
Write-Output "远端已有: $($remote.Count) 个"

$todo = @()
foreach ($d in $dirs) {
    try {
        $sum = (Get-ChildItem $d.FullName -Recurse -File -EA SilentlyContinue | Measure-Object Length -Sum).Sum
        $localSize = if ($null -eq $sum) { [int64]0 } else { [int64]$sum }
    } catch {
        Write-Output ("跳过 {0}: 无法统计大小 - {1}" -f $d.Name, $_.Exception.Message); continue
    }
    if ($localSize -eq 0) { Write-Output ("跳过 {0}: 空目录" -f $d.Name); continue }
    if ($remote.ContainsKey($d.Name) -and $remote[$d.Name] -ge $localSize) { continue }
    $todo += [pscustomobject]@{ Name = $d.Name; Path = $d.FullName; Size = $localSize }
}
if ($todo.Count -eq 0) { Write-Output "全部已是最新,无需上传。"; exit 0 }

$totalGB = ($todo | Measure-Object Size -Sum).Sum / 1GB
Write-Output ("待上传: {0} 个 session, 共 {1:N2} GB" -f $todo.Count, $totalGB)
if ($WhatIf) {
    $todo | ForEach-Object { Write-Output ("   {0}  {1,8:N0} MB" -f $_.Name, ($_.Size/1MB)) }
    Write-Output "(WhatIf: 未实际上传)"
    exit 0
}

$i = 0; $ok = 0; $failed = @()
$swAll = [Diagnostics.Stopwatch]::StartNew()
foreach ($t in $todo) {
    $i++
    # 先传到 .partial_ 再远端改名: 中断后不会留下半个 session 被当成完整的
    $stage = $RemoteRoot + '/.partial_' + $t.Name
    Invoke-Remote ('rm -rf ' + $stage) | Out-Null
    $sw = [Diagnostics.Stopwatch]::StartNew()
    # 大文件传输常见 Broken pipe(瞬时断流),重试 3 次即可,不必人工补
    $r = $null
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        $r = Invoke-Native 'scp.exe' ('-i "' + $KeyFile + '" ' + $SshOpts + ' -r -q "' + $t.Path + '" ' + $Target + ':' + $stage)
        if ($r.ExitCode -eq 0) { break }
        Write-Output ("      第 {0} 次尝试失败 (exit {1}): {2}" -f $attempt, $r.ExitCode, ($r.Err -join ' '))
        Invoke-Remote ('rm -rf ' + $stage) | Out-Null
        Start-Sleep -Seconds (5 * $attempt)
    }
    $sw.Stop()
    if ($r.ExitCode -eq 0) {
        Invoke-Remote ('rm -rf ' + $RemoteRoot + '/' + $t.Name + '; mv ' + $stage + ' ' + $RemoteRoot + '/' + $t.Name) | Out-Null
        $ok++
        Write-Output ("[{0}/{1}] {2}  {3,7:N0} MB  {4,5:N0}s  {5,5:N1} MB/s  OK" -f `
            $i, $todo.Count, $t.Name, ($t.Size/1MB), $sw.Elapsed.TotalSeconds, (($t.Size/1MB)/[Math]::Max($sw.Elapsed.TotalSeconds,0.001)))
    } else {
        Invoke-Remote ('rm -rf ' + $stage) | Out-Null
        $failed += $t.Name
        Write-Output ("[{0}/{1}] {2}  失败 exit={3}  {4}" -f $i, $todo.Count, $t.Name, $r.ExitCode, ($r.Err -join ' '))
    }
}
$swAll.Stop()
Write-Output ("完成 {0}/{1},耗时 {2:N1} 分钟" -f $ok, $todo.Count, $swAll.Elapsed.TotalMinutes)
if ($failed) { Write-Output ("失败: " + ($failed -join ', ')); exit 1 }
