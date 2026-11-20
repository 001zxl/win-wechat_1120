<#
  同步到 GitHub（走 REST API，不用 git 协议：本机到 github.com 的 git 传输经常被重置）
  用法：
    一键同步.bat                  # 同步一次（无变化则什么都不做）
    自动同步.bat                  # 常驻监视，检测到改动后自动推送
    powershell -File 同步到GitHub.ps1 -InitToken    # 重新录入令牌
#>
[CmdletBinding()]
param(
    [switch]$Watch,
    [switch]$InitToken,
    [string]$Token,
    [int]$IntervalSec = 5,
    [string]$Source = "C:\Users\Administrator\Documents\WeChat_11",
    [string]$Owner = "001zxl",
    [string]$Repo = "win-wechat_1120",
    [string]$Branch = "main",
    [string]$TokenFile = "$env:USERPROFILE\.wechat1120.token"
)

$ErrorActionPreference = "Stop"
$LogFile = Join-Path $Source "data\sync.log"
$StateFile = Join-Path $Source "data\sync-state.json"
$ApiBase = "https://api.github.com/repos/$Owner/$Repo"
$script:Token = $null

function Write-Log([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    try {
        $dir = Split-Path $LogFile -Parent
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
        Add-Content -Path $LogFile -Value $line -Encoding UTF8
    } catch { }
}

function Get-PlainToken {
    if ($Token) { return $Token }
    if (Test-Path $TokenFile) {
        try {
            $secure = Get-Content $TokenFile | ConvertTo-SecureString
            $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
            try { return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) }
            finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
        } catch {
            Write-Log "令牌文件无法解密，需要重新录入"
        }
    }
    $secure = Read-Host "粘贴 GitHub 令牌（ghp_... 或 github_pat_...，输入不显示）" -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try { $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
    if (-not $plain) { throw "没有输入令牌" }
    Set-Token $plain
    return $plain
}

function Set-Token([string]$Plain) {
    $secure = ConvertTo-SecureString $Plain -AsPlainText -Force
    $secure | ConvertFrom-SecureString | Set-Content -Path $TokenFile -Encoding ASCII
    Write-Log "令牌已加密保存（仅当前 Windows 用户可用）"
}

function Invoke-Api {
    param([string]$Method, [string]$Path, $Body = $null, [int]$TimeoutSec = 60)
    $headers = @{
        Authorization          = "Bearer $($script:Token)"
        "User-Agent"           = "wechat-sync"
        Accept                 = "application/vnd.github+json"
        "X-GitHub-Api-Version" = "2022-11-28"
    }
    $uri = "$ApiBase$Path"
    if ($null -ne $Body) {
        $json = $Body | ConvertTo-Json -Depth 20 -Compress
        return Invoke-RestMethod -Uri $uri -Method $Method -Headers $headers `
            -Body ([Text.Encoding]::UTF8.GetBytes($json)) -ContentType "application/json" -TimeoutSec $TimeoutSec
    }
    return Invoke-RestMethod -Uri $uri -Method $Method -Headers $headers -TimeoutSec $TimeoutSec
}

function Get-SyncFiles {
    $dirs = @(".venv", "data", "__pycache__", ".git", ".idea", "node_modules")
    $names = @("config.yaml", ".DS_Store")
    $exts = @(".log", ".db", ".sqlite", ".pyc", ".tmp", ".zip")
    $result = @{}
    Get-ChildItem -Path $Source -Recurse -File -Force -ErrorAction SilentlyContinue | ForEach-Object {
        $rel = $_.FullName.Substring($Source.Length + 1)
        $parts = $rel -split '\\'
        if ($parts | Where-Object { $dirs -contains $_ }) { return }
        if ($names -contains $_.Name) { return }
        if ($exts -contains $_.Extension.ToLower()) { return }
        if ($_.Name -like "capture*.png") { return }
        $rel = $rel -replace '\\', '/'
        $result[$rel] = $_.FullName
    }
    return $result
}

function Get-BlobSha([byte[]]$Bytes) {
    $header = [Text.Encoding]::ASCII.GetBytes("blob $($Bytes.Length)`0")
    $all = New-Object byte[] ($header.Length + $Bytes.Length)
    [Array]::Copy($header, 0, $all, 0, $header.Length)
    [Array]::Copy($Bytes, 0, $all, $header.Length, $Bytes.Length)
    $sha1 = [System.Security.Cryptography.SHA1]::Create()
    return (([BitConverter]::ToString($sha1.ComputeHash($all)) -replace '-', '')).ToLower()
}

function Get-LocalHash($Files) {
    $lines = @()
    foreach ($path in ($Files.Keys | Sort-Object)) {
        $info = Get-Item -LiteralPath $Files[$path]
        $lines += "$path|$($info.Length)|$($info.LastWriteTimeUtc.Ticks)"
    }
    $md5 = [System.Security.Cryptography.MD5]::Create()
    $bytes = [Text.Encoding]::UTF8.GetBytes([string]::Join("`n", $lines))
    return (([BitConverter]::ToString($md5.ComputeHash($bytes)) -replace '-', '')).ToLower()
}

function Get-RemoteState {
    try {
        $ref = Invoke-Api -Method GET -Path "/git/ref/heads/$Branch"
        $headSha = $ref.object.sha
        $commit = Invoke-Api -Method GET -Path "/git/commits/$headSha"
        $tree = Invoke-Api -Method GET -Path "/git/trees/$($commit.tree.sha)?recursive=1"
        $map = @{}
        foreach ($item in $tree.tree) {
            if ($item.type -eq "blob") { $map[$item.path] = $item.sha }
        }
        return @{ head = $headSha; tree = $commit.tree.sha; files = $map }
    } catch {
        if ($_.Exception.Response.StatusCode.value__ -eq 404 -or $_.Exception.Message -match "404|Not Found") {
            return @{ head = $null; tree = $null; files = @{} }   # 空仓库
        }
        throw
    }
}

function Invoke-Sync {
    $files = Get-SyncFiles
    $remote = Get-RemoteState
    $uploaded = 0
    $entries = New-Object System.Collections.ArrayList

    foreach ($path in ($files.Keys | Sort-Object)) {
        $bytes = [System.IO.File]::ReadAllBytes($files[$path])
        $sha = Get-BlobSha $bytes
        if ($remote.files[$path] -eq $sha) { continue }
        $body = @{ content = [Convert]::ToBase64String($bytes); encoding = "base64" }
        $blob = Invoke-Api -Method POST -Path "/git/blobs" -Body $body
        [void]$entries.Add(@{ path = $path; mode = "100644"; type = "blob"; sha = $blob.sha })
        $uploaded++
    }

    foreach ($path in $remote.files.Keys) {
        if (-not $files.ContainsKey($path)) {
            [void]$entries.Add(@{ path = $path; mode = "100644"; type = "blob"; sha = $null })
        }
    }

    if ($entries.Count -eq 0) {
        Write-Log "没有改动，跳过"
        return $null
    }

    $treeBody = @{ tree = $entries }
    if ($remote.tree) { $treeBody["base_tree"] = $remote.tree }
    $newTree = Invoke-Api -Method POST -Path "/git/trees" -Body $treeBody

    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $commitBody = @{
        message = "sync: $stamp ($uploaded files)"
        tree    = $newTree.sha
        parents = @()
    }
    if ($remote.head) { $commitBody.parents = @($remote.head) }
    $commit = Invoke-Api -Method POST -Path "/git/commits" -Body $commitBody

    if ($remote.head) {
        Invoke-Api -Method PATCH -Path "/git/refs/heads/$Branch" -Body @{ sha = $commit.sha; force = $false } | Out-Null
    } else {
        Invoke-Api -Method POST -Path "/git/refs" -Body @{ ref = "refs/heads/$Branch"; sha = $commit.sha } | Out-Null
    }
    Write-Log "已推送：$($commit.sha.Substring(0,7))（$uploaded 个文件改动）"
    return $commit.sha
}

function Save-State([string]$Hash, [string]$Commit) {
    try {
        $dir = Split-Path $StateFile -Parent
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
        @{ hash = $Hash; commit = $Commit; ts = (Get-Date -Format "o") } |
            ConvertTo-Json | Set-Content -Path $StateFile -Encoding UTF8
    } catch { }
}

function Get-SavedHash {
    try {
        if (Test-Path $StateFile) {
            $data = Get-Content $StateFile -Raw -Encoding UTF8 | ConvertFrom-Json
            return $data.hash
        }
    } catch { }
    return ""
}

# ---------------- 入口 ----------------
$script:Token = Get-PlainToken
if ($InitToken) {
    Set-Token $script:Token
    if (-not $Watch) { exit 0 }
}

if ($Watch) {
    Write-Log "监视模式启动：每 $IntervalSec 秒检查 $Source（改动后自动推送）"
    $last = Get-SavedHash
    if (-not $last) { $last = Get-LocalHash (Get-SyncFiles) }
    $backoff = 0
    while ($true) {
        Start-Sleep -Seconds $IntervalSec
        try {
            $files = Get-SyncFiles
            $now = Get-LocalHash $files
            if ($now -eq $last) { continue }
            Start-Sleep -Seconds 3
            $files = Get-SyncFiles
            $now = Get-LocalHash $files
            if ($now -eq $last) { continue }
            $sha = Invoke-Sync
            Save-State $now $sha
            $last = $now
            $backoff = 0
        } catch {
            Write-Log ("同步失败：" + $_.Exception.Message)
            $backoff = [Math]::Min(120, [Math]::Max(15, $backoff * 2))
            Start-Sleep -Seconds $backoff
        }
    }
} else {
    try {
        $sha = Invoke-Sync
        Save-State (Get-LocalHash (Get-SyncFiles)) $sha
        exit 0
    } catch {
        Write-Log ("失败：" + $_.Exception.Message)
        exit 1
    }
}
