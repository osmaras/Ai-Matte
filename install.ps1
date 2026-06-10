param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Windows.Forms

function Select-Folder {
    param(
        [string]$Description,
        [string]$InitialPath
    )

    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $dialog.Description = $Description
    if ($InitialPath -and (Test-Path $InitialPath)) {
        $dialog.SelectedPath = $InitialPath
    }

    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        return $dialog.SelectedPath
    }

    return $null
}

function Select-File {
    param(
        [string]$Title,
        [string]$Filter,
        [string]$InitialDirectory
    )

    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = $Title
    $dialog.Filter = $Filter
    if ($InitialDirectory -and (Test-Path $InitialDirectory)) {
        $dialog.InitialDirectory = $InitialDirectory
    }

    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        return $dialog.FileName
    }

    return $null
}

function Ask-YesNo {
    param(
        [string]$Prompt,
        [bool]$Default = $true
    )

    $suffix = if ($Default) { "[Y/n]" } else { "[y/N]" }
    $raw = Read-Host "$Prompt $suffix"
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return $Default
    }

    return $raw.Trim().ToLowerInvariant().StartsWith("y")
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$defaultProjectDir = $scriptDir
$defaultUvPath = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
$defaultCacheRoot = Join-Path $scriptDir "runtime-cache"

Write-Host "MatAnyone SCRATCH Installer" -ForegroundColor Cyan
Write-Host "This will configure launcher.bat for this machine." -ForegroundColor Cyan
Write-Host ""

$projectDir = $defaultProjectDir
if (-not (Ask-YesNo "Use this project folder? $defaultProjectDir" $true)) {
    $selected = Select-Folder -Description "Select AI_matte project folder" -InitialPath $defaultProjectDir
    if (-not $selected) {
        throw "Project folder selection was canceled."
    }
    $projectDir = $selected
}

$bridgeScript = Join-Path $projectDir "scratch_matanyone_bridge.py"
if (-not (Test-Path $bridgeScript)) {
    throw "Could not find scratch_matanyone_bridge.py in: $projectDir"
}

$uvPath = $defaultUvPath
if (-not (Test-Path $uvPath)) {
    Write-Host "Default uv path not found: $uvPath" -ForegroundColor Yellow
}
if (-not (Ask-YesNo "Use uv executable at: $uvPath" (Test-Path $uvPath))) {
    $selectedUv = Select-File -Title "Select uv.exe" -Filter "uv executable (uv.exe)|uv.exe|Executable (*.exe)|*.exe|All files (*.*)|*.*" -InitialDirectory (Split-Path $defaultUvPath -Parent)
    if (-not $selectedUv) {
        throw "uv executable selection was canceled."
    }
    $uvPath = $selectedUv
}
if (-not (Test-Path $uvPath)) {
    throw "uv executable not found: $uvPath"
}

$cacheRoot = $defaultCacheRoot
if (-not (Ask-YesNo "Use cache folder at: $cacheRoot" $true)) {
    $selectedCache = Select-Folder -Description "Select folder for uv cache/temp" -InitialPath (Split-Path $defaultCacheRoot -Parent)
    if (-not $selectedCache) {
        throw "Cache folder selection was canceled."
    }
    $cacheRoot = $selectedCache
}

$requireCuda = Ask-YesNo "Require CUDA (fail fast if GPU is unavailable)?" $true

$cacheDir = Join-Path $cacheRoot ".uv-cache"
$tmpDir = Join-Path $cacheRoot ".uv-tmp"

$launcherPath = Join-Path $projectDir "launcher.bat"
$requireCudaArg = if ($requireCuda) { " --require-cuda" } else { "" }

$launcherContent = @"
@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
setlocal
cd /d "$projectDir"

if not exist "$cacheDir" mkdir "$cacheDir"
if not exist "$tmpDir" mkdir "$tmpDir"
set "UV_CACHE_DIR=$cacheDir"
set "TMP=$tmpDir"
set "TEMP=$tmpDir"

"$uvPath" run ^
    --verbose ^
    --with torch==2.5.1+cu124 ^
    --with torchvision==0.20.1+cu124 ^
    --default-index https://pypi.org/simple ^
    --index https://download.pytorch.org/whl/cu124 ^
    --index-strategy unsafe-best-match ^
    scratch_matanyone_bridge.py$requireCudaArg %%*
pause
"@

Set-Content -Path $launcherPath -Value $launcherContent -Encoding ascii

Write-Host ""
Write-Host "Configured launcher:" -ForegroundColor Green
Write-Host "  $launcherPath"
Write-Host ""
Write-Host "SCRATCH Custom Command settings:" -ForegroundColor Green
Write-Host "  Command  : $launcherPath"
Write-Host "  Arguments: -project %PRJ -group %GRP -construct %CON -shot %SHT"
Write-Host ""
Write-Host "Done." -ForegroundColor Green
