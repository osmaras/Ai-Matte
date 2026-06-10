@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
REM ── Machine-specific paths (edit for your system) ──
set "PROJECT_DIR=C:\PATH\TO\AI_matte"
set "UV_EXE=C:\Users\USERNAME\.local\bin\uv.exe"

REM ── Cache location (drive with ~15 GB free space) ──
set "UV_CACHE_BASE=%PROJECT_DIR%"
REM if exist "X:\" set "UV_CACHE_BASE=X:\Scratch-AI-matte-cache"

if not exist "%UV_CACHE_BASE%\.uv-cache" mkdir "%UV_CACHE_BASE%\.uv-cache"
if not exist "%UV_CACHE_BASE%\.uv-tmp" mkdir "%UV_CACHE_BASE%\.uv-tmp"
set "UV_CACHE_DIR=%UV_CACHE_BASE%\.uv-cache"
set "TMP=%UV_CACHE_BASE%\.uv-tmp"
set "TEMP=%UV_CACHE_BASE%\.uv-tmp"
set "UV_LINK_MODE=copy"

%PROJECT_DIR:~0,2%
cd "%PROJECT_DIR%"

"%UV_EXE%" run ^
	--verbose ^
	--with torch==2.5.1+cu124 ^
	--with torchvision==0.20.1+cu124 ^
	--default-index https://pypi.org/simple ^
	--index https://download.pytorch.org/whl/cu124 ^
	--index-strategy unsafe-best-match ^
	scratch_matanyone_bridge.py %*
pause
