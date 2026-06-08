@echo off
pushd "%~dp0"

set "UV_CACHE_BASE=%~dp0"
if exist "G:\" set "UV_CACHE_BASE=G:\Scratch-AI-matte-cache"

if not exist "%UV_CACHE_BASE%\.uv-cache" mkdir "%UV_CACHE_BASE%\.uv-cache"
if not exist "%UV_CACHE_BASE%\.uv-tmp" mkdir "%UV_CACHE_BASE%\.uv-tmp"
set "UV_CACHE_DIR=%UV_CACHE_BASE%\.uv-cache"
set "TMP=%UV_CACHE_BASE%\.uv-tmp"
set "TEMP=%UV_CACHE_BASE%\.uv-tmp"

"C:\Users\oscar\.local\bin\uv.exe" run ^
	--with torch==2.5.1+cu124 ^
	--with torchvision==0.20.1+cu124 ^
	--default-index https://pypi.org/simple ^
	--index https://download.pytorch.org/whl/cu124 ^
	--index-strategy unsafe-best-match ^
	scratch_matanyone_bridge.py %*
pause