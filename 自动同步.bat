@echo off
chcp 65001 >nul
cd /d "%~dp0"
title GitHub Sync - watch and push on change
echo Watching for changes. Every edit is committed and pushed automatically.
echo Close this window to stop.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0同步到GitHub.ps1" -Watch %*
echo.
pause >nul
