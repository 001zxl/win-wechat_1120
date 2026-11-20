@echo off
chcp 65001 >nul
cd /d "%~dp0"
title GitHub Sync - push once
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0同步到GitHub.ps1" %*
echo.
echo Done. Press any key to close.
pause >nul
