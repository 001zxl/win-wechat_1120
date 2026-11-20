@echo off
chcp 65001 >nul
cd /d "%~dp0"
title WeChat AI Console - real WeChat
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo Connecting to the real WeChat window (dry-run: drafts only, no auto-send)...
"%PY%" launcher.py --real %*
echo.
pause >nul
