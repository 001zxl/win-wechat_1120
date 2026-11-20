@echo off
chcp 65001 >nul
cd /d "%~dp0"
title WeChat AI Console - stop
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" launcher.py --stop %*
echo.
pause >nul
