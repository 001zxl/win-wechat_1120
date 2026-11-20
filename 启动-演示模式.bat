@echo off
chcp 65001 >nul
cd /d "%~dp0"
title WeChat AI Console - demo mode
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo Demo mode: virtual WeChat window, the real account is never touched.
"%PY%" launcher.py --demo %*
echo.
pause >nul
