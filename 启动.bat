@echo off
chcp 65001 >nul
cd /d "%~dp0"
title WeChat AI Console
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo ============================================
echo  WeChat AI auto-reply console
echo  Web UI : http://127.0.0.1:8765/
echo  Real mode needs WeChat PC logged in with its main window open.
echo ============================================
echo.
"%PY%" launcher.py %*
echo.
echo Console exited. Press any key to close.
pause >nul
