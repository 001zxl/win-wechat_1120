@echo off
chcp 65001 >nul
cd /d "%~dp0"
title WeChat AI Console - auto reply
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo ============================================================
echo  Auto-reply mode: real WeChat + REAL SENDING enabled
echo  Only contacts in the whitelist will be answered.
echo  Press Ctrl+C stop, or just close this window.
echo ============================================================
echo.
"%PY%" server.py --real --live --no-browser --port 8765
echo.
echo Console exited. Press any key to close.
pause >nul
