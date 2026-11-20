@echo off
chcp 65001 >nul
cd /d %~dp0
echo 正在启动微信 AI 回复控制台...
start "" http://127.0.0.1:8765/
python server.py %*
pause
