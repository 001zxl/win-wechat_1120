@echo off
chcp 65001 >nul
cd /d %~dp0
echo ============================================
echo  微信 AI 回复 · 一键安装（Windows）
echo ============================================
echo.
where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 没找到 python，请先安装 Python 3.10+ 并勾选 Add to PATH
  pause
  exit /b 1
)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo === 环境自检 ===
python run.py doctor
echo.
echo 安装完成：双击 start.bat 启动控制台，或运行 python server.py
pause
