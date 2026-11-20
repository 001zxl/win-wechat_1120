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
echo === 示例报价表 ===
if not exist data mkdir data
if not exist data\price_list.json (
  copy /y price_list.example.json data\price_list.json >nul
  echo 已复制示例报价表到 data\price_list.json —— 正式使用前请替换成真实价格
) else (
  echo data\price_list.json 已存在，保留不动
)
echo.
echo === 环境自检 ===
python run.py doctor
echo.
echo 安装完成：双击 start.bat 启动控制台，或运行 python server.py
pause
