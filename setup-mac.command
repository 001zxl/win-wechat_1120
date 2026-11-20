#!/bin/bash
# macOS 一键安装：venv + pyobjc（真实微信后端）
cd "$(dirname "$0")" || exit 1
echo "=== 创建 venv 并安装依赖 ==="
python3 -m venv data/venv
data/venv/bin/python -m pip install --quiet --upgrade pip
data/venv/bin/pip install --quiet -r requirements-mac.txt
echo "=== 权限检查 ==="
data/venv/bin/python - <<'PY'
import sys
sys.path.insert(0, '.')
from wxreply.platform.mac import MacPlatform
p = MacPlatform()
print("屏幕录制：", "可用" if p.capture() else "不可用（系统设置→隐私与安全性→屏幕录制）")
print("辅助功能：", "已授权" if MacPlatform.accessibility_trusted() else "未授权（系统设置→隐私与安全性→辅助功能）")
w = p.find_window()
print("微信窗口：", w.title if w else "未找到，请打开微信聊天窗口")
PY
echo "=== 启动 ==="
data/venv/bin/python server.py --paused "$@"
