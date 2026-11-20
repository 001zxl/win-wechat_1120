#!/bin/bash
# macOS 双击启动：打开浏览器聊天控制台
cd "$(dirname "$0")" || exit 1
exec python3 server.py "$@"
