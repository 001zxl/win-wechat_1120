#!/usr/bin/env python3
"""一键启动器：自动选择「真实微信」或「演示模式」，并处理端口占用。

    python launcher.py                # 自动：真实微信窗口在就接微信，否则演示模式
    python launcher.py --real         # 强制连接真实微信
    python launcher.py --demo         # 强制演示模式（虚拟微信，不碰真实账号）
    python launcher.py --port 8766    # 换端口
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import urllib.request
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(ROOT, "server.py")
CONFIG = os.path.join(ROOT, "config.yaml")


def port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def console_alive(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1.5) as resp:
            return b'"ok":true' in resp.read().replace(b" ", b"")
    except Exception:
        return False


def console_platform(port: int) -> str:
    """返回正在运行的控制台平台：win32（真实微信）/ mock（演示）/ ''（读不到）。"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8", "ignore") or "{}")
        return str(((data.get("status") or {}).get("platform")) or "")
    except Exception:
        return ""


def stop_console(port: int) -> int:
    """结束占用该端口的控制台进程（及其子进程）。"""
    pids = set()
    try:
        listing = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True,
                                 text=True, errors="ignore").stdout
        for line in listing.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[1].endswith(f":{port}") and parts[3].upper() == "LISTENING":
                pids.add(parts[4])
    except Exception as exc:
        print(f"[x] 查端口失败：{exc}")
        return 1
    if not pids:
        print(f"端口 {port} 上没有控制台在运行。")
        return 0
    for pid in pids:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"已结束控制台进程 PID={pid}")
    return 0


def run_server(port: int, extra, demo: bool) -> int:
    if not os.path.exists(CONFIG):
        print("[!] 缺少 config.yaml，正在从 config.example.yaml 复制…")
        try:
            import shutil

            shutil.copyfile(os.path.join(ROOT, "config.example.yaml"), CONFIG)
        except Exception as exc:
            print(f"[x] 复制配置失败：{exc}")
            return 1
    args = [sys.executable, SERVER, "--port", str(port)]
    args += ["--demo"] if demo else ["--real"]
    args += list(extra or [])
    return subprocess.call(args, cwd=ROOT)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="微信 AI 自动回复 · 启动器")
    parser.add_argument("--real", action="store_true", help="连接真实微信（默认 dry-run，不发消息）")
    parser.add_argument("--demo", action="store_true", help="演示模式（虚拟微信）")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--stop", action="store_true", help="结束正在运行的控制台")
    args, passthrough = parser.parse_known_args(argv)
    args.extra = passthrough  # 其余参数原样透传给 server.py，如 --live --paused

    if args.stop:
        return stop_console(args.port)

    url = f"http://127.0.0.1:{args.port}/"
    if port_busy(args.port):
        if console_alive(args.port):
            platform = console_platform(args.port)
            mode = "真实微信" if platform == "win32" else ("演示模式（虚拟微信）" if platform == "mock" else "未知模式")
            print(f"控制台已经在运行：{url}")
            print(f"当前模式：{mode}")
            if platform == "mock" and (args.real or not args.demo):
                print("[!] 这个控制台是演示模式，不会连你的真实微信。")
                print("    要切到真实微信：先双击 停止.bat 关掉它，再双击 启动.bat。")
            else:
                print("如需切换模式（演示 ↔ 真实微信），请先双击 停止.bat 再重新启动。")
            webbrowser.open(url)
            return 0
        print(f"[x] 端口 {args.port} 被其他程序占用。可以换个端口：启动.bat --port 8766")
        return 1

    if args.demo:
        return run_server(args.port, args.extra, demo=True)

    if args.real:
        return run_server(args.port, args.extra, demo=False)

    # 自动模式：先尝试真实微信，失败（未登录/无窗口）再退回演示模式
    rc = run_server(args.port, args.extra, demo=False)
    if rc == 0:
        return 0
    print()
    print("[提示] 没连上真实微信（微信未登录，或主窗口被收起 / 未停在聊天界面）。")
    print("       已改用演示模式启动；登录微信 PC 并打开主窗口后，重新双击「启动.bat」即可。")
    print()
    return run_server(args.port, args.extra, demo=True)


if __name__ == "__main__":
    raise SystemExit(main())
