#!/usr/bin/env python3
"""微信 AI 回复 · 浏览器聊天控制台

结构：左侧会话、中间聊天与草稿、右侧工作进度。
Windows 连接真实微信；macOS 用 mock 演示。

    python server.py --mock --demo          # 浏览器打开演示
    python server.py                       # Windows 真机（dry-run）
    python server.py --live                # Windows 真机（真实发送）
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty
from urllib.parse import parse_qs, quote, urlparse

from wxreply.agent import Agent
from wxreply.config import Config, load_config
from wxreply.platform.mock import MockPlatform
from wxreply.platform.win32 import IS_WINDOWS, Win32Platform
from wxreply.settings import SettingsStore

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT, "web")


def resolve_static_path(web_dir: str, rel: str) -> Optional[str]:
    """把请求路径解析为 web/ 内的真实文件；越界（../、绝对路径）一律返回 None。"""
    root = os.path.realpath(web_dir)
    clean = (rel or "").split("?", 1)[0].lstrip("/").replace("\\", "/")
    target = os.path.realpath(os.path.join(root, clean))
    if target != root and not target.startswith(root + os.sep):
        return None
    return target if os.path.isfile(target) else None


def seed_demo(platform) -> None:
    if not isinstance(platform, MockPlatform):
        return
    platform.sessions[0].messages[:] = [
        {"dir": "in", "kind": "text", "text": "你好，在吗？"},
        {"dir": "in", "kind": "image", "text": "", "description": ""},
        {"dir": "in", "kind": "text", "text": "户外折叠桌 FT-120，500 件什么价？"},
        {"dir": "in", "kind": "voice", "text": "", "description": ""},
    ]
    platform.sessions[0].unread = 4
    platform.sessions[1].unread = 1
    platform.sessions[1].messages[:] = [
        ("in", "上次的样品收到了，质量不错"),
        ("in", "想再订 2000 件，什么时候能出货？"),
    ]


def build_setup_checks(platform: str, window_found: bool, ocr_available: bool,
                       layout_saved: bool, whitelist, key_ok: bool) -> dict:
    checks = [
        {"id": "windows", "label": "桌面微信环境", "ok": platform in ("win32", "mac"),
         "hint": "Windows 或 macOS 真机；mock 仅演示", "action": ""},
        {"id": "wechat", "label": "微信窗口", "ok": bool(window_found),
         "hint": "登录微信 PC，保持窗口打开（可以被遮挡）", "action": ""},
        {"id": "ocr", "label": "RapidOCR", "ok": bool(ocr_available),
         "hint": "运行 setup.bat，或 pip install rapidocr_onnxruntime", "action": "ocr"},
        {"id": "layout", "label": "布局校准", "ok": bool(layout_saved),
         "hint": "点「校准布局」拖动滑块贴合微信窗口后保存", "action": "calibrate"},
        {"id": "whitelist", "label": "白名单", "ok": bool(whitelist),
         "hint": "左栏「白名单」添加真实联系人显示名", "action": "whitelist"},
        {"id": "key", "label": "模型 Key", "ok": bool(key_ok),
         "hint": "点「设置」填写当前模型的 API Key", "action": "settings"},
    ]
    return {"platform": platform, "checks": checks,
            "ready": all(item["ok"] for item in checks)}


class ConsoleState:
    def __init__(self, cfg: Config, platform, dry_run: bool) -> None:
        self.cfg = cfg
        self.platform = platform
        self.settings = SettingsStore(cfg.resolve(cfg.settings_path))
        self.agent = Agent(cfg, platform, dry_run=dry_run, notify=lambda _msg: None)
        try:
            from wxreply.channels import ChannelConfig, build_registry
            configs = [ChannelConfig(id=str(item.get("id", item.get("type", "wechat"))),
                                     type=str(item.get("type", "wechat")),
                                     enabled=bool(item.get("enabled", True)),
                                     options={str(k): str(v) for k, v in (item.get("options") or {}).items()})
                       for item in (cfg.channels or [])]
            self.channels = build_registry(configs, self.agent) if configs else None
        except Exception:
            self.channels = None
        self.selected = None
        self.events = deque(maxlen=300)
        self.seq = 0
        self.lock = threading.Lock()
        self.worker = threading.Thread(target=self.agent.run, daemon=True)
        self.worker.start()
        self.pump = threading.Thread(target=self._pump, daemon=True)
        self.pump.start()

    # ---------- 事件 ----------
    def _pump(self) -> None:
        while True:
            try:
                event = self.agent.events.get(timeout=1)
            except Empty:
                continue
            except Exception:
                return
            with self.lock:
                self.seq += 1
                event["seq"] = self.seq
                event["text"] = self._format(event)
                if event.get("peer") and not self.selected and event["type"] in ("incoming", "draft"):
                    self.selected = event["peer"]
                self.events.append(event)

    @staticmethod
    def _format(event) -> str:
        kind = event.get("type")
        if kind == "cycle":
            return f"扫描窗口：{event.get('badges', 0)} 个未读会话"
        if kind == "incoming":
            return f"{event.get('peer')} 发来消息：{' / '.join(event.get('messages', []))[:100]}"
        if kind == "draft":
            return f"草稿就绪（{event.get('model')} · {event.get('latency_ms')}ms）"
        if kind == "sent":
            return f"发送 {event.get('status')}：{event.get('detail', '')}"
        if kind == "skip":
            return f"跳过 {event.get('peer', '')}：{event.get('reason', '')}"
        if kind == "escalate":
            return f"转人工 {event.get('peer', '')}：{event.get('reason', '')}"
        if kind == "blocked":
            return f"拦截回复：{event.get('reason', '')}"
        if kind == "rate_limited":
            return f"限频：{event.get('peer', '')} 等待 {event.get('wait')}s"
        if kind == "inbound":
            return f"外部消息 {event.get('peer', '')}：{str(event.get('incoming', ''))[:40]} → {str(event.get('reply', ''))[:40]}"
        if kind == "error":
            return f"错误：{event.get('message', '')}"
        return str(event.get("message", kind or ""))

    # ---------- 快照 ----------
    def snapshot(self) -> dict:
        with self.lock:
            pending = self.agent.pending_for(self.selected) if self.selected else self.agent.pending
            history = []
            if self.selected:
                history = self.agent.conversation(self.selected)
            conversations = []
            for conv in self.agent.conversations():
                avatar_path = conv.get("avatar_path")
                if not avatar_path:
                    for key, value in self.agent._conv.items():
                        candidate = value.get("avatar_path")
                        if candidate and (key in conv["peer"] or conv["peer"] in key):
                            avatar_path = candidate
                            break
                conversations.append({
                    "peer": conv["peer"], "status": conv["status"],
                    "last_text": conv["last_text"], "unread": conv["unread"],
                    "ts": conv["last_ts"], "time": conv.get("time", ""),
                    "avatar": ("/api/avatar?name=" + quote(conv["peer"]) +
                               ("&v=" + str(int(os.path.getmtime(avatar_path)*1_000_000)) if avatar_path else ""))
                              if avatar_path else "",
                    "simulated": bool(conv.get("simulated") or conv["peer"].startswith("【模拟】")),
                })
            events = list(self.events)[-60:]
            return {
                "status": {
                    "platform": self.platform.name,
                    "dry_run": self.agent.dry_run,
                    "paused": self.agent.paused.is_set(),
                "channels": self.channels.health() if self.channels else [],
                "offscreen": bool(getattr(self.platform, "is_offscreen", lambda: False)()),
                "readonly": bool(getattr(self.platform, "passive", False)),
                    "model": self.agent.current_model(),
                    "models": self.agent.model_options(),
                    "vision": self.agent.vision.available,
                    "tools": self.agent.tools.enabled,
                    "group": self.cfg.runtime.allow_group_chat,
                    "memory": self.cfg.memory.db_path,
                    "whitelist_names": list(self.cfg.safety.whitelist),
                    "handoffs": self.agent.handoff.items(),
                    "handoff_corrupt": bool(self.agent.handoff.corrupt),
                    "layout": self.agent.layout_ratios(),
                    "vision_model": self.cfg.ai.vision_model,
                    "embedding_model": self.cfg.ai.embedding_model,
                    "settings_file": self.cfg.resolve(self.cfg.settings_path),
                    "ocr": self.agent.ocr.available,
                    "persona": self.agent.persona.display_name,
                    "kb": self.agent.kb.size,
                    "kb_version": self.agent.kb.version,
                    "whitelist": len(self.cfg.safety.whitelist),
                    "llm_ready": self.agent.router.client.ready,
                },
                "conversations": conversations,
                "selected": self.selected,
                "history": history,
                "pending": None if not pending else {
                    "peer": pending["peer"], "reply": pending["reply"],
                    "model": pending["model"], "latency_ms": pending["latency_ms"],
                },
                "events": events,
            }

    # ---------- 动作 ----------
    def act(self, action: str, payload: dict) -> dict:
        try:
            if action == "select":
                peer = str(payload.get("peer") or "")
                if peer in {conv["peer"] for conv in self.agent.conversations()}:
                    self.selected = peer
                    self.agent.mark_read(peer)
                    self.agent._view_history[peer] = [("sys", "正在读取聊天记录…")]
                    was_paused = self.agent.paused.is_set()
                    self.agent.pause()   # 读会话期间暂停自动扫描，避免和点击抢窗口

                    def read_passive() -> None:
                        def current_state():
                            shot = self.platform.capture()
                            if shot is None:
                                return None, None, None
                            where = self.agent.layout.regions(shot.width, shot.height)
                            return shot, where, self.agent.sender.read_header(shot, where)

                        try:
                            image, regions, header = current_state()
                            if image is None:
                                self.agent._view_history[peer] = [("sys", "抓图失败")]
                                return
                            if not self.agent._same_peer(header, peer):
                                # 微信当前没打开这个会话：主动点开（等价于你手动点一下会话行）
                                try:
                                    if self.agent.open_conversation(peer):
                                        image, regions, header = current_state()
                                except Exception as exc:
                                    self.agent._view_history[peer] = [("sys", f"打开会话失败：{exc}")]
                                    return
                            if image is not None and self.agent._same_peer(header, peer):
                                self.agent.read_conversation(peer, open_first=False)
                            else:
                                self.agent._view_history[peer] = [
                                    ("sys", "没能自动打开这个会话：请先在微信里点开它，再点「读取当前会话」")
                                ]
                        finally:
                            if not was_paused:
                                self.agent.resume()

                    threading.Thread(target=read_passive, daemon=True).start()
                    return {"ok": True, "reading": True}
                return {"ok": False, "error": "unknown peer"}
            if action == "approve":
                peer = str(payload.get("peer") or self.selected or "")
                if not peer:
                    return {"ok": False, "error": "必须指定联系人（草稿与联系人绑定）",
                            "result": None}
                result = self.agent.approve_pending(payload.get("text"), peer=peer,
                                                    draft_id=payload.get("draft_id"))
                return {"ok": bool(result and result.get("ok")), "result": result}
            if action == "regen":
                result = self.agent.regenerate_pending(peer=self.selected)
                return {"ok": bool(result), "result": result}
            if action == "discard":
                self.agent.discard_pending(peer=self.selected)
                return {"ok": True}
            if action == "pause":
                if payload.get("paused"):
                    self.agent.pause()
                else:
                    self.agent.resume()
                return {"ok": True}
            if action == "stop":
                self.agent.stop_sending()
                return {"ok": True}
            if action == "live":
                self.agent.set_dry_run(not bool(payload.get("live")))
                return {"ok": True}
            if action == "instruction":
                self.agent.set_instruction(str(payload.get("text") or ""))
                return {"ok": True}
            if action == "open":
                peer = str(payload.get("peer") or "")
                return {"ok": self.agent.open_conversation(peer)} if peer else {"ok": False}
            if action == "sync":
                return {"ok": True, "peers": self.agent.sync_conversations(deep=True)}
            if action == "takeover":
                peer = str(payload.get("peer") or self.selected or "")
                if not peer:
                    return {"ok": False, "error": "缺少 peer"}
                self.agent.takeover(peer, str(payload.get("reason") or "人工接管"))
                return {"ok": True, "peer": peer}
            if action == "release":
                peer = str(payload.get("peer") or self.selected or "")
                if not peer:
                    return {"ok": False, "error": "缺少 peer"}
                return {"ok": self.agent.release_handoff(peer), "peer": peer}
            if action == "handoffs":
                return {"ok": True, "items": self.agent.handoff.items()}
            if action == "kb_reload":
                docs = self.agent.kb.reload()
                self.agent.emit("status", message=f"知识库已重新加载：{docs} 段，版本 {self.agent.kb.version}")
                return {"ok": True, "docs": docs, "version": self.agent.kb.version}
            if action == "offscreen":
                platform = self.agent.platform
                if not (hasattr(platform, "mount_offscreen") and hasattr(platform, "restore_from_offscreen")):
                    return {"ok": False, "error": "当前平台不支持离屏挂载（仅 Windows）"}
                try:
                    if getattr(platform, "is_offscreen", lambda: False)():
                        ok = platform.restore_from_offscreen()
                    else:
                        ok = platform.mount_offscreen()
                    return {"ok": bool(ok), "offscreen": bool(getattr(platform, "is_offscreen", lambda: False)())}
                except Exception as exc:
                    return {"ok": False, "error": str(exc)}
            if action == "simulate":
                peer = str(payload.get("peer") or "测试客户")
                text = str(payload.get("text") or "").strip()
                if not text:
                    return {"ok": False, "error": "客户消息为空，请先输入内容"}
                result = self.agent.simulate_incoming(peer, text, bool(payload.get("auto", True)))
                if result:
                    self.selected = result.get("peer", peer)
                    return {"ok": True, "result": result}
                return {"ok": False,
                        "error": getattr(self.agent, "_last_error", "") or "生成回复失败，请查看右侧工作进度"}
            if action == "read_current":
                peer = self.agent.read_current()
                if peer:
                    self.selected = peer
                return {"ok": peer is not None, "peer": peer}
            if action == "process_current":
                result = self.agent.process_current()
                return {"ok": result is not None, "result": result}
            if action == "model":
                return {"ok": self.agent.set_model(str(payload.get("id") or ""))}
            if action == "api_keys":
                keys = payload.get("keys") or {}
                if not isinstance(keys, dict):
                    return {"ok": False, "error": "keys must be an object"}
                self.settings.set_api_keys(keys)
                self.settings.save()
                self.agent.apply_settings(self.settings.data)
                return {"ok": True}
            if action == "whitelist":
                names = payload.get("names") or []
                if not isinstance(names, list):
                    return {"ok": False, "error": "names must be an array"}
                self.settings.set_whitelist([str(item) for item in names])
                self.settings.save()
                self.agent.apply_settings(self.settings.data)
                return {"ok": True, "names": list(self.cfg.safety.whitelist)}
            if action == "layout":
                ratios = payload.get("ratios") or {}
                if not isinstance(ratios, dict):
                    return {"ok": False, "error": "ratios must be an object"}
                self.settings.set_layout(ratios)
                self.settings.save()
                self.agent.apply_settings(self.settings.data)
                return {"ok": True, "layout": self.agent.layout_ratios()}
            if action == "models":
                self.settings.set_models(payload.get("vision_model"), payload.get("embedding_model"))
                self.settings.save()
                self.agent.apply_settings(self.settings.data)
                return {"ok": True}
            return {"ok": False, "error": f"unknown action {action}"}
        except Exception as exc:  # 防御：UI 不能把服务打挂
            return {"ok": False, "error": str(exc)}

    def close(self) -> None:
        self.agent.stop()
        try:
            self.agent.close()
        except Exception:
            pass


def make_handler(state: ConsoleState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "wxreply-console/0.2"

        def log_message(self, *args):  # 静默
            return

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/state":
                self._send(200, json.dumps(state.snapshot(), ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
                return
            if path == "/api/health":
                worker = getattr(state, "worker", None)
                alive = bool(worker and worker.is_alive())
                last = float(getattr(state.agent, "last_scan_ts", 0) or 0)
                payload = {
                    "ok": alive,
                    "worker": alive,
                    "last_scan_age": round(time.time() - last, 1) if last else None,
                }
                self._send(200 if alive else 503,
                           json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
                return
            if path == "/api/avatar":
                query = parse_qs(urlparse(self.path).query)
                name = (query.get("name") or [""])[0]
                entry = state.agent._conv.get(name) or {}
                avatar_path = entry.get("avatar_path")
                if not avatar_path:
                    # 名称 OCR 偶有尾字差异，做一次包含匹配兜底
                    for key, value in state.agent._conv.items():
                        if value.get("avatar_path") and (key in name or name in key):
                            avatar_path = value["avatar_path"]
                            break
                if avatar_path and os.path.isfile(avatar_path):
                    with open(avatar_path, "rb") as handle:
                        data = handle.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self._send(404, b"not found", "text/plain")
                return
            if path == "/api/setup":
                window = state.platform.find_window()
                key_ok = any(option.api_key for option in state.cfg.ai.options()) or (
                    bool(state.cfg.ai.api_key) and "REPLACE" not in state.cfg.ai.api_key
                )
                payload = build_setup_checks(
                    platform=state.platform.name,
                    window_found=window is not None,
                    ocr_available=state.agent.ocr.available,
                    layout_saved=bool(state.settings.layout()),
                    whitelist=state.cfg.safety.whitelist,
                    key_ok=key_ok,
                )
                self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
                return
            if path == "/api/calibrate":
                preview = state.agent.capture_preview()
                if preview is None:
                    self._send(200, json.dumps({"ok": False, "error": "抓图失败"}, ensure_ascii=False).encode("utf-8"),
                               "application/json; charset=utf-8")
                    return
                image, regions = preview
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                payload = {
                    "ok": True,
                    "image": "data:image/png;base64," + encoded,
                    "width": image.width,
                    "height": image.height,
                    "ratios": state.agent.layout_ratios(),
                    "regions": {name: list(box) for name, box in regions.to_dict().items()},
                }
                self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
                return
            if path == "/":
                path = "/index.html"
            # 静态目录必须锁死在 web/ 内：`/../config.yaml` 这类跳转一律 404
            file_path = resolve_static_path(WEB_DIR, path)
            if file_path is None:
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            content_type = {
                ".html": "text/html; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8",
            }.get(os.path.splitext(file_path)[1], "application/octet-stream")
            with open(file_path, "rb") as handle:
                self._send(200, handle.read(), content_type)

        def _handle_inbound(self, path: str) -> None:
            from wxreply.channels import route_inbound

            channel_id = path[len("/api/inbound/"):].strip("/")
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                self._send(400, b'{"ok":false,"error":"bad json"}', "application/json; charset=utf-8")
                return
            result = route_inbound(state.channels, state.agent, channel_id, payload)
            self._send(200, json.dumps(result, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _action_allowed(self) -> bool:
            """操作接口的基本防护：只接受本机、JSON、可选口令。

            浏览器同源请求会自动带 Origin，跨站页面伪造 POST 会被拒。
            """
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                return False
            host = (self.headers.get("Host") or "").split(":")[0].strip().lower()
            if host not in ("127.0.0.1", "localhost", "[::1]", "::1"):
                return False
            origin = (self.headers.get("Origin") or "").strip().lower()
            if origin:
                parsed = urlparse(origin)
                if parsed.scheme not in ("http", "https") or \
                        (parsed.hostname or "") not in ("127.0.0.1", "localhost", "::1"):
                    return False
            token = os.environ.get("WX_CONSOLE_TOKEN", "").strip()
            if token and (self.headers.get("X-Console-Token") or "").strip() != token:
                return False
            return True

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path.startswith("/api/inbound/"):
                self._handle_inbound(path)
                return
            if path != "/api/action":
                self._send(404, b"not found", "text/plain")
                return
            if not self._action_allowed():
                self._send(403, b'{"ok":false,"error":"forbidden"}',
                           "application/json; charset=utf-8")
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                self._send(400, b'{"ok":false}', "application/json")
                return
            result = state.act(str(payload.get("action") or ""), payload)
            self._send(200, json.dumps(result, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

    return Handler


def build_platform(cfg: Config, mock: bool, allow_actions: bool = False):
    if mock:
        return MockPlatform()
    if sys.platform == "darwin":
        from wxreply.platform.mac import MacPlatform

        passive = not (allow_actions or cfg.runtime.allow_real_actions)
        platform = MacPlatform(activate_on_click=cfg.wechat.mac_activate_on_click, passive=passive)
        if platform.find_window() is None:
            raise SystemExit("未找到微信窗口：请先打开微信并停在聊天界面")
        return platform
    if not IS_WINDOWS:
        return MockPlatform()
    platform = Win32Platform(cfg.wechat.title_keywords, cfg.wechat.class_keywords, cfg.wechat.process_names)
    if platform.find_window() is None:
        raise SystemExit("未找到微信窗口，请先登录微信 PC 并保持窗口打开")
    return platform


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="微信 AI 回复 · 浏览器聊天控制台")
    parser.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    parser.add_argument("--mock", action="store_true", help="用合成窗口（默认）")
    parser.add_argument("--real", action="store_true", help="连接真实微信（默认只读，不点击不发送）")
    parser.add_argument("--allow-actions", action="store_true",
                        help="显式允许真实鼠标/发送（默认为只读，绝不操作微信）")
    parser.add_argument("--demo", action="store_true", help="预置演示会话")
    parser.add_argument("--live", action="store_true", help="允许真实发送")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--auto-exit", type=int, default=0, help="N 秒后退出（测试用）")
    parser.add_argument("--paused", action="store_true", help="启动后暂停自动扫描（只用界面手动处理）")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    platform = build_platform(cfg, args.mock or not args.real, args.allow_actions)
    if args.demo:
        seed_demo(platform)
        cfg.runtime.poll_interval_s = 0.6
    state = ConsoleState(cfg, platform, dry_run=not args.live)
    if args.paused:
        state.agent.pause()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"聊天控制台已启动：{url}")
    print(f"platform={platform.name} dry_run={state.agent.dry_run} "
          f"persona={state.agent.persona.display_name} kb={state.agent.kb.size}")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    if args.auto_exit:
        threading.Timer(args.auto_exit, lambda: os._exit(0)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
