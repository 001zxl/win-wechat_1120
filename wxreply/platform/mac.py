"""macOS 微信后端：CGWindow 抓图 + CGEvent 发送到微信进程 + Vision OCR 配合。

注意：需要系统权限
- 屏幕录制（screencapture 可用）
- 辅助功能（CGEventPostToPid 注入键鼠）
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
from typing import List, Optional, Tuple

from PIL import Image

from .base import MessageItem, WindowInfo

try:  # pyobjc
    import AppKit
    import Quartz
    from ApplicationServices import (
        AXIsProcessTrusted,
        AXIsProcessTrustedWithOptions,
        kAXTrustedCheckOptionPrompt,
    )

    HAVE_QUARTZ = True
except Exception:  # pragma: no cover - 非 macOS 或未装 pyobjc
    HAVE_QUARTZ = False

VK_V = 9
VK_RETURN = 36
FLAG_COMMAND = 1 << 20  # kCGEventFlagMaskCommand


class MacPlatform:
    name = "mac"
    is_windows = False
    strict_name_precheck = True

    def __init__(self, owner_names: Optional[Tuple[str, ...]] = None,
                 activate_on_click: bool = True, passive: bool = True) -> None:
        if not HAVE_QUARTZ:
            raise RuntimeError("macOS 后端需要 pyobjc：pip install pyobjc-framework-Cocoa "
                               "pyobjc-framework-Quartz pyobjc-framework-ApplicationServices pyobjc-framework-Vision")
        self.owner_names = owner_names or ("WeChat", "微信", "Weixin")
        self.activate_on_click = activate_on_click
        # passive=True：只截图/OCR，绝不点击、滚动、聚焦微信（避免抢占鼠标）
        self.passive = passive
        self.window: Optional[WindowInfo] = None
        self._window_number = 0
        self._pid = 0
        self._bounds: Tuple[float, float, float, float] = (0, 0, 0, 0)  # x, y, w, h (points)
        self._scale = 1.0
        self._previous_pid = 0

    # ---------- 权限 ----------
    @staticmethod
    def accessibility_trusted() -> bool:
        try:
            return bool(AXIsProcessTrusted())
        except Exception:
            return False

    @staticmethod
    def request_accessibility() -> None:
        try:
            AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})
        except Exception:
            pass

    def layout_preset(self) -> dict:
        """微信 Mac 4.x 的初始比例（可用校准面板微调）。"""
        return {"nav_ratio": 0.067, "list_ratio": 0.337,
                "header_ratio": 0.08, "input_ratio": 0.40}

    # ---------- 窗口 ----------
    def _enumerate(self) -> List[dict]:
        try:
            windows = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionAll,
                                                        Quartz.kCGNullWindowID)
        except Exception:
            return []
        result = []
        for item in windows or []:
            owner = str(item.get(Quartz.kCGWindowOwnerName, ""))
            if owner not in self.owner_names:
                continue
            if int(item.get(Quartz.kCGWindowLayer, 0)) != 0:
                continue
            bounds = item.get(Quartz.kCGWindowBounds) or {}
            width = float(bounds.get("Width", 0) or 0)
            height = float(bounds.get("Height", 0) or 0)
            if width < 300 or height < 200:
                continue
            result.append({
                "number": int(item.get(Quartz.kCGWindowNumber, 0)),
                "pid": int(item.get(Quartz.kCGWindowOwnerPID, 0)),
                "name": str(item.get(Quartz.kCGWindowName, "") or owner),
                "bounds": (float(bounds.get("X", 0) or 0), float(bounds.get("Y", 0) or 0),
                           width, height),
                "onscreen": bool(item.get(Quartz.kCGWindowIsOnscreen, False)),
            })
        return result

    def find_window(self) -> Optional[WindowInfo]:
        candidates = self._enumerate()
        if not candidates:
            self.window = None
            return None

        def area(item: dict) -> float:
            return item["bounds"][2] * item["bounds"][3]

        best = max(candidates, key=area)
        self._window_number = best["number"]
        self._pid = best["pid"]
        self._bounds = best["bounds"]
        self.window = WindowInfo(hwnd=best["pid"], title=best["name"] or "WeChat",
                                 class_name="wechat-mac")
        return self.window

    # ---------- 抓图 ----------
    def capture(self) -> Optional[Image.Image]:
        if self.window is None:
            self.find_window()
        if self.window is None:
            return None
        path = os.path.join(tempfile.gettempdir(), f"wechat-mac-{int(time.time() * 1000)}.png")
        try:
            subprocess.run(
                ["screencapture", "-x", "-o", "-l", str(self._window_number), path],
                check=True, timeout=15,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            image = Image.open(path).convert("RGB")
        except Exception:
            x, y, width, height = [int(v) for v in self._bounds]
            try:
                subprocess.run(
                    ["screencapture", "-x", "-R", f"{x},{y},{width},{height}", path],
                    check=True, timeout=15,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                image = Image.open(path).convert("RGB")
            except Exception:
                return None
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        if self._bounds[2] > 0:
            self._scale = image.width / max(1.0, self._bounds[2])
        return image

    # ---------- 鼠标 ----------
    def _post_mouse(self, event_type, point: Tuple[float, float]) -> None:
        event = Quartz.CGEventCreateMouseEvent(None, event_type, point, Quartz.kCGMouseButtonLeft)
        Quartz.CGEventPostToPid(self._pid, event)

    def click(self, x: float, y: float) -> None:
        if self.passive:
            return
        if self.window is None:
            self.find_window()
        if not self._pid:
            return
        if self.activate_on_click:
            self.ensure_visible()
        gx = self._bounds[0] + x / max(self._scale, 0.01)
        gy = self._bounds[1] + y / max(self._scale, 0.01)
        point = Quartz.CGPointMake(float(gx), float(gy))
        try:
            before = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        except Exception:
            before = None
        for event_type in (Quartz.kCGEventMouseMoved, Quartz.kCGEventLeftMouseDown,
                           Quartz.kCGEventLeftMouseUp):
            event = Quartz.CGEventCreateMouseEvent(None, event_type, point,
                                                   Quartz.kCGMouseButtonLeft)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            time.sleep(0.03)
        if before is not None:  # 光标归位
            event = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventMouseMoved,
                                                   before, Quartz.kCGMouseButtonLeft)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        time.sleep(0.05)

    # ---------- 前台 ----------
    def get_foreground(self) -> int:
        try:
            app = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
            return int(app.processIdentifier()) if app else 0
        except Exception:
            return 0

    def _wait_frontmost(self, timeout: float = 2.5) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.get_foreground() == self._pid:
                return True
            time.sleep(0.1)
        return False

    def _raise_window_ax(self) -> None:
        """用辅助功能把窗口取消最小化并抬到最前（Cmd+F 等快捷键需要窗口可见）。"""
        try:
            from ApplicationServices import (
                AXUIElementCreateApplication, AXUIElementCopyAttributeValue,
                AXUIElementSetAttributeValue, AXUIElementPerformAction,
                kAXWindowsAttribute, kAXMinimizedAttribute, kAXRaiseAction,
            )
            app = AXUIElementCreateApplication(self._pid)
            err, windows = AXUIElementCopyAttributeValue(app, kAXWindowsAttribute, None)
            if err != 0 or not windows:
                return
            for window in windows:
                try:
                    err_min, minimized = AXUIElementCopyAttributeValue(
                        window, kAXMinimizedAttribute, None)
                    if err_min == 0 and minimized:
                        AXUIElementSetAttributeValue(window, kAXMinimizedAttribute, False)
                    AXUIElementPerformAction(window, kAXRaiseAction)
                except Exception:
                    continue
        except Exception:
            return

    def ensure_visible(self) -> bool:
        if self.passive:
            return True
        """微信窗口隐藏/最小化时，点击会落到别的应用上；先把它叫到前台。"""
        if self.window is None:
            self.find_window()
        if self.window is None:
            return False
        self._raise_window_ax()
        if self.get_foreground() == self._pid:
            return True
        if self.focus(self.window):
            return True
        for command in (["open", "-a", "WeChat"], ["open", "-b", "com.tencent.xinWeChat"]):
            try:
                subprocess.run(command, check=False, timeout=10)
            except Exception:
                continue
            time.sleep(1.0)
            self.find_window()
            if self.focus(self.window) or self._wait_frontmost(2.5):
                return True
        return False

    def focus(self, window: WindowInfo) -> bool:
        if self.passive:
            return True
        try:
            if window is None:
                return False
            current = self.get_foreground()
            if current and current != self._pid:
                self._previous_pid = current
            app = AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(self._pid)
            if app is None:
                return False
            app.activateWithOptions_(2)  # NSApplicationActivateIgnoringOtherApps
            return self._wait_frontmost(1.2)
        except Exception:
            return False

    def restore_foreground(self, hwnd: int) -> None:
        if not hwnd or hwnd == self._pid:
            return
        try:
            app = AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(int(hwnd))
            if app is not None:
                app.activateWithOptions_(2)
        except Exception:
            pass

    def scroll(self, x: float, y: float, delta: int, times: int = 1) -> None:
        if self.passive:
            return
        """在窗口内 (x,y) 处滚动鼠标滚轮（正数向上，负数向下），完成后光标归位。"""
        if self.window is None:
            self.find_window()
        if not self._pid:
            return
        # 只在微信不在前台时唤醒一次，避免每次滚轮都做聚焦等待
        if self.get_foreground() != self._pid:
            self.ensure_visible()
        gx = self._bounds[0] + x / max(self._scale, 0.01)
        gy = self._bounds[1] + y / max(self._scale, 0.01)
        point = Quartz.CGPointMake(float(gx), float(gy))
        try:
            before = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        except Exception:
            before = None
        Quartz.CGEventPost(Quartz.kCGHIDEventTap,
                           Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventMouseMoved,
                                                          point, Quartz.kCGMouseButtonLeft))
        time.sleep(0.05)
        for _ in range(max(1, times)):
            event = Quartz.CGEventCreateScrollWheelEvent(None, Quartz.kCGScrollEventUnitLine,
                                                         1, int(delta))
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            time.sleep(0.05)
        if before is not None:
            Quartz.CGEventPost(Quartz.kCGHIDEventTap,
                               Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventMouseMoved,
                                                              before, Quartz.kCGMouseButtonLeft))
        time.sleep(0.1)

    # ---------- 剪贴板/键盘 ----------
    def get_clipboard(self) -> Optional[str]:
        try:
            result = subprocess.run(["pbpaste"], capture_output=True, timeout=10)
            return result.stdout.decode("utf-8", errors="replace")
        except Exception:
            return None

    def set_clipboard(self, text: str) -> bool:
        try:
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True, timeout=10)
            return True
        except Exception:
            return False

    def _post_key(self, keycode: int, flags: int = 0) -> None:
        for down in (True, False):
            event = Quartz.CGEventCreateKeyboardEvent(None, keycode, down)
            Quartz.CGEventSetFlags(event, flags)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            time.sleep(0.02)

    def press_command_key(self, keycode: int) -> None:
        """发送 Cmd+<key>（HID）。"""
        self.ensure_visible()
        for down in (True, False):
            event = Quartz.CGEventCreateKeyboardEvent(None, keycode, down)
            Quartz.CGEventSetFlags(event, FLAG_COMMAND)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            time.sleep(0.03)

    def press_ctrl_v(self) -> None:
        self.ensure_visible()
        self._post_key(VK_V, FLAG_COMMAND)
        time.sleep(0.05)

    def press_escape(self) -> None:
        self.ensure_visible()
        self._post_key(53)  # kVK_Escape

    def press_enter(self) -> None:
        self.ensure_visible()
        self._post_key(VK_RETURN)
        time.sleep(0.05)

    # ---------- 结构化通道（走 OCR） ----------
    def read_header_text(self) -> Optional[str]:
        return None

    def read_incoming_items(self) -> Optional[List[MessageItem]]:
        return None

    def last_outgoing_text(self) -> Optional[str]:
        return None

    def send_image(self, path: str) -> bool:
        return False

    def close(self) -> None:
        return None
