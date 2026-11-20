"""Windows 微信窗口：PrintWindow 离屏抓图 + PostMessage 零鼠标点击 + 剪贴板/键盘。

包含 DWM 边框、前台还原与结构化接口骨架。
"""
from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes
from typing import List, Optional

from .base import WindowInfo

IS_WINDOWS = hasattr(ctypes, "windll")

if IS_WINDOWS:
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    kernel32 = ctypes.windll.kernel32
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass
else:  # macOS 上仅做导入兼容
    user32 = gdi32 = kernel32 = None  # type: ignore


WM_MOUSEMOVE = 0x0200
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202

SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040

VK_CONTROL = 0x11
VK_V = 0x56
VK_RETURN = 0x0D
KEYEVENTF_KEYUP = 0x0002

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class Win32Platform:
    name = "win32"
    is_windows = True

    def __init__(self, title_keywords: List[str], class_keywords: List[str],
                 process_names: Optional[List[str]] = None) -> None:
        if not IS_WINDOWS:
            raise RuntimeError("Win32Platform 只能在 Windows 上使用")
        self.title_keywords = [k.lower() for k in title_keywords]
        self.class_keywords = [k.lower() for k in class_keywords]
        self.process_names = [k.lower() for k in (process_names or ["WeChat.exe", "Weixin.exe"])]
        self.window: Optional[WindowInfo] = None
        self._clipboard = ""

    # ---------- 窗口发现 ----------
    def _enumerate(self) -> List[WindowInfo]:
        result: List[WindowInfo] = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def callback(hwnd, _lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            class_buffer = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, class_buffer, 256)
            result.append(
                WindowInfo(int(hwnd), buffer.value.strip(), class_buffer.value.strip())
            )
            return True

        user32.EnumWindows(callback, 0)
        return result

    def _process_name(self, hwnd: int) -> str:
        """窗口所属进程的可执行文件名（小写），用于锁定真微信、排除同名浏览器窗口。"""
        try:
            pid = wintypes.DWORD(0)
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if not pid.value:
                return ""
            handle = kernel32.OpenProcess(0x1000, False, pid.value)  # QUERY_LIMITED_INFORMATION
            if not handle:
                return ""
            try:
                buffer = ctypes.create_unicode_buffer(1024)
                size = wintypes.DWORD(len(buffer))
                if kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                    return os.path.basename(buffer.value).lower()
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return ""
        return ""

    MAIN_CLASS_HINTS = ("wechatmainwndforpc", "qwindowicon")

    def find_window(self) -> Optional[WindowInfo]:
        """只认 WeChat.exe / Weixin.exe 的窗口；主窗口收在托盘时会自动还原。

        不再用「标题 / 窗口类」兜底匹配：浏览器标题里带「微信」「WeChat」时
        会被误当成微信窗口（截图/点击全打到浏览器上）。
        """

        def area(item: WindowInfo) -> int:
            try:
                left, top, right, bottom = self._window_rect(item.hwnd)
            except Exception:
                return 0
            return max(0, right - left) * max(0, bottom - top)

        owned = []
        for window in self._enumerate():
            class_name = window.class_name.lower()
            if "tool" in class_name:  # Qt 托盘/工具窗，不是聊天主窗口
                continue
            if self._process_name(window.hwnd) not in self.process_names:
                continue
            owned.append(window)
        if not owned:
            self.window = None
            return None

        main = [item for item in owned
                if any(hint in item.class_name.lower() for hint in self.MAIN_CLASS_HINTS)] or owned
        main.sort(key=area, reverse=True)
        target = main[0]
        if user32.IsIconic(target.hwnd):  # SW_RESTORE：微信收在托盘时把主窗口调出来
            user32.ShowWindow(target.hwnd, 9)
            time.sleep(0.35)
        self.window = target
        return target

    # ---------- 几何 ----------
    def _window_rect(self, hwnd: int):
        rect = RECT()
        try:  # DWM 扩展边框（更贴近可见窗口）
            user32.DwmGetWindowAttribute(
                hwnd, ctypes.c_uint(9), ctypes.byref(rect), ctypes.sizeof(rect)
            )
            if rect.right > rect.left and rect.bottom > rect.top:
                return rect.left, rect.top, rect.right, rect.bottom
        except Exception:
            pass
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            raise RuntimeError("GetWindowRect 失败")
        return rect.left, rect.top, rect.right, rect.bottom

    # ---------- 离屏抓图 ----------
    def _ensure_window(self) -> Optional[WindowInfo]:
        """缓存的 hwnd 失效（微信重开窗口/重新登录）时自动重新绑定。"""
        if self.window is not None and user32.IsWindow(self.window.hwnd):
            return self.window
        self.window = None
        return self.find_window()

    def capture(self):
        from PIL import Image
        import numpy as np

        if self._ensure_window() is None:
            raise RuntimeError("尚未找到微信窗口")
        hwnd = self.window.hwnd
        left, top, right, bottom = self._window_rect(hwnd)
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            raise RuntimeError("窗口尺寸无效（最小化？）")
        hwnd_dc = user32.GetWindowDC(hwnd)
        mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
        bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
        gdi32.SelectObject(mem_dc, bitmap)
        try:
            ok = user32.PrintWindow(hwnd, mem_dc, 2)  # PW_RENDERFULLCONTENT
            if not ok:
                user32.PrintWindow(hwnd, mem_dc, 0)
            header = BITMAPINFOHEADER()
            header.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            header.biWidth = width
            header.biHeight = -height
            header.biPlanes = 1
            header.biBitCount = 32
            header.biCompression = 0
            buffer = ctypes.create_string_buffer(width * height * 4)
            gdi32.GetDIBits(mem_dc, bitmap, 0, height, buffer, ctypes.byref(header), 0)
            array = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)
            return Image.fromarray(array[:, :, [2, 1, 0]].copy())
        finally:
            gdi32.DeleteObject(bitmap)
            gdi32.DeleteDC(mem_dc)
            user32.ReleaseDC(hwnd, hwnd_dc)

    # ---------- 零鼠标点击 ----------
    def click(self, x: float, y: float) -> None:
        if self._ensure_window() is None:
            return
        hwnd = self.window.hwnd
        wl, wt, _, _ = self._window_rect(hwnd)
        origin = POINT(0, 0)
        user32.ClientToScreen(hwnd, ctypes.byref(origin))
        client_x = int(wl + x - origin.x)
        client_y = int(wt + y - origin.y)
        lparam = (client_y << 16) | (client_x & 0xFFFF)
        user32.PostMessageW(hwnd, WM_MOUSEMOVE, 0, lparam)
        time.sleep(0.01)
        user32.PostMessageW(hwnd, WM_LBUTTONDOWN, 1, lparam)
        time.sleep(0.02)
        user32.PostMessageW(hwnd, WM_LBUTTONUP, 0, lparam)
        time.sleep(0.03)

    # ---------- 离屏挂载（SetWindowPos 负坐标 + PrintWindow） ----------
    def is_offscreen(self) -> bool:
        if self.window is None:
            return False
        try:
            left, top, _right, _bottom = self._window_rect(self.window.hwnd)
        except Exception:
            return False
        return left < -1000 or top < -1000

    def mount_offscreen(self) -> bool:
        """把微信移到屏幕外 (-3000,-3000)：桌面零打扰，PrintWindow 仍可抓图。"""
        if self.window is None:
            return False
        hwnd = self.window.hwnd
        try:
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            left, top, right, bottom = self._window_rect(hwnd)
            width, height = right - left, bottom - top
            if width <= 0 or height <= 0:
                return False
            if not self.is_offscreen():
                self._desktop_rect = (left, top, width, height)
            # SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOZORDER
            user32.SetWindowPos(hwnd, 0, -3000, -3000, width, height, 0x0001 | 0x0010 | 0x0004)
            time.sleep(0.15)
            return self.is_offscreen()
        except Exception:
            return False

    def restore_from_offscreen(self) -> bool:
        if self.window is None:
            return False
        hwnd = self.window.hwnd
        rect = getattr(self, "_desktop_rect", None)
        try:
            if rect:
                left, top, width, height = rect
            else:
                left, top, width, height = 100, 100, 1100, 750
            user32.SetWindowPos(hwnd, 0, int(left), int(top), int(width), int(height),
                                0x0010 | 0x0004 | 0x0040)
            time.sleep(0.15)
            return not self.is_offscreen()
        except Exception:
            return False

    # ---------- 前台 ----------
    def get_foreground(self) -> int:
        return int(user32.GetForegroundWindow())

    def focus(self, window: WindowInfo) -> bool:
        hwnd = window.hwnd
        current_thread = kernel32.GetCurrentThreadId()
        foreground = user32.GetForegroundWindow()
        attached = []
        try:
            for target in (foreground, hwnd):
                thread_id = user32.GetWindowThreadProcessId(target, None)
                if thread_id and thread_id != current_thread:
                    if user32.AttachThreadInput(current_thread, thread_id, True):
                        attached.append(thread_id)
            user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_SHOWWINDOW)
            user32.SetForegroundWindow(hwnd)
            user32.BringWindowToTop(hwnd)
            user32.SwitchToThisWindow(hwnd, True)
            user32.SetWindowPos(hwnd, -2, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE)
        finally:
            for thread_id in attached:
                user32.AttachThreadInput(current_thread, thread_id, False)
        time.sleep(0.05)
        return int(user32.GetForegroundWindow()) == hwnd

    def restore_foreground(self, hwnd: int) -> None:
        if not hwnd or hwnd == getattr(self.window, "hwnd", None):
            return
        if not user32.IsWindow(hwnd):
            return
        try:
            user32.SetForegroundWindow(hwnd)
        except Exception:
            pass

    # ---------- 剪贴板 ----------
    def get_clipboard(self) -> Optional[str]:
        try:
            kernel32.GlobalLock.restype = ctypes.c_void_p
            kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
            kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
            user32.GetClipboardData.restype = ctypes.c_void_p
            user32.GetClipboardData.argtypes = [wintypes.UINT]
            if not user32.OpenClipboard(None):
                return None
            try:
                handle = user32.GetClipboardData(CF_UNICODETEXT)
                if not handle:
                    return None
                pointer = kernel32.GlobalLock(ctypes.c_void_p(handle))
                if not pointer:
                    return None
                try:
                    return ctypes.wstring_at(pointer)
                finally:
                    kernel32.GlobalUnlock(ctypes.c_void_p(handle))
            finally:
                user32.CloseClipboard()
        except Exception:
            return None

    def set_clipboard(self, text: str) -> bool:
        # 关键：HGLOBAL/指针是 64 位，必须声明 restype，否则句柄被截断 → 写入失败
        kernel32.GlobalAlloc.restype = ctypes.c_void_p
        kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
        kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
        user32.SetClipboardData.restype = ctypes.c_void_p
        user32.SetClipboardData.argtypes = [wintypes.UINT, ctypes.c_void_p]
        user32.OpenClipboard.argtypes = [wintypes.HWND]
        for _ in range(5):
            try:
                if not user32.OpenClipboard(None):
                    time.sleep(0.05)
                    continue
                try:
                    user32.EmptyClipboard()
                    data = ctypes.create_unicode_buffer(text)
                    size = ctypes.sizeof(data)
                    handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
                    if not handle:
                        time.sleep(0.05)
                        continue
                    pointer = kernel32.GlobalLock(handle)
                    if not pointer:
                        kernel32.GlobalFree(handle)
                        time.sleep(0.05)
                        continue
                    ctypes.memmove(pointer, data, size)
                    kernel32.GlobalUnlock(handle)
                    if user32.SetClipboardData(CF_UNICODETEXT, handle):
                        self._clipboard = text
                        return True
                finally:
                    user32.CloseClipboard()
            except Exception:
                time.sleep(0.05)
        return False

    # ---------- 键盘（只发键，不动鼠标） ----------
    def press_ctrl_v(self) -> None:
        user32.keybd_event(VK_CONTROL, 0, 0, 0)
        user32.keybd_event(VK_V, 0, 0, 0)
        time.sleep(0.02)
        user32.keybd_event(VK_V, 0, KEYEVENTF_KEYUP, 0)
        user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)
        time.sleep(0.04)

    def press_enter(self) -> None:
        user32.keybd_event(VK_RETURN, 0, 0, 0)
        time.sleep(0.02)
        user32.keybd_event(VK_RETURN, 0, KEYEVENTF_KEYUP, 0)
        time.sleep(0.04)

    # ---------- 结构化通道（Windows 视觉方案没有，返回 None 走 OCR） ----------
    def read_header_text(self) -> Optional[str]:
        return None

    def read_incoming_items(self):
        return None

    def last_outgoing_text(self) -> Optional[str]:
        return None

    def send_image(self, path: str) -> bool:
        """图片消息暂未实现（需要把图片写入剪贴板 CF_DIB）。"""
        return False

    def close(self) -> None:
        return None
