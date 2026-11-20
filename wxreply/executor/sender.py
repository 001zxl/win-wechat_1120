from __future__ import annotations

import difflib
import re
import time
from dataclasses import dataclass
from typing import Optional

from ..config import Config
from ..perception.chat import changed_ratio, has_new_outgoing
from ..perception.layout import Regions
from ..platform.base import Platform, SendOutcome
from ..policy import Whitelist, normalize_name
from .risk import SendRiskController

ABORT_HEADER_MISMATCH = "header_mismatch"
ABORT_NOT_WHITELIST = "not_whitelisted"
ABORT_FOCUS = "focus_failed"


def _flat(text: str) -> str:
    return re.sub(r"[\s，。！？,.!?~～]", "", text or "").lower()


@dataclass
class SendResult:
    outcome: SendOutcome
    header: str = ""
    changed: float = 0.0


class SafeSender:
    """发送安全链：白名单 → header 校验 → 前台校验 → 粘贴/回车 → 还焦点 → 差分验收。"""

    def __init__(self, platform: Platform, config: Config, whitelist: Whitelist, ocr=None) -> None:
        self.platform = platform
        self.config = config
        self.whitelist = whitelist
        self.ocr = ocr
        self.risk = SendRiskController(config.safety)

    # ---------- 读取当前会话标题 ----------
    def read_header(self, image, regions: Regions) -> str:
        structured = self.platform.read_header_text()
        if structured is not None:
            return structured
        if self.ocr is None or not self.ocr.available:
            return ""
        try:
            from PIL import Image  # noqa: F401

            x1, y1, x2, y2 = regions.header
            width = max(120, int((x2 - x1) * 0.62))
            # 标题文字常贴着 header 下边缘，向下多留 14px，避免把字裁掉
            box = (int(x1), int(y1), int(min(x2, x1 + width)), int(min(image.height, y2 + 14)))
            crop = image.crop(box)
            crop = crop.resize((crop.width * 2, crop.height * 2))
            lines = self.ocr.recognize(crop)
        except Exception:
            return ""
        # 标题栏右侧是窗口按钮（— □ ×），取左侧字高最大的一行正文
        best = None
        for line in lines:
            text = (line.text or "").strip()
            if not text:
                continue
            if line.bbox[3] - line.bbox[1] < 24:      # 2x 放大后 <12px 原高：图标/小字
                continue
            if any(token in text for token in ("×", "—", "□", "⋯", "…")):
                continue
            if text.replace(":", "").replace("：", "").isdigit():
                continue
            score = (line.bbox[3] - line.bbox[1], line.score)
            if best is None or score > best[0]:
                best = (score, text)
        if best:
            return best[1]
        try:
            return self.ocr.read_text(crop)
        except Exception:
            return ""

    # ---------- 发送主流程 ----------
    def send(self, peer: str, text: str, pre_image, regions: Regions) -> SendResult:
        runtime = self.config.runtime
        if getattr(self.platform, "passive", False):
            return SendResult(SendOutcome("aborted", "readonly_passive_macos"))
        safety = self.config.safety
        allowed, _matched = self.whitelist.matches(peer)
        if not allowed:
            return SendResult(SendOutcome("aborted", ABORT_NOT_WHITELIST))

        header = self.read_header(pre_image, regions)
        if safety.require_header_match and normalize_name(header) != normalize_name(peer):
            return SendResult(SendOutcome("aborted", ABORT_HEADER_MISMATCH), header=header)

        window = self.platform.find_window()
        if window is None:
            return SendResult(SendOutcome("aborted", "window_not_found"), header=header)
        # macOS：微信可能被隐藏/最小化，先唤醒再抢焦点
        if hasattr(self.platform, "ensure_visible"):
            try:
                self.platform.ensure_visible()
            except Exception:
                pass

        # 剪贴板原子保护：发送前快照，发送后还原用户原内容
        original_clipboard = None
        clipboard_getter = getattr(self.platform, "get_clipboard", None)
        if callable(clipboard_getter):
            try:
                original_clipboard = clipboard_getter()
            except Exception:
                original_clipboard = None

        if not runtime.dry_run:
            wait_s, reason = self.risk.plan(peer, text)
            if wait_s is None:
                return SendResult(SendOutcome("aborted", "risk_limit"), header=header)
            if wait_s > 0:
                time.sleep(wait_s)

        attempts = max(1, int(runtime.send_retries) + 1)
        last = SendResult(SendOutcome("uncertain", "no_attempt"), header=header)
        for attempt in range(attempts):
            original_foreground = self.platform.get_foreground()
            if not self.platform.focus(window):
                if hasattr(self.platform, "ensure_visible") and self.platform.ensure_visible():
                    pass
                if not self.platform.focus(window):
                    return SendResult(SendOutcome("aborted", ABORT_FOCUS), header=header)
            if safety.require_header_match and self.platform.get_foreground() != window.hwnd:
                self.platform.restore_foreground(original_foreground)
                return SendResult(SendOutcome("aborted", ABORT_FOCUS), header=header)
            try:
                input_x = regions.input[0] + max(40, (regions.input[2] - regions.input[0]) * 0.2)
                input_y = regions.input[1] + max(24, min(48, (regions.input[3] - regions.input[1]) * 0.3))
                self.platform.click(input_x, input_y)
                time.sleep(0.05)
                if not self.platform.set_clipboard(text):
                    last = SendResult(SendOutcome("failed", "clipboard_failed"), header=header)
                    break
                self.platform.press_ctrl_v()
                time.sleep(0.08)
                self.platform.press_enter()
            finally:
                self.platform.restore_foreground(original_foreground)
                if original_clipboard is not None and original_clipboard != text:
                    try:
                        time.sleep(0.05)
                        self.platform.set_clipboard(original_clipboard)
                    except Exception:
                        pass

            if not safety.verify_send:
                return SendResult(SendOutcome("unverified", "verify_disabled"), header=header)

            time.sleep(runtime.settle_s)
            post_image = self.platform.capture()
            if post_image is None:
                last = SendResult(SendOutcome("uncertain", "capture_after_send_failed"), header=header)
                continue
            changed = changed_ratio(pre_image, post_image, regions)
            if has_new_outgoing(post_image, regions) and changed >= 0.002:
                self.risk.record(peer)
                return SendResult(SendOutcome("verified", "bubble_confirmed"), header=header, changed=changed)
            if self._confirm_by_readback(text, post_image, regions):
                self.risk.record(peer)
                return SendResult(SendOutcome("verified", "readback_confirmed"), header=header, changed=changed)
            last = SendResult(SendOutcome("uncertain", f"attempt={attempt + 1} changed={changed:.4f}"),
                              header=header, changed=changed)
            if attempt < attempts - 1:
                time.sleep(0.4)
        return last

    # ---------- 读回确认（失败重试前的幂等检查） ----------
    def _confirm_by_readback(self, text: str, image, regions: Regions) -> bool:
        structured = self.platform.last_outgoing_text()
        if structured is not None:
            return self._similar(structured, text)
        if self.ocr is None or not self.ocr.available:
            return False
        try:
            lines = self.ocr.recognize(image.crop(regions.messages))
        except Exception:
            return False
        offset_x = regions.messages[0]
        for line in lines:
            x1, y1, x2, y2 = line.bbox
            line.bbox = (x1 + offset_x, y1 + regions.messages[1], x2 + offset_x, y2 + regions.messages[1])
        chat_left, _top, chat_right, _bottom = regions.chat
        width = max(1, chat_right - chat_left)
        outgoing = [line.text for line in lines if (line.center[0] - chat_left) / width >= 0.55]
        return any(self._similar(line, text) for line in outgoing[-6:])

    @staticmethod
    def _similar(a: str, b: str) -> bool:
        left, right = _flat(a), _flat(b)
        if not left or not right:
            return False
        if left in right or right in left:
            return True
        return difflib.SequenceMatcher(None, left, right).ratio() >= 0.6
