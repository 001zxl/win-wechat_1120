from __future__ import annotations

import functools
import os
import queue
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from .audit import AuditStore
from .brain.llm import ModelRouter
from .brain.memory import MemoryStore
from .brain.persona import Persona, build_system_prompt, load_persona
from .brain.rag import EmbeddingClient, KnowledgeBase
from .brain.kb_weknora import WeKnoraClient, as_chunks as weknora_chunks
from .brain.tools import ToolBox
from .brain.vision import VisionClient
from .config import Config
from .executor.sender import SafeSender
from .handoff import HandoffStore
from .perception.chat import lines_from_ocr, settle_capture
from .perception.layout import LayoutCalculator, Regions
from .perception.dual import PixelSentinel, downscale_for_ocr, lines_from_ocr_colored
from .perception.media import detect_chat_images, extract_items
from .perception.ocr import OcrEngine
from .perception.unread import click_point, find_red_badges
from .platform.base import MessageItem, Platform
from .policy import (
    Guardrails,
    RateLimiter,
    Whitelist,
    group_allows,
    in_quiet_hours,
    is_group_name,
    is_same_peer_strict,
    normalize_name,
)


def _wx_locked(method):
    """所有微信操作串行执行：自动扫描、控制台点击、人工发送共用一把锁。"""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.wx_lock:
            return method(self, *args, **kwargs)
    return wrapper


_SYSTEM_PATTERNS = (
    "拍了拍", "撤回了一条消息", "以下是新消息", "该消息已过期",
    "对方正在输入", "查看更多消息", "已读", "以下为新消息",
)


def _is_system_text(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return True
    if any(pattern in value for pattern in _SYSTEM_PATTERNS):
        return True
    # 纯时间/日期分隔
    if value.replace(":", "").replace("：", "").replace("/", "").replace(".", "").isdigit():
        return True
    return False


class Agent:
    """消息从「发现」到「发送」的完整流水线，事件通过队列推给 UI。"""

    def __init__(
        self,
        config: Config,
        platform: Platform,
        dry_run: Optional[bool] = None,
        notify: Callable[[str], None] = print,
    ) -> None:
        self.config = config
        self.platform = platform
        self.runtime = config.runtime
        self.dry_run = self.runtime.dry_run if dry_run is None else dry_run
        # 单一事实来源：启动参数决定的发送模式必须写回 runtime，
        # 否则 --live 启动时 runtime.dry_run 仍为 True，风控分支会被跳过
        self.runtime.dry_run = self.dry_run
        self.notify = notify

        self.persona: Persona = load_persona(config.resolve(config.persona))
        self.memory = MemoryStore(
            config.resolve(config.memory.db_path),
            config.memory.recent_limit,
            config.memory.summarize_after,
        )
        self.tools = ToolBox(
            config.resolve(config.tools.price_file),
            config.resolve(config.tools.leads_file),
            config.tools.enabled,
        )
        self.vision = VisionClient(config.ai)
        self.embeddings = EmbeddingClient(
            config.ai.embedding_model,
            config.ai.embedding_base_url or config.ai.base_url,
            config.ai.embedding_api_key or config.ai._fallback_key(
                config.ai.embedding_base_url or config.ai.base_url),
            config.resolve(config.rag.embedding_cache),
        )
        self.kb = KnowledgeBase(config.resolve(config.kb_dir), config.rag.mode, self.embeddings)
        # 知识库后端：local（当前）或 weknora（远程服务，未部署时自动回退本地）
        self.weknora = WeKnoraClient(
            config.rag.weknora_base_url, config.rag.weknora_api_key,
            config.rag.weknora_search_path,
        ) if str(config.rag.provider).lower() == "weknora" else None
        self.router = ModelRouter(config.ai, dry_run=self.dry_run)
        self.router.client.dry_run = not self.router.client.ready

        self.ocr = OcrEngine(backend=getattr(self.runtime, "ocr_backend", "auto"))
        self.whitelist = Whitelist(config.safety.whitelist, self.runtime.allow_group_chat)
        self.guardrails = Guardrails(
            config.safety.forbidden_keywords,
            config.safety.escalate_keywords,
            config.safety.forbidden_self_disclosure,
            config.safety.max_reply_chars,
        )
        self.limiter = RateLimiter(self.runtime.max_actions_per_min, self.runtime.cooldown_s)
        # 所有微信操作（扫描/点击/发送）共用同一把锁，避免自动线程与人手操作互相打断
        self.wx_lock = threading.RLock()
        self.sender = SafeSender(platform, config, self.whitelist, self.ocr, lock=self.wx_lock)
        saved_layout = False
        settings_path = config.resolve(config.settings_path)
        if os.path.exists(settings_path):
            try:
                import json as _json
                with open(settings_path, "r", encoding="utf-8") as handle:
                    saved_layout = bool((_json.load(handle) or {}).get("layout"))
            except Exception:
                saved_layout = False
        if not saved_layout and hasattr(platform, "layout_preset"):
            try:
                for key, value in platform.layout_preset().items():
                    setattr(config.layout, key, float(value))
            except Exception:
                pass
        self.layout = LayoutCalculator(
            config.layout.nav_ratio,
            config.layout.list_ratio,
            config.layout.header_ratio,
            config.layout.input_ratio,
            config.layout.overrides,
        )
        self.audit = AuditStore(
            config.resolve(config.audit.db_path),
            config.resolve(config.audit.evidence_dir),
            config.audit.save_screenshots,
        )
        self.media_dir = config.resolve("data/media")
        os.makedirs(self.media_dir, exist_ok=True)
        self.avatar_dir = config.resolve("data/avatars")
        os.makedirs(self.avatar_dir, exist_ok=True)
        # 人工接管（持久化）：接管期间该会话不再自动回复，直到显式恢复
        self.handoff = HandoffStore(config.resolve(config.handoff_path))

        # ---- 运行状态 ----
        self.events: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1000)
        self.paused = threading.Event()
        self._stopped = threading.Event()
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._pending_order: List[str] = []
        self._replied: Dict[str, str] = {}
        self._conv: Dict[str, Dict[str, Any]] = {}
        self._instruction: str = ""
        self._summarizing: set = set()
        self._last_capture_error: float = 0.0
        self._deep_collect: Optional[List[str]] = None
        self._view_history: Dict[str, List[Tuple[str, str]]] = {}
        self._last_click: Dict[str, float] = {}
        self._sentinel = PixelSentinel()
        self._peer_positions: Dict[str, Dict[str, float]] = {}
        self._row_index: List[Tuple[str, float]] = []   # 可见行 (名字, 全局 y)：红点行 → 会话名
        self._event_seq: Dict[str, int] = {}            # 每个会话的事件轮次
        self._event_state: Dict[str, Dict[str, Any]] = {}
        self._deep_page: Optional[int] = None
        self._last_error: str = ""
        self._reading: set = set()
        self.last_scan_ts: float = 0.0
        self.last_cycle_error: str = ""
        # 重启后恢复待重试队列（与发送台账同库）
        self._load_retries()
        if self.handoff.corrupt:
            self.emit("error", message="人工接管状态文件损坏：自动发送已暂停，请检查 data/handoff.json")

    # ---------- 运行控制 ----------
    def set_dry_run(self, value: bool) -> None:
        self.dry_run = bool(value)
        # 配置里的 runtime.dry_run 也要同步，否则发送限频等按配置判断的分支会被跳过
        try:
            self.runtime.dry_run = self.dry_run
        except Exception:
            pass
        self.router.set_dry_run(not self.router.client.ready)
        self.emit("status", message=f"真实发送={'开' if not self.dry_run else '关（仅草稿）'}")

    def set_model(self, model_id: str) -> bool:
        ok = self.router.switch(model_id)
        self.router.client.dry_run = not self.router.client.ready
        self.emit("model", message=f"模型已切换为 {self.router.current}", model=self.router.current, ok=ok)
        return ok

    def current_model(self) -> str:
        return self.router.current

    def model_options(self) -> List[Dict[str, object]]:
        result = []
        for option in self.router.options:
            item = dict(option.as_dict())
            item["has_key"] = bool(option.api_key)
            result.append(item)
        return result

    def layout_ratios(self) -> Dict[str, float]:
        return {
            "nav_ratio": self.config.layout.nav_ratio,
            "list_ratio": self.config.layout.list_ratio,
            "header_ratio": self.config.layout.header_ratio,
            "input_ratio": self.config.layout.input_ratio,
        }

    def capture_preview(self) -> Optional[Tuple[Any, Regions]]:
        image = self.platform.capture()
        if image is None:
            return None
        return image, self.layout.regions(image.width, image.height)

    def apply_settings(self, data: Dict[str, Any]) -> None:
        """控制台保存设置后热更新：Key / 白名单 / 布局 / 视觉与向量模型。"""
        keys = data.get("api_keys") or {}
        if isinstance(keys, dict):
            self.config.ai.model_keys = {str(k): str(v) for k, v in keys.items() if v}
        whitelist = data.get("whitelist")
        if isinstance(whitelist, list):
            self.config.safety.whitelist = [str(item) for item in whitelist if str(item).strip()]
            self.whitelist = Whitelist(self.config.safety.whitelist, self.runtime.allow_group_chat)
        layout = data.get("layout") or {}
        if isinstance(layout, dict):
            for key in ("nav_ratio", "list_ratio", "header_ratio", "input_ratio"):
                if key in layout:
                    try:
                        setattr(self.config.layout, key, float(layout[key]))
                    except (TypeError, ValueError):
                        continue
            self.layout = LayoutCalculator(
                self.config.layout.nav_ratio, self.config.layout.list_ratio,
                self.config.layout.header_ratio, self.config.layout.input_ratio,
                self.config.layout.overrides,
            )
        if isinstance(data.get("vision_model"), str):
            self.config.ai.vision_model = data["vision_model"]
            self.vision = VisionClient(self.config.ai)
        if isinstance(data.get("embedding_model"), str):
            self.config.ai.embedding_model = data["embedding_model"]
            self.embeddings = EmbeddingClient(
                self.config.ai.embedding_model, self.config.ai.base_url, self.config.ai.api_key,
                self.config.resolve(self.config.rag.embedding_cache),
            )
            self.kb = KnowledgeBase(self.config.resolve(self.config.kb_dir),
                                    self.config.rag.mode, self.embeddings)
        # 重建模型路由（保留当前模型）
        current = self.router.current
        self.router = ModelRouter(self.config.ai, dry_run=self.dry_run)
        if not self.router.switch(current):
            self.router.switch(self.config.ai.model)
        self.router.client.dry_run = not self.router.client.ready
        self.emit("settings", message="设置已应用")

    def set_instruction(self, text: str) -> None:
        self._instruction = (text or "").strip()

    def pause(self) -> None:
        self.paused.set()
        self.emit("status", message="已暂停自动回复")

    def resume(self) -> None:
        self.paused.clear()
        self.emit("status", message="已恢复自动回复")

    def stop(self) -> None:
        self._stopped.set()
        self.paused.clear()

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    # ---------- 事件 ----------
    def emit(self, kind: str, **fields: Any) -> None:
        event = {"type": kind, "ts": time.time(), **fields}
        try:
            self.events.put_nowait(event)
        except queue.Full:
            pass
        summary = fields.get("message") or fields.get("reason") or fields.get("status") or ""
        peer = fields.get("peer", "")
        if kind in ("draft", "sent", "escalate", "error", "tool", "model"):
            self.notify(f"[{kind}] {peer} {summary}".strip())

    # ---------- 主循环 ----------
    def run(self, cycles: Optional[int] = None, once: bool = False) -> None:
        self.emit("status", message=f"启动：platform={self.platform.name} dry_run={self.dry_run} "
                                    f"model={self.router.current} persona={self.persona.display_name} "
                                    f"kb={self.kb.size} 记忆={self.config.memory.db_path}")
        if getattr(self.runtime, "offscreen", False) and hasattr(self.platform, "mount_offscreen"):
            try:
                if self.platform.mount_offscreen():
                    self.emit("status", message="微信已移入离屏空间（PrintWindow 抓图）")
            except Exception:
                pass
        iterations = 1 if once else cycles
        executed = 0
        try:
            while not self.stopped:
                if iterations is not None and executed >= iterations:
                    return
                if self.paused.is_set():
                    now = time.time()
                    if now - getattr(self, "_last_sync", 0) >= 2.0:
                        self._last_sync = now
                        self.sync_conversations()
                    time.sleep(0.2)
                    continue
                self.cycle()
                executed += 1
                if iterations is not None and executed >= iterations:
                    return
                time.sleep(self.runtime.poll_interval_s)
        except KeyboardInterrupt:
            self.emit("status", message="收到 Ctrl+C，退出")
        finally:
            self.platform.close()

    # ---------- 单轮 ----------
    @_wx_locked
    def cycle(self) -> int:
        if getattr(self.platform, "passive", False):
            if not getattr(self, "_passive_notice", False):
                self._passive_notice = True
                self.emit("status", message="只读模式：不点击、不滚动、不发送；请在微信手动切换会话")
            return 0
        if in_quiet_hours(self.runtime.quiet_hours):
            return 0
        image = self.platform.capture()
        if image is None:
            now = time.time()
            if now - self._last_capture_error > 10:
                self._last_capture_error = now
                self.emit("error", message="微信窗口不可用（未运行/已隐藏），已暂停自动扫描；恢复后点「继续」")
                self.pause()
            return 0
        regions = self.layout.regions(image.width, image.height)
        self.last_scan_ts = time.time()
        try:
            self.sync_conversations()
        except Exception as exc:
            self.last_cycle_error = str(exc)
        # 先补做失败重试 / 限频后补发，再扫描新未读
        try:
            self.drive_pending_work()
        except Exception as exc:
            self.emit("error", message=f"重试队列异常：{exc}")
        if not self._pending and not self._sentinel.should_scan(image, regions):
            return 0
        targets_fn = getattr(self.platform, "unread_targets", None)
        if callable(targets_fn):
            processed = 0
            for _name, x, y in (targets_fn() or [])[: self.runtime.max_unread_per_cycle]:
                badge = type("B", (), {"center": (x, y)})()
                if self._handle_unread(badge, regions):
                    processed += 1
            return processed
        badges = find_red_badges(image, regions)
        if not badges:
            # 当前打开的会话可能显示绿色选中徽标，不产生红点；白名单内也处理
            if self.process_current() is not None:
                return 1
            return 0
        self.emit("cycle", badges=len(badges), dry_run=self.dry_run)

        # 同一行可能检出多个红点（头像徽标 + 免打扰图标），先按 y 合并
        merged = []
        for badge in sorted(badges, key=lambda item: item.center[1]):
            if merged and abs(badge.center[1] - merged[-1].center[1]) <= 40:
                continue
            merged.append(badge)
        badges = merged

        # 有 OCR 时先用会话名预检：只点白名单会话，避免误标已读
        candidates = []
        names = [] if getattr(self, "_row_index", None) else self._session_names(image, regions)
        for badge in badges:
            # 先用「可见行 ↔ 名字」索引定位红点属于哪一行，避免把消息预览当会话名
            name = self._match_name(badge, getattr(self, "_badge_index", []), tolerance=40) or (
                self._name_for_badge(image, regions, badge)
                or (self._match_name(badge, names) if names else None)
            )
            if name:
                known = name in self._conv
                skip, reason = self._should_skip_by_name(name)
                if skip:
                    if known:
                        self._touch(name, status="已跳过")
                    self.audit.record(peer=name, action="skip", status="precheck_skip", detail=reason)
                    self.emit("skip", peer=name, reason=reason)
                    continue
            elif getattr(self.platform, "strict_name_precheck", False):
                self.audit.record(peer="", action="skip", status="precheck_unknown",
                                  detail="未识别会话名，跳过以免误点")
                self.emit("skip", peer="", reason="未识别会话名（未点开）")
                continue
            if name and time.time() - self._last_click.get(name, 0) < 15:
                self.emit("skip", peer=name, reason="点击冷却中（15s）")
                continue
            if name:
                self._last_click[name] = time.time()
            candidates.append(badge)
        processed = 0
        for badge in candidates[: self.runtime.max_unread_per_cycle]:
            if self._handle_unread(badge, regions):
                processed += 1
        if processed == 0 and self.process_current() is not None:
            processed = 1
        return processed

    # ---------- 会话/历史 ----------
    def _touch(self, peer: str, status: str = "", unread: Optional[int] = None, text: str = "") -> None:
        entry = self._conv.setdefault(
            peer, {"peer": peer, "unread": 0, "last_ts": 0.0, "last_text": "", "status": ""}
        )
        if not entry.get("avatar_path"):
            generated = self._ensure_avatar(peer)
            if generated:
                entry["avatar_path"] = generated
        entry["last_ts"] = time.time()
        if status:
            entry["status"] = status
        if unread is not None:
            entry["unread"] = unread
        if text:
            entry["last_text"] = text

    def mark_read(self, peer: str) -> None:
        """点开会话即视为已读：清掉控制台红点，避免点完还挂 1。"""
        if not peer:
            return
        entry = self._conv.get(peer)
        if entry is not None:
            entry["unread"] = 0

    def conversation(self, peer: str) -> List[Dict[str, str]]:
        snapshot = self._view_history.get(peer)
        if snapshot is not None:
            return [{"dir": direction, "kind": "text", "text": text}
                    for direction, text in snapshot]
        rows = self.memory.recent(peer, 30)
        if rows:
            return [{"dir": direction, "kind": kind, "text": text}
                    for _id, direction, kind, text in rows]
        return [{"dir": "sys", "kind": "text",
                 "text": "（还没有读取到该会话记录：请在微信里打开聊天，然后点顶部「读取当前会话」）"}]

    def _read_chat_page(self, image, regions: Regions):
        """读取当前可见聊天区：返回 [(dir, text)] 列表。"""
        try:
            crop = image.crop(regions.messages)
            small, scale = downscale_for_ocr(crop)
            lines = self.ocr.recognize(small)
        except Exception:
            return []
        offset_x, offset_y = regions.messages[0], regions.messages[1]
        for line in lines:
            x1, y1, x2, y2 = line.bbox
            line.bbox = (x1 / scale + offset_x, y1 / scale + offset_y,
                         x2 / scale + offset_x, y2 / scale + offset_y)
        lines = [line for line in lines if not _is_system_text(line.text)]
        media_items = detect_chat_images(image, regions, self.media_dir)
        image_boxes = [item.bbox for item in media_items if item.kind == "image" and item.bbox]

        def inside_image(line) -> bool:
            cx, cy = line.center
            return any(x1 - 4 <= cx <= x2 + 4 and y1 - 4 <= cy <= y2 + 4
                       for x1, y1, x2, y2 in image_boxes)

        lines = [line for line in lines if not inside_image(line)]
        chat_lines = lines_from_ocr_colored(image, regions, lines, scale=1.0)
        messages = []
        for line in chat_lines:
            text = line.text.strip()
            if text:
                messages.append((line.direction, text))
        for _item in media_items:
            messages.append(("in", "[图片]"))
        return messages

    def select_conversation(self, peer: str) -> bool:
        """点开指定会话并读取聊天记录（可见列表优先）。"""
        self._reading.add(peer)
        self._view_history[peer] = [("sys", "正在读取聊天记录…")]
        try:
            return self._select_conversation_inner(peer)
        finally:
            self._reading.discard(peer)

    def _select_conversation_inner(self, peer: str) -> bool:
        if not self.open_conversation(peer):
            self._view_history[peer] = [
                ("in", "（无法自动打开该会话：请在微信里手动打开，然后点「读取当前会话」）")
            ]
            self._touch(peer, status="打开失败")
            return False
        return self.read_conversation(peer, open_first=False)

    def simulate_incoming(self, peer: str, text: str, auto: bool = True) -> Optional[Dict[str, Any]]:
        """项目内模拟客户来消息：走完整思考/回复链路，绝不操作真实微信。"""
        text = (text or "").strip()
        if not text:
            return None
        peer = (peer or "模拟客户").strip()
        if not peer.startswith("【模拟】"):
            peer = "【模拟】" + peer
        # 接管期间模拟会话也不再自动回复，保证演示和正式流程一致
        if self.handoff.is_taken(peer):
            label = "（人工接管中，未自动回复）"
            self._touch(peer, status="人工接管", unread=0, text=text)
            self.emit("simulate", peer=peer, incoming=text, reply=label, auto=False,
                      model="handoff", latency_ms=0, tools=[], escalated=True)
            return {"peer": peer, "incoming": text, "reply": label, "auto": False,
                    "model": "handoff", "latency_ms": 0, "tools": [], "escalated": True}
        history = self.memory.history_lines(peer, self.config.memory.recent_limit)
        profile = self.memory.profile(peer)
        decision = self.guardrails.assess_incoming(text)
        self.memory.add(peer, "in", "text", text)
        view = self._view_history.setdefault(peer, [])
        view.append(("in", text))
        if decision.action == "escalate":
            label = f"（转人工）{decision.reason}"
            view.append(("out", label))
            self.memory.add(peer, "out", "text", label)
            self._touch(peer, status="转人工", unread=0, text=label)
            self.emit("simulate", peer=peer, incoming=text, reply=label, auto=True,
                      model="guardrail", latency_ms=0, tools=[], escalated=True)
            return {"peer": peer, "incoming": text, "reply": label, "auto": True,
                    "model": "guardrail", "latency_ms": 0, "tools": [],
                    "escalated": True}
        tool_results = self.tools.prefetch(text)
        for result in tool_results:
            self.emit("tool", name=result.name, summary=result.summary, ok=result.ok,
                      simulated=True)
        reply, model, latency = self._generate(
            [text], text, history, profile, tool_results, is_group_name(peer), peer=peer
        )
        if reply is None:  # 偶发网络/限流，重试一次
            reply, model, latency = self._generate(
                [text], text, history, profile, tool_results, is_group_name(peer)
            )
        if reply is None:
            return None
        if auto:
            view.append(("out", reply))
            self.memory.add(peer, "out", "text", reply)
            status = "模拟已回复"
        else:
            self._pending[peer] = {
                "peer": peer, "reply": reply, "incoming": text, "image": None,
                "regions": None, "model": model, "latency_ms": latency,
                "incoming_items": [text], "history": history, "profile": profile,
                "tool_results": [r.as_dict() for r in tool_results],
                "is_group": is_group_name(peer), "simulated": True,
            }
            if peer in self._pending_order:
                self._pending_order.remove(peer)
            self._pending_order.append(peer)
            status = "模拟草稿"
        self._touch(peer, status=status, unread=0, text=reply)
        self._conv[peer]["simulated"] = True
        self.audit.record(peer=peer, action="simulate", status=status,
                          incoming=text, reply=reply, model=model, latency_ms=latency,
                          detail="项目内模拟，未操作真实微信")
        self.emit("simulate", peer=peer, incoming=text, reply=reply, auto=auto,
                  model=model, latency_ms=latency,
                  tools=[r.name for r in tool_results])
        return {"peer": peer, "incoming": text, "reply": reply, "auto": auto,
                "model": model, "latency_ms": latency,
                "tools": [r.summary for r in tool_results]}

    def handle_inbound(self, peer: str, text: str, auto: bool = True) -> Optional[Dict[str, Any]]:
        """外部 IM（LINE/Telegram/Webhook）入站：走同一套人设/知识库/工具/护栏。"""
        text = (text or "").strip()
        if not text:
            return None
        peer = (peer or "unknown").strip()
        history = self.memory.history_lines(peer, self.config.memory.recent_limit)
        profile = self.memory.profile(peer)
        decision = self.guardrails.assess_incoming(text)
        self.memory.add(peer, "in", "text", text)
        view = self._view_history.setdefault(peer, [])
        view.append(("in", text))
        if decision.action == "escalate":
            reply = f"（转人工）{decision.reason}"
            view.append(("out", reply))
            self.memory.add(peer, "out", "text", reply)
            self._touch(peer, status="转人工", unread=0, text=reply)
            self.emit("inbound", peer=peer, incoming=text, reply=reply, escalated=True)
            return {"peer": peer, "incoming": text, "reply": reply, "escalated": True}
        tool_results = self.tools.prefetch(text)
        reply, model, latency = self._generate(
            [text], text, history, profile, tool_results, is_group_name(peer), peer=peer
        )
        if reply is None:
            return None
        self._touch(peer, status="外部消息已回复" if auto else "外部消息草稿",
                    unread=0, text=reply)
        if auto:
            view.append(("out", reply))
            self.memory.add(peer, "out", "text", reply)
        else:
            self._pending[peer] = {"peer": peer, "reply": reply, "incoming": text,
                                   "image": None, "regions": None, "model": model,
                                   "latency_ms": latency, "incoming_items": [text],
                                   "history": history, "profile": profile,
                                   "tool_results": [r.as_dict() for r in tool_results],
                                   "is_group": is_group_name(peer), "external": True}
            if peer in self._pending_order:
                self._pending_order.remove(peer)
            self._pending_order.append(peer)
        self.audit.record(peer=peer, action="inbound", status="已回复" if auto else "草稿",
                          incoming=text, reply=reply, model=model, latency_ms=latency,
                          detail="外部 IM 入站")
        self.emit("inbound", peer=peer, incoming=text, reply=reply, auto=auto,
                  model=model, latency_ms=latency)
        return {"peer": peer, "incoming": text, "reply": reply, "auto": auto,
                "model": model, "latency_ms": latency}

    @_wx_locked
    def read_current(self) -> Optional[str]:
        """读取微信当前已经打开的会话（最可靠，不尝试切换会话）。"""
        image = self.platform.capture()
        if image is None:
            return None
        regions = self.layout.regions(image.width, image.height)
        peer = self.sender.read_header(image, regions)
        if not peer:
            return None
        if self.read_conversation(peer, open_first=False):
            return peer
        return None

    @_wx_locked
    def read_conversation(self, peer: str, open_first: bool = True) -> bool:
        """读取聊天历史；open_first=False 表示只读当前打开会话。"""
        if open_first and not self.open_conversation(peer):
            self.emit("error", message=f"打开会话失败：{peer}")
            self._view_history[peer] = [("in", "（无法自动打开该会话，请在微信中手动打开后点「读取当前会话」）")]
            return False
        collected: List[Tuple[str, str]] = []
        seen = set()
        try:
            image = self.platform.capture()
            if image is None:
                return False
            regions = self.layout.regions(image.width, image.height)
            # 先滚到底（刚点开的会话本来就在底部，滚几次就够）
            if callable(getattr(self.platform, "scroll", None)):
                x = regions.chat[0] + (regions.chat[2] - regions.chat[0]) * 0.5
                y = regions.messages[1] + (regions.messages[3] - regions.messages[1]) * 0.5
                self.platform.scroll(x, y, -8, 3)
                time.sleep(0.2)
            for page in range(2):
                image = self.platform.capture()
                if image is None:
                    break
                regions = self.layout.regions(image.width, image.height)
                page_messages = self._read_chat_page(image, regions)
                new_count = 0
                for direction, text in page_messages:
                    key = (direction, text)
                    if key in seen:
                        continue
                    seen.add(key)
                    collected.insert(0, (direction, text))
                    new_count += 1
                if page == 0 and collected:
                    # 首屏先显示，别让用户干等
                    self._view_history[peer] = list(reversed(collected))[-80:]
                    self.emit("read", peer=peer, count=len(collected), partial=True)
                if page >= 1 and new_count == 0:
                    break
                if callable(getattr(self.platform, "scroll", None)):
                    x = regions.chat[0] + (regions.chat[2] - regions.chat[0]) * 0.5
                    y = regions.messages[1] + (regions.messages[3] - regions.messages[1]) * 0.5
                    self.platform.scroll(x, y, 6, 1)  # 向上翻一页
                    time.sleep(0.2)
        except Exception as exc:
            self.emit("error", message=f"读取聊天记录失败：{exc}")
        if not collected:
            collected = [("in", "（未读取到聊天记录：可能为空会话、纯图片或 OCR 失败）")]
        history = list(reversed(collected))[-80:]   # 按时间正序（早 → 晚）
        self._view_history[peer] = history
        self._touch(peer, status="已读取", unread=0,
                    text=history[-1][1] if history else "")
        self.emit("read", peer=peer, count=len(history))
        return True

    def conversations(self) -> List[Dict[str, Any]]:
        return sorted(self._conv.values(),
                      key=lambda item: (item.get("order", 10**9), -item["last_ts"]))

    @property
    def pending(self) -> Optional[Dict[str, Any]]:
        if not self._pending_order:
            return None
        return self._pending.get(self._pending_order[-1])

    def pending_for(self, peer: str) -> Optional[Dict[str, Any]]:
        return self._pending.get(peer)

    # ---------- 会话同步 ----------
    def _scroll_session_list(self, delta: int, times: int = 1) -> None:
        scroll = getattr(self.platform, "scroll", None)
        if not callable(scroll):
            return
        image = self.platform.capture()
        if image is None:
            return
        regions = self.layout.regions(image.width, image.height)
        left, _top, right, _bottom = regions.session_list
        x = left + (right - left) * 0.5
        y = image.height * 0.45
        try:
            scroll(x, y, delta, times)
        except Exception:
            pass
        time.sleep(0.3)

    def _session_names(self, image, regions: Regions) -> List[Tuple[str, float]]:
        if not self.ocr.available:
            return []
        try:
            lines = self.ocr.recognize(image.crop(regions.session_list))
        except Exception:
            return []
        names = []
        for line in lines:
            text = (line.text or "").strip()
            if not text or line.score < 0.35:
                continue
            height = line.bbox[3] - line.bbox[1]
            if height < 20 or len(text) > 30:
                continue
            if any(token in text for token in ("…", "：", "［", "］", "[", "]", "条]", "拍了拍",
                                                "..", "．．", "。。", "⋯", "搜索")):
                continue
            if text.replace(":", "").replace("：", "").isdigit():
                continue
            names.append((text, (line.bbox[1] + line.bbox[3]) / 2.0))
        return names

    @staticmethod
    def _clean_peer_name(name: str) -> str:
        import re as _re
        value = (name or "").strip()
        value = _re.sub(r"[\s·.．…⋯/]+$", "", value)
        return _re.sub(r"\s+", "", value).lower()

    def _canonical_peer(self, name: str) -> str:
        import difflib
        cleaned = self._clean_peer_name(name)
        if not cleaned:
            return name
        best, best_ratio = None, 0.0
        for existing in self._conv.keys():
            other = self._clean_peer_name(existing)
            if not other:
                continue
            if other == cleaned:
                return existing
            ratio = difflib.SequenceMatcher(None, other, cleaned).ratio()
            if ratio > best_ratio:
                best_ratio, best = ratio, existing
        if best and best_ratio >= 0.86:
            return best
        return name

    @staticmethod
    def _detect_avatar(image, regions: Regions, name_x: float, row_y: float, row_h: float):
        """在名字左侧区域找出最大的近似方形连通块，作为头像框。"""
        import numpy as _np
        from collections import deque

        session_left = int(regions.session_list[0])
        session_top = int(regions.session_list[1])
        left = session_left + 6
        right = int(min(session_left + name_x - 4, left + 118))
        top = int(max(0, session_top + row_y - row_h * 2.4))
        bottom = int(min(image.height, session_top + row_y + row_h * 2.4))
        if right - left < 30 or bottom - top < 40:
            return None
        band = image.crop((left, top, right, bottom)).convert("RGB")
        array = _np.array(band, dtype=_np.int16)
        border = _np.concatenate([
            array[0:2].reshape(-1, 3), array[-2:].reshape(-1, 3),
            array[:, 0:2].reshape(-1, 3), array[:, -2:].reshape(-1, 3),
        ])
        background = _np.median(border, axis=0)
        mask = _np.abs(array - background).sum(axis=2) > 34
        red = (array[:, :, 0] >= 140) & (array[:, :, 0] - array[:, :, 1] >= 30) \
            & (array[:, :, 0] - array[:, :, 2] >= 30)
        mask &= ~red
        height, width = mask.shape
        seen = _np.zeros_like(mask, dtype=bool)
        best = None
        for sy in range(height):
            if not mask[sy].any():
                continue
            for sx in _np.where(mask[sy] & ~seen[sy])[0]:
                if seen[sy, sx]:
                    continue
                queue = deque([(int(sy), int(sx))])
                seen[sy, sx] = True
                min_x = max_x = int(sx)
                min_y = max_y = int(sy)
                pixels = 0
                while queue:
                    cy, cx = queue.popleft()
                    pixels += 1
                    if cx < min_x: min_x = cx
                    if cx > max_x: max_x = cx
                    if cy < min_y: min_y = cy
                    if cy > max_y: max_y = cy
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            queue.append((ny, nx))
                w = max_x - min_x + 1
                h = max_y - min_y + 1
                if pixels < 350 or w < 36 or h < 36 or w > 130 or h > 130:
                    continue
                aspect = w / float(h)
                if aspect < 0.65 or aspect > 1.55:
                    continue
                score = pixels
                if best is None or score > best[0]:
                    best = (score, min_x, min_y, max_x, max_y)
        if best is None:
            return None
        _score, x1, y1, x2, y2 = best
        pad = 3
        gx1 = max(0, left + x1 - pad)
        gy1 = max(0, top + y1 - pad)
        size = max(x2 - x1 + 1, y2 - y1 + 1) + pad * 2
        piece = image.crop((gx1, gy1, min(image.width, gx1 + size), min(image.height, gy1 + size)))
        if float(_np.array(piece.convert("RGB")).std()) < 4.0:
            return None
        return piece

    def _ensure_avatar(self, name: str) -> Optional[str]:
        import hashlib as _hash
        path = os.path.join(self.avatar_dir, _hash.sha1(name.encode("utf-8")).hexdigest() + ".png")
        if os.path.exists(path):
            return path
        try:
            from PIL import ImageDraw
            digest = _hash.md5(name.encode("utf-8")).hexdigest()
            color = (int(digest[0:2], 16) % 160 + 80,
                     int(digest[2:4], 16) % 160 + 80,
                     int(digest[4:6], 16) % 160 + 80)
            image = Image.new("RGB", (64, 64), color)
            draw = ImageDraw.Draw(image)
            draw.text((24, 22), (name.strip()[:1] or "?"), fill=(255, 255, 255))
            image.save(path)
            return path
        except Exception:
            return None

    def _apply_order(self, names: List[str]) -> None:
        ordered = []
        for name in names:
            if name in self._conv and name not in ordered:
                ordered.append(name)
        rest = sorted([name for name in self._conv if name not in ordered],
                      key=lambda name: self._conv[name].get("order", 10**9))
        for index, name in enumerate(ordered + rest):
            self._conv[name]["order"] = index

    def _name_for_badge(self, image, regions: Regions, badge) -> Optional[str]:
        if not self.ocr.available:
            return None
        left = regions.session_list[0] + 70
        right = regions.session_list[2] - 10
        top = max(0, int(badge.center[1]) - 70)
        bottom = min(image.height, int(badge.center[1]) + 35)
        if right - left < 40 or bottom - top < 30:
            return None
        try:
            crop = image.crop((left, top, right, bottom))
            crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
            lines = self.ocr.recognize(crop)
        except Exception:
            return None
        best = None
        for line in lines:
            text = (line.text or "").strip()
            if line.score < 0.35 or not text:
                continue
            height = line.bbox[3] - line.bbox[1]
            if height < 16:
                continue
            if any(token in text for token in ("…", "：", "[", "]", "条]", "搜索", "拍了拍")):
                continue
            if any(token in text for token in _SYSTEM_PATTERNS):
                continue
            if text.replace(":", "").replace("：", "").isdigit():
                continue
            if best is None or height > best[0]:
                best = (height, text)
        return best[1] if best else None

    def _match_name(self, badge, names: List[Tuple[str, float]],
                    tolerance: float = 90.0) -> Optional[str]:
        if not names:
            return None
        _cx, cy = badge.center
        best = min(names, key=lambda item: abs(item[1] - cy))
        return best[0] if abs(best[1] - cy) <= tolerance else None

    def _should_skip_by_name(self, name: str) -> Tuple[bool, str]:
        if is_group_name(name):
            if not self.runtime.allow_group_chat:
                return True, "群聊自动回复未开启（未点开）"
            if self.runtime.group_whitelist:
                normalized = normalize_name(name)
                matched = any(normalize_name(item) in normalized or normalized in normalize_name(item)
                              for item in self.runtime.group_whitelist if item.strip())
                if not matched:
                    return True, "群不在群白名单（未点开）"
            return False, ""
        allowed, _matched = self.whitelist.matches(name)
        return (False, "") if allowed else (True, "不在白名单（未点开）")

    def _read_items(self, image, regions: Regions) -> List[MessageItem]:
        structured = self.platform.read_incoming_items()
        if structured is not None:
            items = list(structured)
        else:
            if not self.ocr.available:
                self.emit("error", message="OCR 不可用")
                return []
            try:
                lines = self.ocr.recognize(image.crop(regions.messages))
            except Exception as exc:
                self.emit("error", message=f"OCR 失败: {exc}")
                return []
            offset_x, offset_y = regions.messages[0], regions.messages[1]
            for line in lines:
                x1, y1, x2, y2 = line.bbox
                line.bbox = (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
            lines = [line for line in lines if not _is_system_text(line.text)]
            chat_lines = lines_from_ocr(lines, regions)
            last_out = -1
            for index, line in enumerate(chat_lines):
                if line.direction == "out":
                    last_out = index
            items = [MessageItem(kind="text", text=line.text)
                     for line in chat_lines[last_out + 1:] if line.direction == "in"]
            media_items = extract_items(image, regions, lines, self.media_dir,
                                        detect_media=self.vision.available)
            items.extend(item for item in media_items if item.kind != "text")
        for item in items:
            if item.kind == "image":
                if self.vision.available and item.media_path:
                    item.description = self.vision.describe(item.media_path)
                elif not item.description:
                    item.description = "（未启用视觉模型，无法读图）"
                self.emit("media", media_kind=item.kind, description=item.description,
                          media_path=item.media_path)
        return items

    def _find_avatar_boxes(self, image, regions: Regions) -> List[Tuple[int, int, int, int]]:
        """在会话列表左侧找头像方块（约 36~44px），返回图片全局坐标。"""
        import numpy as np

        left = max(0, int(regions.session_list[0]))
        top = max(0, int(regions.session_list[1]))
        right = min(image.width, int(regions.session_list[2]))
        bottom = min(image.height, int(regions.session_list[3]))
        if right - left < 60 or bottom - top < 80:
            return []
        crop = image.crop((left, top, right, bottom)).convert("RGB")
        array = np.asarray(crop).astype(np.int16)
        # 背景色：取列表右侧空白列的中位数
        sample = array[:, max(0, array.shape[1] - 40):, :].reshape(-1, 3)
        background = np.median(sample, axis=0)
        mask = (np.abs(array - background).sum(axis=2) > 40).astype(np.uint8)
        boxes: List[Tuple[int, int, int, int]] = []
        try:
            import cv2

            count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
                mask, 8, cv2.CV_32S)
        except Exception:
            return boxes
        for index in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[index])
            if not (24 <= width <= 80 and 24 <= height <= 80):
                continue
            if x > 120 or area < 400:          # 头像贴着列表左边缘
                continue
            if not (0.65 <= width / max(1, height) <= 1.55):
                continue
            boxes.append((left + x, top + y, left + x + width, top + y + height))
        return boxes

    @staticmethod
    def _match_avatar(boxes, name_x: float, name_top: float, name_h: float):
        """按垂直位置把头像框配给某一行；尺寸异常时用统一头像列兜底。"""
        if not boxes:
            return None
        widths = sorted(box[2] - box[0] for box in boxes)
        sizes = sorted(box[3] - box[1] for box in boxes)
        size = widths[len(widths) // 2]
        height = sizes[len(sizes) // 2]
        lefts = sorted(box[0] for box in boxes)
        left = lefts[len(lefts) // 2]
        best, best_dy = None, 1e9
        for box in boxes:
            x1, y1, x2, y2 = box
            if x1 >= name_x:                   # 头像必须在名字左边
                continue
            dy = abs(y1 - name_top)
            if dy < best_dy:
                best_dy, best = dy, box
        if best is not None and best_dy <= max(14.0, name_h):
            width, box_height = best[2] - best[0], best[3] - best[1]
            if (abs(width - size) <= size * 0.35 and abs(box_height - height) <= height * 0.35):
                return best
        # 兜底：沿用检测到的头像列与尺寸，只按名字行重算 y（避免半截头像）
        if not (24 <= size <= 80) or left >= name_x:
            return None
        return (left, int(name_top), left + size, int(name_top) + max(size, height))

    def _detect_unread(self, area, red, name_x: float, name_y1: float) -> int:
        """按名字位置反推头像上的红色未读数角标。"""
        import numpy as np

        avatar_left = int(name_x - 87)
        bx1, bx2 = max(0, avatar_left + 42), min(area.shape[1], avatar_left + 92)
        by1, by2 = max(0, int(name_y1) - 30), min(area.shape[0], int(name_y1) + 10)
        if bx2 <= bx1 or by2 <= by1:
            return 0
        patch = red[by1:by2, bx1:bx2]
        if int(patch.sum()) < 20:
            return 0
        rows_idx, cols_idx = np.where(patch)
        height = int(rows_idx.max() - rows_idx.min() + 1)
        width = int(cols_idx.max() - cols_idx.min() + 1)
        if not (8 <= width <= 30 and 8 <= height <= 30):
            return 0
        # 不再对每个红点单独跑一次 OCR 数字（一次约 1 秒，是同步慢的主因），有红点即算未读
        return 1

    @_wx_locked
    def sync_conversations(self, deep: bool = False) -> int:
        snapshot = getattr(self.platform, "session_snapshot", None)
        if callable(snapshot):
            rows = snapshot() or []
            order_names = []
            for row in rows:
                name = str(row.get("name") or "")
                if not name:
                    continue
                entry = self._conv.get(name)
                if entry is None:
                    self._touch(name, status="未读" if row.get("unread") else "空闲",
                                text=str(row.get("preview") or ""))
                    entry = self._conv[name]
                else:
                    entry["last_text"] = str(row.get("preview") or entry.get("last_text", ""))
                entry["unread"] = int(row.get("unread") or 0)
                entry["time"] = str(row.get("time") or "")
                avatar = self._ensure_avatar(name)
                if avatar:
                    entry["avatar_path"] = avatar
                order_names.append(name)
            self._apply_order(order_names)
            return len(rows)

        if not self.ocr.available:
            return 0
        scroll = getattr(self.platform, "scroll", None)
        if deep and callable(scroll):
            self._scroll_session_list(8, 14)
            collected: List[str] = []
            self._deep_collect = collected
            try:
                for index in range(20):
                    before = len(collected)
                    self._deep_page = index
                    self.sync_conversations(False)
                    self._scroll_session_list(-6, 1)
                    if index >= 1 and len(collected) == before:
                        break
            finally:
                self._deep_collect = None
                self._deep_page = None
            self._scroll_session_list(8, 12)
            ordered = [name for name in dict.fromkeys(collected)]
            for index, name in enumerate(ordered):
                if name in self._conv:
                    self._conv[name]["order"] = index
            return len(ordered)

        if self.platform.find_window() is None:
            return 0
        image = self.platform.capture()
        if image is None:
            return 0
        regions = self.layout.regions(image.width, image.height)
        # 列表画面没变化就复用上次结果：OCR 一次要 2~4 秒，不能每 2 秒重跑
        try:
            import hashlib as _hashlib

            fingerprint = _hashlib.md5(image.crop(regions.session_list).tobytes()).hexdigest()
        except Exception:
            fingerprint = ""
        if fingerprint and fingerprint == getattr(self, "_list_fingerprint", "") and self._conv \
                and time.time() - getattr(self, "_list_cached_at", 0) < 20:
            return getattr(self, "_list_count", len(self._conv))
        try:
            lines = self.ocr.recognize(image.crop(regions.session_list))
        except Exception:
            return 0
        if not lines:
            return 0
        import re as _re
        import numpy as np
        time_re = _re.compile(r"^(\d{1,2}[:：]\d{2}|\d{1,2}/\d{1,2}|昨天|前天|星期[一二三四五六日天]|上午|下午|中午|晚上).*$")

        def is_time(text: str) -> bool:
            return bool(time_re.match((text or "").strip().rstrip(".")))

        def is_bad_name(text: str) -> bool:
            value = (text or "").strip()
            if not value or len(value) > 30 or len(value) < 2:
                return True
            if value in ("搜索", "A", "AI") or "搜索" in value:
                return True
            if any(token in value for token in ("…", "：", "［", "］", "[", "]", "条]", "拍了拍",
                                                "..", "．．", "。。", "⋯")):
                return True
            if _re.search(r"\d{6,}", value):
                return True
            return value.replace(":", "").replace("：", "").isdigit()

        area = np.array(image.crop(regions.session_list).convert("RGB"))
        gray = np.array(image.crop(regions.session_list).convert("L"))
        red = (area[:, :, 0].astype(np.int16) >= 170) & (area[:, :, 1].astype(np.int16) <= 110) \
            & (area[:, :, 2].astype(np.int16) <= 110)
        times = [ln for ln in lines if is_time(ln.text)]
        parsed = []
        for anchor in times:
            anchor_y = (anchor.bbox[1] + anchor.bbox[3]) / 2.0
            candidates = [ln for ln in lines
                          if not is_time(ln.text) and ln.bbox[0] < 320
                          and not is_bad_name(ln.text)
                          and anchor_y - 30 <= (ln.bbox[1] + ln.bbox[3]) / 2.0 <= anchor_y + 22]
            # 会话名是黑色，消息预览是灰色：优先取黑色那行，避免把系统提示当会话名
            dark_candidates = [
                ln for ln in candidates
                if gray[int(ln.bbox[1]):int(ln.bbox[3]), int(ln.bbox[0]):int(ln.bbox[2])].size
                and float(np.percentile(
                    gray[int(ln.bbox[1]):int(ln.bbox[3]), int(ln.bbox[0]):int(ln.bbox[2])], 10)) <= 110
            ]
            candidates = dark_candidates or candidates
            if not candidates:
                continue
            name_line = min(candidates,
                            key=lambda ln: (abs((ln.bbox[1] + ln.bbox[3]) / 2.0 - anchor_y),
                                            -(ln.bbox[3] - ln.bbox[1]), -ln.score))
            name = name_line.text.strip()
            if any(token in name for token in _SYSTEM_PATTERNS):
                continue
            if name.endswith("禁") and len(name) > 2 and "群" in name:
                original = name
                name = name[:-1]
                self._conv.pop(original, None)
            name = self._canonical_peer(name)
            if any(item["name"] == name for item in parsed):
                continue
            preview = ""
            later = [t for t in times if (t.bbox[1] + t.bbox[3]) / 2.0 > anchor_y + 10]
            next_y = (min((t.bbox[1] + t.bbox[3]) / 2.0 for t in later) - 8) if later else anchor_y + 110
            for ln in sorted(lines, key=lambda l: l.bbox[1]):
                if ln is name_line or is_time(ln.text):
                    continue
                ly = (ln.bbox[1] + ln.bbox[3]) / 2.0
                if name_line.bbox[3] + 1 <= ly <= next_y and ln.bbox[0] < 320:
                    preview = (ln.text or "").strip()
                    break
            name_x = float(name_line.bbox[0])
            name_y1 = int(name_line.bbox[1])
            unread = self._detect_unread(area, red, name_x, float(name_y1))
            parsed.append({"name": name, "preview": preview, "unread": unread,
                           "time": anchor.text.strip().rstrip("."),
                           "name_x": name_x, "anchor_y": anchor_y,
                           "row_y": float(name_line.bbox[1]),
                           "name_h": float(name_line.bbox[3] - name_line.bbox[1])})

        # ---- 兜底：WeChat 4.x 只给部分会话画时间，按行聚类补全其余会话 ----
        clusters: List[List[Any]] = []
        for line in sorted(lines, key=lambda item: item.bbox[1]):
            for cluster in clusters:
                if abs(line.bbox[1] - cluster[-1].bbox[1]) <= 30:
                    cluster.append(line)
                    break
            else:
                clusters.append([line])
        known_tops = [float(item.get("row_y") or 0) for item in parsed]
        for cluster in clusters:
            cluster.sort(key=lambda item: item.bbox[1])
            head = cluster[0]
            if head.bbox[1] < 70:                      # 搜索框那一行
                continue
            if head.bbox[0] > 150:                     # 名字列在左边；右边是时间/预览
                continue
            if any(abs(float(head.bbox[1]) - top) < 18 for top in known_tops):
                continue                               # 这一行已经由时间锚点解析过
            # 会话名是黑色粗体，消息预览是灰色：灰色行不当作会话名
            patch = gray[int(head.bbox[1]):int(head.bbox[3]), int(head.bbox[0]):int(head.bbox[2])]
            if patch.size == 0 or float(np.percentile(patch, 10)) > 110:
                continue
            name = self._canonical_peer((head.text or "").strip())
            if not name or is_bad_name(name) or any(item["name"] == name for item in parsed):
                continue
            if any(token in name for token in _SYSTEM_PATTERNS):
                continue
            preview = ""
            for other in cluster[1:]:
                if other.bbox[0] < 250 and other.bbox[1] >= head.bbox[3] - 2:
                    preview = (other.text or "").strip()
                    break
            name_x = float(head.bbox[0])
            name_y1 = int(head.bbox[1])
            parsed.append({
                "name": name, "preview": preview, "time": "",
                "unread": self._detect_unread(area, red, name_x, float(name_y1)),
                "name_x": name_x,
                "anchor_y": (head.bbox[1] + head.bbox[3]) / 2.0,
                "row_y": float(name_y1),
                "name_h": float(head.bbox[3] - head.bbox[1]),
            })
            known_tops.append(float(name_y1))

        parsed.sort(key=lambda it: float(it.get("anchor_y") or 0))
        # 名字不该等于某行的消息预览（把"在呢"这类预览文字误当会话名的情况干掉）
        previews = {(item.get("preview") or "").strip() for item in parsed}
        previews.discard("")
        if previews:
            parsed = [item for item in parsed if item["name"].strip() not in previews]
        for index, item in enumerate(parsed):
            if index + 1 < len(parsed):
                delta = float(parsed[index + 1].get("anchor_y") or 0) - float(item.get("anchor_y") or 0)
            else:
                delta = 92.0
            item["row_height"] = min(120.0, max(56.0, delta))

        avatar_boxes = self._find_avatar_boxes(image, regions)
        for item in parsed:
            name = item["name"]
            entry = self._conv.get(name)
            if entry is None:
                self._touch(name, status="未读" if item["unread"] else "空闲",
                            text=item["preview"] or None)
                entry = self._conv[name]
            else:
                entry["last_text"] = item["preview"] or entry.get("last_text", "")
            entry["unread"] = item["unread"]
            entry["time"] = item["time"]
            entry["last_seen"] = time.time()
            name_h = float(item.get("name_h") or 16.0)
            name_top = float(item.get("row_y") or 0)
            name_x_global = regions.session_list[0] + float(item.get("name_x") or 0)
            box = self._match_avatar(avatar_boxes, name_x_global, name_top, name_h)
            if box:
                avatar = image.crop(box)
                path = os.path.join(self.avatar_dir, __import__("hashlib").sha1(
                    name.encode("utf-8")).hexdigest() + ".png")
                avatar.save(path)
                entry["avatar_path"] = path
            else:
                generated = self._ensure_avatar(name)
                if generated:
                    entry["avatar_path"] = generated
        for item in parsed:
            self._peer_positions[item["name"]] = {
                "page": float(self._deep_page or 0),
                "y": float(item.get("row_y", item.get("anchor_y", 0))),
            }
        self._row_index = [
            (item["name"],
             float(regions.session_list[1]) + float(item.get("row_y") or 0)
             + float(item.get("row_height") or 60) / 2.0)
            for item in parsed if item.get("name")
        ]
        # 红点贴在头像右上角，比行中心高约 22px：单独给红点匹配用，避免串到上一行
        self._badge_index = [(name, y - 22.0) for name, y in self._row_index]
        page_names = [item["name"] for item in parsed if item.get("name")]
        if self._deep_collect is not None:
            self._deep_collect.extend(page_names)
        else:
            self._apply_order(page_names)
        for entry in self._conv.values():
            if not entry.get("avatar_path"):
                generated = self._ensure_avatar(entry["peer"])
                if generated:
                    entry["avatar_path"] = generated
        valid = {item["name"] for item in parsed}
        # 合并 OCR / 头部读取抖动造成的重复会话（漏字、多点号等，如「岛开门见开心乐」与「青岛开门见开心乐」）
        import difflib as _difflib

        canonical_names: Dict[str, str] = {}
        for name in list(self._conv.keys()):
            key = self._clean_peer_name(name)
            if not key:
                continue
            match_key = None
            for existing_key in canonical_names:
                if existing_key == key:
                    match_key = existing_key
                    break
                if len(key) >= 4 and len(existing_key) >= 4 and \
                        _difflib.SequenceMatcher(None, key, existing_key).ratio() >= 0.9:
                    match_key = existing_key
                    break
            if match_key is None:
                canonical_names[key] = name
                continue
            keep = canonical_names[match_key]
            if keep == name:
                continue
            keep_score = (1 if keep in valid else 0, len(str(self._conv[keep].get("avatar_path") or "")))
            name_score = (1 if name in valid else 0, len(str(self._conv[name].get("avatar_path") or "")))
            good, drop = (keep, name) if keep_score >= name_score else (name, keep)
            canonical_names[match_key] = good
            self._conv.pop(drop, None)
            self._peer_positions.pop(drop, None)
            if drop in valid:
                valid.discard(drop)
                valid.add(good)
        for name in list(self._conv.keys()):
            if name in valid:
                continue
            if any(token in name for token in ("［", "］", "条]", "…", "：")):
                self._conv.pop(name, None)
        # 清理长时间没在列表里出现的旧条目（保留 5 分钟，避免刚点过就"unknown peer"）
        protected = set(self._pending) | set(self._view_history)
        now = time.time()
        for name in list(self._conv.keys()):
            if name in valid or name in protected:
                continue
            entry = self._conv[name]
            if entry.get("simulated"):
                continue
            if str(entry.get("status") or "").startswith(("草稿", "模拟", "已发送", "转人工")):
                continue
            try:
                allowed, _matched = self.whitelist.matches(name)
            except Exception:
                allowed = False
            if allowed:
                continue
            if now - float(entry.get("last_seen") or now) < 300:
                continue
            self._conv.pop(name, None)
            self._peer_positions.pop(name, None)
        for entry in self._conv.values():
            if not entry.get("avatar_path"):
                generated = self._ensure_avatar(entry["peer"])
                if generated:
                    entry["avatar_path"] = generated
        self._list_count = len(parsed)
        self._list_fingerprint = fingerprint if parsed else ""
        self._list_cached_at = time.time()
        return len(parsed)

    # ---------- 发送闸门（所有发送入口共用） ----------
    def send_gate(self, peer: str, manual: bool = False) -> Tuple[bool, str]:
        if self.dry_run:
            return False, "dry_run"
        if self.handoff.corrupt:
            return False, "handoff_corrupt"
        if not self.router.client.ready:
            return False, "model_not_ready"
        if not manual and self.handoff.is_taken(peer):
            return False, "manual_takeover"
        allowed, _matched = self.whitelist.matches_exact(peer)
        if not allowed:
            return False, "not_whitelisted"
        if not self.limiter.allowed(peer):
            return False, "rate_limited"
        return True, ""

    def _already_sent(self, peer: str, signature: str) -> Optional[str]:
        """同一批客户消息是否已经发过（含状态不明的），防止重复回复。"""
        if not signature:
            return None
        try:
            row = self.audit.find_send(peer, signature)
        except Exception:
            return None
        if row is None:
            return None
        state = row[4]
        return state if state in ("verified", "uncertain") else None

    def takeover(self, peer: str, reason: str = "人工接管") -> None:
        self.handoff.take(peer, reason)
        self._touch(peer, status="人工接管")
        self.emit("handoff", peer=peer, reason=reason)

    def release_handoff(self, peer: str) -> bool:
        released = self.handoff.release(peer)
        if released:
            self._touch(peer, status="已恢复自动")
            self.emit("handoff", peer=peer, reason="已恢复自动回复")
        return released

    # ---------- 处理一个未读 ----------
    def _handle_unread(self, badge, regions: Regions) -> bool:
        with self.wx_lock:
            return self._handle_unread_locked(badge, regions)

    def _handle_unread_locked(self, badge, regions: Regions) -> bool:
        if hasattr(self.platform, "ensure_visible"):
            try:
                self.platform.ensure_visible()
            except Exception:
                pass
        x, y = click_point(badge, regions)
        self.platform.click(x, y)
        image = settle_capture(
            self.platform.capture, regions,
            self.runtime.settle_s, self.runtime.aggregate_max_wait_s,
        )
        if image is None:
            return False

        header = self.sender.read_header(image, regions)
        peer = header or ""
        is_group = is_group_name(peer)

        items = self._read_items(image, regions)
        if not items:
            self.emit("skip", peer=peer, reason="没有新消息")
            return False

        if is_group:
            allowed, reason = group_allows(
                peer, items,
                self.runtime.allow_group_chat,
                self.runtime.group_whitelist,
                self.runtime.group_reply_mode,
                self.runtime.group_keywords,
            )
            if not allowed:
                self._touch(peer, status="已跳过", unread=0)
                self.audit.record(peer=peer, action="skip", status="group_skipped", detail=reason)
                self.emit("skip", peer=peer, reason=reason)
                return False
            self.emit("group", peer=peer, reason=f"群聊放行（{self.runtime.group_reply_mode}）")
        else:
            allowed, _matched = self.whitelist.matches_exact(peer)
            if not allowed:
                self._touch(peer, status="已跳过", unread=0)
                fuzzy, _name = self.whitelist.matches(peer)
                detail = "名称疑似截断，需人工确认" if fuzzy else "不在白名单"
                self.audit.record(peer=peer, action="skip", status="not_whitelisted",
                                  detail=detail)
                self.emit("skip", peer=peer, reason=detail)
                return False

        # 人工接管中：只记录消息，不回复（同一批只记一次）
        if self.handoff.is_taken(peer):
            sig = normalize_name(" / ".join(item.prompt_text() for item in items))
            _key, fresh = self._event_key(peer, sig)
            if fresh:
                for item in items:
                    self.memory.add(peer, "in", item.kind, item.prompt_text())
            self._mark_event(peer, "done")
            self._touch(peer, status="人工接管", unread=0)
            self.audit.record(peer=peer, action="skip", status="manual_takeover",
                              incoming=" / ".join(item.prompt_text() for item in items))
            self.emit("skip", peer=peer, reason="人工接管中，不自动回复")
            return False

        return self._respond(peer, items, image, regions, is_group)

    # ---------- 事件状态机：待处理 / 待重试 / 待发送 / 已完成 ----------
    def _event_key(self, peer: str, signature: str) -> Tuple[str, bool]:
        """内容相同的「新消息」必须算新事件；失败/限频的事件允许到期后重试。"""
        now = time.time()
        state = self._event_state.get(peer)
        if state and state.get("signature") == signature:
            status = state.get("status")
            if status == "running":
                return state["key"], False
            if status == "held":
                return state["key"], False
            if status == "retry":
                if now < float(state.get("next_at") or 0):
                    return state["key"], False
                state["status"] = "running"
                state["started"] = now
                return state["key"], True
            if status == "done":
                pass      # 已回复过，客户又说了一遍同样的话 → 新事件
        seq = max(self._event_seq.get(peer, 0), self._try_ledger_max(peer)) + 1
        self._event_seq[peer] = seq
        key = "%s#%d" % (signature, seq)
        self._event_state[peer] = {"signature": signature, "key": key, "seq": seq,
                                   "status": "running", "started": now,
                                   "attempts": 0, "next_at": 0.0}
        try:
            self.audit.delete_retry(peer)     # 新事件开始，旧的待重试记录作废
        except Exception:
            pass
        return key, True

    def _load_retries(self) -> None:
        try:
            rows = self.audit.list_retries()
        except Exception:
            return
        for row in rows:
            peer, signature, event_seq, status, attempts, next_at, last_error = row
            seq = int(event_seq or 0)
            self._event_state[peer] = {
                "signature": signature or "",
                "key": "%s#%d" % (signature or "", seq),
                "seq": seq,
                "status": status or "retry",
                "attempts": int(attempts or 0),
                "next_at": float(next_at or 0),
                "started": 0.0,
                "last_error": last_error or "",
            }
            self._event_seq[peer] = max(self._event_seq.get(peer, 0), seq)

    def _mark_event(self, peer: str, status: str, delay: float = 0.0) -> None:
        state = self._event_state.get(peer)
        if not state:
            return
        state["status"] = status
        if delay:
            state["next_at"] = time.time() + delay
        if status == "retry":
            state["attempts"] = int(state.get("attempts", 0)) + 1
        try:
            if status == "retry":
                self.audit.upsert_retry(
                    peer, state.get("signature", ""), int(state.get("seq") or 0),
                    status, int(state.get("attempts", 0)),
                    float(state.get("next_at") or 0), str(state.get("last_error") or ""))
            else:
                # held 的草稿在内存里，重启即丢失，不写重试表；done 要清掉旧记录
                self.audit.delete_retry(peer)
        except Exception:
            pass

    def _try_ledger_max(self, peer: str) -> int:
        try:
            return self.audit.max_event(peer)
        except Exception:
            return 0

    def _remember_items(self, peer: str, items, recorded: set) -> None:
        for item in items:
            key = (item.kind, item.prompt_text())
            if key in recorded:
                continue
            recorded.add(key)
            self.memory.add(peer, "in", item.kind, item.prompt_text())

    # ---------- 失败重试 / 限频后补发 ----------
    def drive_pending_work(self) -> None:
        """每轮扫描时调用：重试失败的生成、补发被限频挡住的草稿。"""
        with self.wx_lock:
            self._drive_retries()
            self._flush_held()

    def _drive_retries(self) -> None:
        now = time.time()
        for peer, state in list(self._event_state.items()):
            if state.get("status") != "retry" or now < float(state.get("next_at") or 0):
                continue
            if self.handoff.is_taken(peer):
                self._mark_event(peer, "done")
                continue
            if int(state.get("attempts", 0)) > 5:
                self.emit("skip", peer=peer, reason="重试多次仍失败，转人工处理")
                self.handoff.take(peer, "自动回复多次失败")
                self._mark_event(peer, "done")
                continue
            try:
                if not self.open_conversation(peer):
                    self._mark_event(peer, "retry", delay=60.0)
                    continue
                image = self.platform.capture()
                if image is None:
                    self._mark_event(peer, "retry", delay=30.0)
                    continue
                regions = self.layout.regions(image.width, image.height)
                items = self._read_items(image, regions)
                if not items:
                    self._mark_event(peer, "done")
                    continue
                self.emit("retry", peer=peer, reason="重试自动回复")
                self._respond(peer, items, image, regions, is_group_name(peer), remember=False)
            except Exception as exc:
                self.emit("error", message=f"重试失败：{exc}")
                self._mark_event(peer, "retry", delay=60.0)

    def _flush_held(self) -> None:
        """被限频/闸门挡住但已有草稿的会话：条件恢复后逐个补发（打开目标会话再校验版本）。"""
        peers = [name for name in self._pending_order if name in self._pending]
        for peer in peers:
            pending = self._pending.get(peer)
            if not pending or not pending.get("held_reason"):
                continue
            if self.handoff.is_taken(peer):
                continue
            ok, _reason = self.send_gate(peer)
            if not ok:
                continue
            # 切到目标会话，重新截图，避免"当前开着别人"时用错截图
            if not self.open_conversation(peer):
                continue
            image = self.platform.capture()
            if image is None:
                continue
            regions = self.layout.regions(image.width, image.height)
            latest = self._read_items(image, regions)
            if not latest:
                self._drop_pending(peer)
                continue
            signature = normalize_name(" / ".join(item.prompt_text() for item in latest))
            if signature != pending.get("signature"):
                self._drop_pending(peer)
                self._mark_event(peer, "retry", delay=1.0)
                self.emit("skip", peer=peer, reason="草稿过期（客户有新消息），重新生成")
                continue
            pending["image"] = image
            pending["regions"] = regions
            guard = self._make_guard(peer, signature)
            outcome = self.sender.send(peer, pending["reply"], image, regions, guard=guard)
            self._finish_send(peer, pending["reply"], outcome, pending.get("incoming", ""),
                              pending.get("model", ""), pending.get("latency_ms", 0),
                              pending.get("event_key", ""), int(pending.get("event_seq") or 0))
            time.sleep(0.3)      # 多客户公平补发，避免一次把窗口占满

    def _make_guard(self, peer: str, signature: str):
        def guard():
            # 等待/粘贴前重新检查全部授权状态：暂停、关掉 live、移除白名单都必须能取消发送
            if self.paused.is_set():
                return "paused"
            if self.dry_run or self.runtime.dry_run:
                return "dry_run"
            if self.handoff.corrupt:
                return "handoff_corrupt"
            if self.handoff.is_taken(peer):
                return "manual_takeover"
            allowed, _matched = self.whitelist.matches_exact(peer)
            if not allowed:
                return "not_whitelisted"
            latest = self._latest_incoming(peer)
            if latest is None:
                return None                 # 无法确认：交给 header / 输入框检查兜底
            if not latest:
                return "human_replied"
            current = normalize_name(" / ".join(item.prompt_text() for item in latest))
            if current != signature:
                return "new_message"
            return None
        return guard

    # ---------- 生成并（在允许时）发送 ----------
    def _respond(self, peer: str, items, image, regions: Regions, is_group: bool,
                 remember: bool = True) -> bool:
        incoming = [item.prompt_text() for item in items]
        joined = " / ".join(incoming)
        signature = normalize_name(joined)

        event_key, is_new = self._event_key(peer, signature)
        if not is_new:
            self.emit("skip", peer=peer, reason="同一批消息处理中或等待重试，已忽略")
            return False
        already = self._already_sent(peer, event_key)
        if already:
            self._mark_event(peer, "done")
            self.emit("skip", peer=peer, reason=f"这批消息已发送过（{already}），不重复发送")
            return False

        pending = self._pending.get(peer)
        if pending and pending.get("event_key") == event_key:
            self._mark_event(peer, "held")
            self.emit("skip", peer=peer, reason="已有同批消息的待发草稿，直接复用")
            return True

        self._touch(peer, status="处理中", unread=0)
        # 只有确认是「新事件」才写记忆：重复扫描/重试不会把同一句客户消息记两遍
        recorded = set()
        if remember:
            self._remember_items(peer, items, recorded)
        self.emit("incoming", peer=peer, messages=incoming, kinds=[item.kind for item in items])

        history = self.memory.history_lines(peer, self.config.memory.recent_limit)
        profile = self.memory.profile(peer)

        decision = self.guardrails.assess_incoming(joined)
        if decision.action == "escalate":
            self.handoff.take(peer, decision.reason)
            self._touch(peer, status="转人工")
            self.audit.record(peer=peer, action="escalate", status="handoff",
                              incoming=joined, detail=decision.reason)
            self.emit("escalate", peer=peer, reason=decision.reason)
            self._mark_event(peer, "done")
            return False

        reply, model, latency, tool_results = None, "", 0, []
        tool_results = self.tools.prefetch(joined)
        chunks = self._search_knowledge(joined)
        # 价格/库存类问题：既没有工具结果、也没有知识依据时，转人工而不是让模型猜
        if self._needs_evidence(joined) and not any(result.ok for result in tool_results) \
                and not chunks:
            self.handoff.take(peer, "没有可靠依据（产品/价格）")
            self._touch(peer, status="转人工")
            self.audit.record(peer=peer, action="escalate", status="no_evidence", incoming=joined)
            self.emit("escalate", peer=peer, reason="没有可靠依据，转人工确认")
            self._mark_event(peer, "done")
            return False
        converged = False
        # 生成期间客户可能连续补充：最多重生成 2 次；超过仍不一致就挂起，绝不发旧答案
        for _round in range(3):
            if _round > 0:
                tool_results = self.tools.prefetch(joined)
                chunks = self._search_knowledge(joined)
            for result in tool_results:
                self.emit("tool", name=result.name, summary=result.summary, ok=result.ok)
            reply, model, latency = self._generate(incoming, joined, history, profile,
                                                   tool_results, is_group, peer=peer,
                                                   chunks=chunks)
            if reply is None:
                self._touch(peer, status="生成失败待重试")
                state = self._event_state.get(peer)
                if state is not None:
                    state["last_error"] = self._last_error
                self._mark_event(peer, "retry", delay=15.0)
                self.emit("skip", peer=peer, reason="模型失败，稍后自动重试")
                return False
            latest = self._latest_incoming(peer)
            if latest is None:
                converged = True       # 无法复核，交给发送前的 header/输入框检查
                break
            if not latest:
                self.emit("skip", peer=peer, reason="客户已被人工回复，放弃自动草稿")
                self.audit.record(peer=peer, action="skip", status="human_replied", incoming=joined)
                self._mark_event(peer, "done")
                return False
            latest_joined = " / ".join(item.prompt_text() for item in latest)
            if normalize_name(latest_joined) == signature:
                converged = True
                break
            items = latest
            incoming = [item.prompt_text() for item in items]
            joined = latest_joined
            signature = normalize_name(joined)
            self._remember_items(peer, items, recorded)
            self.emit("incoming", peer=peer, messages=incoming, kinds=[item.kind for item in items])
            decision = self.guardrails.assess_incoming(joined)
            if decision.action == "escalate":
                self.handoff.take(peer, decision.reason)
                self._touch(peer, status="转人工")
                self.audit.record(peer=peer, action="escalate", status="handoff",
                                  incoming=joined, detail=decision.reason)
                self.emit("escalate", peer=peer, reason=decision.reason)
                self._mark_event(peer, "done")
                return False
        if reply is None:
            return False
        if not converged:
            # 客户一直在改：答案已经对不上最新需求，先挂起，等下一轮重新生成
            self._touch(peer, status="对话仍在变化，暂缓回复")
            self.audit.record(peer=peer, action="skip", status="conversation_moving",
                              incoming=joined, reply=reply)
            self.emit("skip", peer=peer, reason="客户仍在补充消息，暂缓回复")
            self._mark_event(peer, "retry", delay=3.0)
            return False

        state = self._event_state.get(peer) or {}
        event_seq = int(state.get("seq") or 0)
        self._pending[peer] = {
            "peer": peer, "reply": reply, "incoming": joined, "image": image,
            "regions": regions, "model": model, "latency_ms": latency,
            "incoming_items": incoming, "history": history, "profile": profile,
            "tool_results": [result.as_dict() for result in tool_results], "is_group": is_group,
            "event_key": event_key, "event_seq": event_seq, "signature": signature,
            "draft_id": uuid.uuid4().hex, "version": event_key,
        }
        if peer in self._pending_order:
            self._pending_order.remove(peer)
        self._pending_order.append(peer)
        self._touch(peer, status="草稿待发", unread=0, text=reply)
        self.audit.record(peer=peer, action="draft",
                          status="dry_run" if self.dry_run else "draft",
                          incoming=joined, reply=reply, model=model,
                          latency_ms=latency, image=image)
        self.emit("draft", peer=peer, reply=reply, model=model,
                  latency_ms=latency, dry_run=self.dry_run,
                  tools=[result.name for result in tool_results])

        ok, reason = self.send_gate(peer)
        if not ok:
            # 草稿保留在待发队列，闸门恢复后由 _flush_held 补发
            self._pending[peer]["held_reason"] = reason
            self.audit.record(peer=peer, action="draft", status=f"held_{reason}",
                              incoming=joined, reply=reply, model=model, latency_ms=latency)
            self.emit("held", peer=peer, reason=reason, reply=reply)
            self._mark_event(peer, "held")
            return True

        guard = self._make_guard(peer, signature)
        outcome = self.sender.send(peer, reply, image, regions, guard=guard)
        status, detail = outcome.outcome.status, outcome.outcome.detail
        if status == "aborted" and detail in ("manual_takeover", "human_replied"):
            self._mark_event(peer, "done")
        elif status == "aborted" and detail == "new_message":
            self._drop_pending(peer)
            self._mark_event(peer, "retry", delay=1.0)
            self.emit("skip", peer=peer, reason="发送前客户又改了消息，重新生成")
        elif status == "aborted" and detail == "input_not_empty":
            self.handoff.take(peer, "输入框里有未发送内容")
            self.emit("handoff", peer=peer, reason="输入框有内容，暂停自动发送，请人工处理")
            self._mark_event(peer, "done")
        return self._finish_send(peer, reply, outcome, joined, model, latency,
                                 event_key, event_seq)

    def _latest_incoming(self, peer: str):
        """生成期间复核：当前会话是否还是同一个客户、有没有新消息。

        返回 None = 无法确认；[] = 没有待回复消息（客户已被人工回复）；[items] = 待回复。
        """
        try:
            image = self.platform.capture()
            if image is None:
                return None
            regions = self.layout.regions(image.width, image.height)
            if not is_same_peer_strict(self.sender.read_header(image, regions), peer):
                return None
            return self._read_items(image, regions)
        except Exception:
            return None

    def _search_knowledge(self, query: str) -> List[Any]:
        """统一检索入口：WeKnora 优先，不可用/无命中时回退本地知识库。"""
        client = getattr(self, "weknora", None)
        if client is not None and client.enabled:
            hits = client.search(query, top_k=self.config.rag.top_k)
            if hits:
                return weknora_chunks(hits)
            if client.last_error:
                self.emit("error", message=f"WeKnora 检索失败，已回退本地知识库：{client.last_error}")
        return self.kb.search(query, top_k=self.config.rag.top_k)

    @staticmethod
    def _needs_evidence(text: str) -> bool:
        return any(word in (text or "") for word in
                   ("价", "多少钱", "报价", "价格", "库存", "有货", "现货", "型号", "美金", "usd"))

    # ---------- 生成 ----------
    def _friend_persona(self, peer: str) -> str:
        table = self.config.friend_personas or {}
        return (table.get(peer) or table.get("默认") or "").strip()

    def _generate(self, incoming: List[str], joined: str, history: List[str],
                  profile: Dict[str, str], tool_results: List[Any], is_group: bool,
                  peer: str = "", chunks: Optional[List[Any]] = None) -> Tuple[Optional[str], str, int]:
        if chunks is None:
            chunks = self._search_knowledge(joined)
        system = build_system_prompt(self.persona, chunks, profile)
        if profile.get("summary"):
            system += f"\n【历史摘要】{profile['summary']}"
        if tool_results:
            ok_lines = [result.summary for result in tool_results if result.ok]
            bad_lines = [result.summary for result in tool_results if not result.ok]
            if ok_lines:
                system += f"\n【工具查询结果（可直接引用，不要编造）】{'；'.join(ok_lines)}"
            if bad_lines:
                system += (f"\n【工具没有给出可靠结果，必须按下面的要求处理，不要编造价格/库存】"
                           f"{'；'.join(bad_lines)}")
        if is_group:
            system += "\n当前是群聊：回复要简短，必要时点名对方，不要刷屏。"
        friend_persona = self._friend_persona(peer)
        if friend_persona:
            system += f"\n【该好友专属人设】{friend_persona}"
        if self._instruction:
            system += f"\n【额外要求】{self._instruction}"

        messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]
        for example in self.persona.examples or []:
            if example.get("user") and example.get("assistant"):
                messages.append({"role": "user", "content": example["user"]})
                messages.append({"role": "assistant", "content": example["assistant"]})
        for line in history:
            role = "assistant" if line.startswith("我方") else "user"
            content = line
            for prefix in ("我方：", "对方：", "我方", "对方"):
                if content.startswith(prefix):
                    content = content[len(prefix):]
                    break
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": "\n".join(incoming)})

        schemas = self.tools.schemas() if (self.tools.enabled and self.router.client.ready) else None
        result = self.router.client.chat(
            messages, tools=schemas, executor=self.tools.execute if schemas else None
        )
        if self.router.client.dry_run and any(r.ok for r in tool_results):
            text = "（dry-run）根据工具查询：" + "；".join(r.summary for r in tool_results if r.ok) + \
                   "。配置 api_key 后由模型润色成自然回复。"
            checked = self.guardrails.check_reply(text)
            if checked.action == "reply":
                return checked.text, "dry-run", result.latency_ms
        if result.error or not result.text.strip():
            if self.router.client.dry_run and any(r.ok for r in tool_results):
                text = "（dry-run）根据工具查询：" + "；".join(r.summary for r in tool_results if r.ok) + \
                       "。配置 api_key 后由模型润色成自然回复。"
                checked = self.guardrails.check_reply(text)
                return (checked.text if checked.action == "reply" else None), "dry-run", result.latency_ms
            self.audit.record(peer="", action="error", status="llm_failed",
                              incoming=joined, detail=result.error or "空回复",
                              model=result.model, latency_ms=result.latency_ms)
            self._last_error = f"模型调用失败：{result.error or '空回复'}"
            self.emit("error", message=self._last_error)
            return None, result.model, result.latency_ms
        checked = self.guardrails.check_reply(result.text)
        if checked.action != "reply":
            self.audit.record(peer="", action="block", status=checked.action,
                              incoming=joined, reply=result.text, detail=checked.reason,
                              model=result.model, latency_ms=result.latency_ms)
            self._last_error = f"回复被护栏拦截：{checked.reason}"
            self.emit("blocked", reason=checked.reason, reply=result.text)
            return None, result.model, result.latency_ms
        return checked.text, result.model, result.latency_ms

    # ---------- 发送结果 ----------
    def _finish_send(self, peer: str, reply: str, outcome, joined: str,
                     model: str, latency: int, signature: str, event_seq: int = 0) -> bool:
        status = outcome.outcome.status
        detail = outcome.outcome.detail
        self.audit.record(peer=peer, action="send", status=status,
                          incoming=joined, reply=reply, detail=detail,
                          model=model, latency_ms=latency)
        try:
            self.audit.record_send(peer, signature, reply, status, detail, event=event_seq)
        except Exception:
            pass
        if status == "verified":
            self.memory.add(peer, "out", "text", reply)
            self._touch(peer, status="已发送", unread=0, text=reply)
            state = self._event_state.get(peer) or {}
            if state.get("key") == signature:
                state["sent"] = True
            self._mark_event(peer, "done")
            self.limiter.record(peer, cost=1)
            self._pending.pop(peer, None)
            if peer in self._pending_order:
                self._pending_order.remove(peer)
            self.emit("sent", peer=peer, status="verified", reply=reply, detail=detail)
            self._maybe_summarize(peer)
            return True
        if status == "uncertain":
            # 已经按过回车但没能确认：不再自动重发，转人工核查，避免重复轰炸客户
            self.handoff.take(peer, "发送状态不明，需人工核查")
            self._touch(peer, status="发送待人工确认")
            self._mark_event(peer, "done")
            self.emit("handoff", peer=peer, reason="发送状态不明，已暂停自动回复")
        else:
            self._touch(peer, status=f"未发送（{detail}）")
            if status == "failed":
                self._mark_event(peer, "retry", delay=20.0)
        self.emit("sent", peer=peer, status=status, reply=reply, detail=detail)
        return False

    # ---------- 摘要（长记忆压缩） ----------
    def _maybe_summarize(self, peer: str) -> None:
        if peer in self._summarizing or not self.memory.needs_summary(peer):
            return
        if self.router.client.dry_run or not self.router.client.ready:
            return
        self._summarizing.add(peer)
        try:
            rows = self.memory.pending_messages(peer)
            if not rows:
                return
            text = "\n".join(f"{'对方' if d == 'in' else '我方'}：{t}" for _i, d, _k, t in rows)
            old_summary = (self.memory.profile(peer).get("summary") or "").strip()
            prompt_text = text[:4000]
            if old_summary:
                prompt_text = f"【已有摘要，必须保留其中的需求/报价/交期等事实】\n{old_summary}\n\n【新增消息】\n{prompt_text}"
            summary_prompt = [
                {"role": "system", "content": "把下面的聊天合并成 3 句话以内的客户档案摘要：保留旧摘要里的关键事实，再补充新信息；不要丢掉已有的需求、报价、交期、称呼。"},
                {"role": "user", "content": prompt_text},
            ]
            result = self.router.client.chat(summary_prompt, retries=0)
            if result.text and not result.error:
                self.memory.set_summary(peer, result.text.strip(), self.memory.last_id(peer))
                self.emit("memory", peer=peer, message="已更新会话摘要", summary=result.text.strip()[:120])
        finally:
            self._summarizing.discard(peer)

    @staticmethod
    def _same_peer(current: str, peer: str) -> bool:
        left, right = normalize_name(current), normalize_name(peer)
        if not left or not right:
            return False
        if left == right or left in right or right in left:
            return True
        head = min(4, len(right))
        return left.startswith(right[:head]) or right.startswith(left[:head])

    def _open_by_search(self, peer: str) -> bool:
        """用微信搜索（Cmd+F）打开会话；名称逐级截短，兼容 OCR 误差。"""
        press_command = getattr(self.platform, "press_command_key", None)
        if not callable(press_command):
            return False
        if hasattr(self.platform, "ensure_visible"):
            try:
                self.platform.ensure_visible()
            except Exception:
                pass
        queries = []
        for query in (peer, peer[:6], peer[:4], peer[:2]):
            if len(query) >= 2 and query not in queries:
                queries.append(query)
        for query in queries:
            try:
                press_command(3)  # Cmd+F
                time.sleep(0.35)
                if not self.platform.set_clipboard(query):
                    continue
                self.platform.press_ctrl_v()
                time.sleep(0.55)
                self.platform.press_enter()
                time.sleep(0.85)
                image = self.platform.capture()
                if image is None:
                    continue
                regions = self.layout.regions(image.width, image.height)
                if self._same_peer(self.sender.read_header(image, regions), peer):
                    return True
            except Exception:
                continue
        # 清掉搜索框，避免影响后续操作
        try:
            if callable(getattr(self.platform, "press_escape", None)):
                self.platform.press_escape()
        except Exception:
            pass
        return False

    def _open_by_position(self, peer: str) -> bool:
        position = self._peer_positions.get(peer)
        if not position:
            return False
        page = int(position.get("page", 0))
        y = float(position.get("y", 0))
        self._scroll_session_list(8, 14)   # 回到顶部
        for _ in range(page):
            self._scroll_session_list(-6, 1)
        image = self.platform.capture()
        if image is None:
            return False
        regions = self.layout.regions(image.width, image.height)
        left, _top, right, _bottom = regions.session_list
        x = left + (right - left) * 0.55
        for offset in (0, 12, -12, 24, -24):
            if hasattr(self.platform, "ensure_visible"):
                try:
                    self.platform.ensure_visible()
                except Exception:
                    pass
            self.platform.click(x, max(0, y + offset))
            time.sleep(0.8)
            current = self.platform.capture()
            if current is None:
                continue
            check = self.layout.regions(current.width, current.height)
            if self._same_peer(self.sender.read_header(current, check), peer):
                return True
        return False

    def _click_peer_row(self, regions: Regions, y: float, peer: str) -> bool:
        """点会话列表某一行并校验标题。"""
        x = regions.session_list[0] + (regions.session_list[2] - regions.session_list[0]) * 0.55
        if hasattr(self.platform, "ensure_visible"):
            try:
                self.platform.ensure_visible()
            except Exception:
                pass
        self.platform.click(x, y)
        time.sleep(0.45)
        current = self.platform.capture()
        if current is None:
            return False
        check = self.layout.regions(current.width, current.height)
        return self._same_peer(self.sender.read_header(current, check), peer)

    def _open_by_visible(self, peer: str) -> bool:
        """当前可见列表里直接找行点击（最稳，不需要搜索）。"""
        image = self.platform.capture()
        if image is None:
            return False
        regions = self.layout.regions(image.width, image.height)
        if self._same_peer(self.sender.read_header(image, regions), peer):
            return True

        def strict(a: str, b: str) -> bool:
            left_, right_ = normalize_name(a), normalize_name(b)
            if not left_ or not right_:
                return False
            if left_ == right_:
                return True
            return (left_ in right_ or right_ in left_) and min(len(left_), len(right_)) >= 3

        # 1) 用上次同步得到的行坐标直接点：不用 OCR 整屏，快一个数量级
        for name, row_y in getattr(self, "_row_index", []):
            if not strict(name, peer):
                continue
            if self._click_peer_row(regions, row_y, peer):
                return True
            break

        # 2) 兜底：OCR 列表逐行匹配
        try:
            lines = self.ocr.recognize(image.crop(regions.session_list))
        except Exception:
            return False
        target = None
        for line in lines:
            if strict(line.text, peer):
                target = line
                break
        if target is None:
            return False
        y = (target.bbox[1] + target.bbox[3]) / 2.0
        return self._click_peer_row(regions, y, peer)

    @_wx_locked
    def open_conversation(self, peer: str) -> bool:
        """打开任意会话：可见列表 → 滚动查找 → 搜索兜底。"""
        if self._open_by_visible(peer):
            return True
        if self._scroll_find_and_click(peer):
            return True
        if self._open_by_search(peer):
            return True
        image = self.platform.capture()
        if image is None:
            return False
        regions = self.layout.regions(image.width, image.height)
        if self._same_peer(self.sender.read_header(image, regions), peer):
            return True
        try:
            lines = self.ocr.recognize(image.crop(regions.session_list))
        except Exception:
            return False
        wanted = normalize_name(peer)
        best = None
        for line in lines:
            text = normalize_name(line.text)
            if text and (text == wanted or wanted in text or text in wanted):
                best = line
                break
        if best is None:
            # 列表里没有：滚动会话列表逐页找
            if self._scroll_find_and_click(peer):
                return True
            # 再尝试搜索框
            if self._search_and_open(image, regions, peer):
                return True
            return False
        y = (best.bbox[1] + best.bbox[3]) / 2.0
        x = regions.session_list[0] + (regions.session_list[2] - regions.session_list[0]) * 0.55
        if hasattr(self.platform, "ensure_visible"):
            try:
                self.platform.ensure_visible()
            except Exception:
                pass
        self.platform.click(x, y)
        time.sleep(0.8)
        current = self.platform.capture()
        if current is None:
            return False
        regions = self.layout.regions(current.width, current.height)
        return self._same_peer(self.sender.read_header(current, regions), peer)

    def _scroll_find_and_click(self, peer: str) -> bool:
        """滚动微信会话列表，找到目标会话并点开。"""
        wanted = normalize_name(peer)
        self._scroll_session_list(8, 10)  # 滚到顶部
        for _page in range(10):
            image = self.platform.capture()
            if image is None:
                return False
            regions = self.layout.regions(image.width, image.height)
            try:
                lines = self.ocr.recognize(image.crop(regions.session_list))
            except Exception:
                return False
            target = None
            for line in lines:
                text = normalize_name(line.text)
                if text and (text == wanted or wanted in text or text in wanted):
                    target = line
                    break
            if target is not None:
                y = (target.bbox[1] + target.bbox[3]) / 2.0
                x = regions.session_list[0] + (regions.session_list[2] - regions.session_list[0]) * 0.55
                if hasattr(self.platform, "ensure_visible"):
                    try:
                        self.platform.ensure_visible()
                    except Exception:
                        pass
                self.platform.click(x, y)
                time.sleep(0.9)
                current = self.platform.capture()
                if current is None:
                    return False
                check = self.layout.regions(current.width, current.height)
                if self._same_peer(self.sender.read_header(current, check), peer):
                    return True
            self._scroll_session_list(-4, 1)
        return False

    def _search_and_open(self, image, regions: Regions, peer: str) -> bool:
        """用微信搜索框搜索联系人并点开。"""
        left, _top, right, _bottom = regions.session_list
        width = max(1, right - left)
        search_x = left + width * 0.35
        search_y = 58
        if hasattr(self.platform, "ensure_visible"):
            try:
                self.platform.ensure_visible()
            except Exception:
                pass
        self.platform.click(search_x, search_y)
        time.sleep(0.3)
        if not self.platform.set_clipboard(peer):
            return False
        self.platform.press_ctrl_v()
        time.sleep(1.0)
        latest = self.platform.capture()
        if latest is None:
            return False
        try:
            lines = self.ocr.recognize(latest.crop((left, 0, right, latest.height)))
        except Exception:
            return False
        wanted = normalize_name(peer)
        best = None
        for line in lines:
            text = normalize_name(line.text)
            if text and (text == wanted or wanted in text):
                best = line
                break
        if best is None:
            return False
        y = (best.bbox[1] + best.bbox[3]) / 2.0 + 10
        self.platform.click(left + width * 0.55, y)
        time.sleep(0.9)
        current = self.platform.capture()
        if current is None:
            return False
        check_regions = self.layout.regions(current.width, current.height)
        return normalize_name(self.sender.read_header(current, check_regions)) == wanted

    # ---------- 手动：处理当前打开的会话 ----------
    def process_current(self) -> Optional[Dict[str, Any]]:
        with self.wx_lock:
            return self._process_current_locked()

    def _process_current_locked(self) -> Optional[Dict[str, Any]]:
        image = self.platform.capture()
        if image is None:
            self.emit("error", message="抓图失败")
            return None
        regions = self.layout.regions(image.width, image.height)
        header = self.sender.read_header(image, regions)
        peer = header or ""
        if not peer:
            self.emit("error", message="读不到会话标题（OCR/权限问题）")
            return None
        if len(peer.strip()) < 2 or any(token in peer for token in _SYSTEM_PATTERNS) \
                or "×" in peer or peer.strip() in ("…", "···"):
            self.emit("skip", peer=peer, reason="会话标题识别不完整，跳过")
            return None
        items = self._read_items(image, regions)
        if not items:
            self.emit("skip", peer=peer, reason="当前会话没有可读消息")
            return None
        if is_group_name(peer):
            allowed, reason = group_allows(
                peer, items, self.runtime.allow_group_chat,
                self.runtime.group_whitelist, self.runtime.group_reply_mode,
                self.runtime.group_keywords,
            )
            if not allowed:
                self.emit("skip", peer=peer, reason=reason)
                return None
        else:
            allowed, _matched = self.whitelist.matches(peer)
            if not allowed:
                self.emit("skip", peer=peer, reason="不在白名单（当前会话）")
                return None
        if self.handoff.is_taken(peer):
            self.emit("skip", peer=peer, reason="人工接管中，不自动回复")
            return None
        # 与未读路径走同一套「生成 → 闸门 → 发送」，不再出现"这里永远只出草稿"
        handled = self._respond(peer, items, image, regions, is_group_name(peer))
        if not handled:
            return None
        draft = self._pending.get(peer)
        if draft is None:
            return {"peer": peer, "reply": "", "sent": True, "incoming": []}
        return {"peer": peer, "reply": draft.get("reply", ""), "dry_run": self.dry_run,
                "incoming": [item.prompt_text() for item in items]}

    # ---------- 人工介入 ----------
    def _resolve_pending(self, peer: Optional[str]) -> Optional[Dict[str, Any]]:
        if peer:
            return self._pending.get(peer)
        return self.pending

    def approve_pending(self, text_override: Optional[str] = None,
                        peer: Optional[str] = None,
                        draft_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self.wx_lock:
            return self._approve_pending_locked(text_override, peer, draft_id)

    def _approve_pending_locked(self, text_override: Optional[str] = None,
                                peer: Optional[str] = None,
                                draft_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        # 发送必须绑定到明确联系人：不再"没找到就回退到别人的草稿"
        if not peer:
            return {"ok": False, "status": "blocked",
                    "detail": "必须指定要发送的联系人（草稿与联系人绑定）"}
        pending = self._pending.get(peer)
        if not pending:
            return {"ok": False, "status": "blocked",
                    "detail": f"「{peer}」没有待发送草稿，拒绝跨会话发送", "peer": peer}
        # 草稿 ID 绑定：前端拿着旧草稿（或别的会话的草稿）来发，一律拒绝
        if draft_id and pending.get("draft_id") and draft_id != pending.get("draft_id"):
            return {"ok": False, "status": "stale",
                    "detail": "草稿已更新，请刷新后重新确认", "peer": peer}
        reply = (text_override or "").strip() or pending["reply"]
        checked = self.guardrails.check_reply(reply)
        if checked.action != "reply":
            return {"ok": False, "status": "blocked", "detail": checked.reason,
                    "peer": peer, "reply": reply}
        pending["reply"] = checked.text
        event_key = pending.get("event_key") or normalize_name(pending.get("incoming", ""))
        event_seq = int(pending.get("event_seq") or 0)

        # 模拟会话：只走项目内的"发送"，不碰真实微信
        if peer.startswith("【模拟】"):
            self.audit.record_send(peer, event_key, pending["reply"], "verified", "simulated",
                                   event=event_seq)
            self.memory.add(peer, "out", "text", pending["reply"])
            self._touch(peer, status="模拟已回复", text=pending["reply"])
            self._drop_pending(peer)
            self.emit("sent", peer=peer, status="verified", reply=pending["reply"],
                      detail="simulated")
            return {"ok": True, "status": "verified", "detail": "simulated",
                    "peer": peer, "reply": pending["reply"]}

        # 先把目标会话切到前台，再做版本校验：否则"当前打开的是别人"会被误判成无法确认
        current = self.platform.capture()
        if current is not None:
            regions = self.layout.regions(current.width, current.height)
            header_now = self.sender.read_header(current, regions)
            if not is_same_peer_strict(header_now, peer):
                if not self.open_conversation(peer):
                    return {"ok": False, "status": "aborted",
                            "detail": "无法切回目标会话", "peer": peer,
                            "reply": pending["reply"]}
                current = self.platform.capture()
        if current is None:
            return {"ok": False, "status": "aborted", "detail": "抓图失败", "peer": peer}
        pending["image"] = current
        pending["regions"] = self.layout.regions(current.width, current.height)

        # 草稿版本校验：客户又发了新消息、或已被人工回复，旧草稿作废
        latest = self._latest_incoming(peer)
        if latest is None:
            return {"ok": False, "status": "blocked",
                    "detail": "无法确认当前会话状态，请稍后重试", "peer": peer}
        if not latest:
            self._drop_pending(peer)
            return {"ok": False, "status": "stale",
                    "detail": "当前没有待回复的消息（可能已被人工回复），草稿已作废", "peer": peer}
        latest_joined = " / ".join(item.prompt_text() for item in latest)
        if normalize_name(latest_joined) != pending.get("signature", ""):
            self._drop_pending(peer)
            self.emit("skip", peer=peer, reason="客户已有新消息，旧草稿作废")
            return {"ok": False, "status": "stale",
                    "detail": "客户已有新消息，草稿已过期，请重新生成", "peer": peer}

        # 统一闸门：人工批准也必须过 live / 模型就绪 / 白名单 / 限频（接管状态由人工覆盖）
        ok, reason = self.send_gate(peer, manual=True)
        if not ok:
            self.audit.record(peer=peer, action="send", status=f"blocked_{reason}",
                              reply=pending["reply"])
            return {"ok": False, "status": "blocked", "detail": reason,
                    "peer": peer, "reply": pending["reply"]}

        already = self._already_sent(peer, event_key)
        if already == "verified":
            return {"ok": False, "status": "duplicate",
                    "detail": "这批消息已确认发送过，拒绝重复发送",
                    "peer": peer, "reply": pending["reply"]}

        guard = self._make_guard(peer, pending.get("signature", ""))
        outcome = self.sender.send(peer, pending["reply"], pending["image"],
                                   pending["regions"], guard=guard)
        ok = self._finish_send(peer, pending["reply"], outcome, pending.get("incoming", ""),
                               pending.get("model", ""), pending.get("latency_ms", 0),
                               event_key, event_seq)
        return {"ok": ok, "status": outcome.outcome.status, "detail": outcome.outcome.detail,
                "peer": peer, "reply": pending["reply"]}

    def _drop_pending(self, peer: str) -> None:
        self._pending.pop(peer, None)
        if peer in self._pending_order:
            self._pending_order.remove(peer)

    def stop_sending(self) -> None:
        """停止操作：取消所有待发任务，事件转为可重试，不会在恢复后突然发出去。"""
        with self.wx_lock:
            self.pause()
            for peer in list(self._pending.keys()):
                self._drop_pending(peer)
                self._mark_event(peer, "retry", delay=30.0)
            self.emit("status", message="已停止：待发任务全部取消")

    def discard_pending(self, peer: Optional[str] = None) -> None:
        target = self._resolve_pending(peer)
        if target:
            self._pending.pop(target["peer"], None)
            if target["peer"] in self._pending_order:
                self._pending_order.remove(target["peer"])
        self.emit("status", message="已丢弃当前草稿")

    def regenerate_pending(self, peer: Optional[str] = None) -> Optional[Dict[str, Any]]:
        pending = self._resolve_pending(peer)
        if not pending:
            return None
        reply, model, latency = self._generate(
            pending.get("incoming_items") or [pending.get("incoming", "")],
            pending.get("incoming", ""),
            pending.get("history") or [],
            pending.get("profile") or {},
            pending.get("tool_results") or [],
            pending.get("is_group", False),
            peer=pending.get("peer", ""),
        )
        if reply is None:
            return None
        pending.update({"reply": reply, "model": model, "latency_ms": latency})
        self.emit("draft", peer=pending["peer"], reply=reply, model=model,
                  latency_ms=latency, dry_run=self.dry_run, regenerated=True)
        return {"reply": reply, "peer": pending["peer"]}

    def send_manual(self, peer: str, text: str) -> Optional[Dict[str, Any]]:
        with self.wx_lock:
            image = self.platform.capture()
            if image is None:
                return {"ok": False, "status": "capture_failed"}
            regions = self.layout.regions(image.width, image.height)
            checked = self.guardrails.check_reply(text)
            if checked.action != "reply":
                return {"ok": False, "status": "blocked", "detail": checked.reason}
            ok, reason = self.send_gate(peer, manual=True)
            if not ok:
                return {"ok": False, "status": "blocked", "detail": reason}
            outcome = self.sender.send(peer, checked.text, image, regions)
            if outcome.outcome.status == "verified":
                self.memory.add(peer, "out", "text", checked.text)
                self._touch(peer, status="已发送", text=checked.text)
            return {"ok": outcome.outcome.status == "verified",
                    "status": outcome.outcome.status, "detail": outcome.outcome.detail}

    def close(self) -> None:
        self.audit.close()
        self.memory.close()
