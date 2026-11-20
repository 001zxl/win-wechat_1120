from __future__ import annotations

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server import WEB_DIR, resolve_static_path
from wxreply.agent import Agent
from wxreply.config import Config
from wxreply.executor.sender import SafeSender
from wxreply.handoff import HandoffStore
from wxreply.perception.layout import LayoutCalculator
from wxreply.platform.mock import MockPlatform
from wxreply.policy import Whitelist, is_same_peer_strict, normalize_name


def make_config(tmpdir: str) -> Config:
    cfg = Config(base_dir=ROOT)
    cfg.persona = "personas/sales_zhang.yaml"
    cfg.kb_dir = "kb"
    cfg.audit.db_path = os.path.join(tmpdir, "audit.db")
    cfg.memory.db_path = os.path.join(tmpdir, "memory.db")
    cfg.rag.embedding_cache = os.path.join(tmpdir, "emb.json")
    cfg.tools.price_file = os.path.join(tmpdir, "price.json")
    cfg.tools.leads_file = os.path.join(tmpdir, "leads.jsonl")
    cfg.audit.evidence_dir = os.path.join(tmpdir, "evidence")
    cfg.audit.save_screenshots = False
    cfg.handoff_path = os.path.join(tmpdir, "handoff.json")
    cfg.runtime.settle_s = 0.01
    cfg.runtime.aggregate_max_wait_s = 0.02
    cfg.runtime.dry_run = True
    cfg.runtime.send_retries = 2
    cfg.safety.risk_enabled = False
    cfg.safety.whitelist = ["张三", "李四"]
    cfg.ai.api_key = ""
    return cfg


class SwitchOnFocusPlatform(MockPlatform):
    """「校验时是张三，抢到焦点后切到李四」"""

    def __init__(self) -> None:
        super().__init__()
        self.current = 0
        self.switch_to = 1

    def focus(self, window) -> bool:
        ok = super().focus(window)
        if self.switch_to is not None:
            self.current = self.switch_to
            self.switch_to = None
        return ok


class DropSendPlatform(MockPlatform):
    """按了回车但没发出去，同时客户又发来一条新消息"""

    def __init__(self) -> None:
        super().__init__()
        self.current = 0

    def press_enter(self) -> None:
        self.sessions[self.current].messages.append({"dir": "in", "kind": "text", "text": "在吗？"})


class CaptureFailAfterSendPlatform(MockPlatform):
    """消息发出去了，但发送后截图失败"""

    def __init__(self) -> None:
        super().__init__()
        self.current = 0
        self.captures = 0
        self.fail_after = None

    def capture(self):
        self.captures += 1
        if self.fail_after is not None and self.captures > self.fail_after:
            return None
        return super().capture()


def regions_for(platform: MockPlatform):
    return LayoutCalculator().regions(platform.WIDTH, platform.HEIGHT)


class TestSendRecipientGuard(unittest.TestCase):
    def test_header_changed_before_paste_aborts(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            cfg.runtime.dry_run = False
            platform = SwitchOnFocusPlatform()
            image = platform.capture()
            sender = SafeSender(platform, cfg, Whitelist(cfg.safety.whitelist), None)
            result = sender.send("张三", "在的，您说", image, regions_for(platform))
            self.assertEqual(result.outcome.status, "aborted")
            self.assertEqual(result.outcome.detail, "header_changed")
            self.assertEqual(platform.sent, [])


class TestSendVerification(unittest.TestCase):
    def test_no_new_bubble_is_uncertain(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            cfg.runtime.dry_run = False
            platform = DropSendPlatform()
            image = platform.capture()
            sender = SafeSender(platform, cfg, Whitelist(cfg.safety.whitelist), None)
            result = sender.send("张三", "在的，您说", image, regions_for(platform))
            self.assertEqual(result.outcome.status, "uncertain", result.outcome.detail)
            self.assertNotEqual(result.outcome.status, "verified")
            self.assertEqual(platform.sent, [])

    def test_capture_after_send_fails_no_duplicate(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            cfg.runtime.dry_run = False
            platform = CaptureFailAfterSendPlatform()
            platform.fail_after = 2          # 第 1 次：发送前；第 2 次：粘贴前核对；第 3 次：发送后 → 失败
            image = platform.capture()
            platform.captures = 1
            sender = SafeSender(platform, cfg, Whitelist(cfg.safety.whitelist), None)
            result = sender.send("张三", "在的，您说", image, regions_for(platform))
            self.assertEqual(result.outcome.status, "uncertain")
            self.assertEqual(len(platform.sent), 1, "不确定时必须停止重试，不能重复发送")

    def test_mock_send_verified_baseline(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            cfg.runtime.dry_run = False
            platform = MockPlatform()
            platform.current = 0
            image = platform.capture()
            sender = SafeSender(platform, cfg, Whitelist(cfg.safety.whitelist), None)
            result = sender.send("张三", "好的，稍等", image, regions_for(platform))
            self.assertEqual(result.outcome.status, "verified", result.outcome.detail)


class TestSendGate(unittest.TestCase):
    def test_gate_blocks_dry_run_model_and_takeover(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            agent = Agent(cfg, MockPlatform(), dry_run=True)
            ok, reason = agent.send_gate("张三")
            self.assertFalse(ok)
            self.assertEqual(reason, "dry_run")

            agent.set_dry_run(False)          # 没配 key → 模型不可用
            ok, reason = agent.send_gate("张三")
            self.assertFalse(ok)
            self.assertEqual(reason, "model_not_ready")

            agent.router.client.config.api_key = "sk-test"   # 假装模型就绪
            ok, reason = agent.send_gate("张三")
            self.assertTrue(ok, reason)

            agent.handoff.take("张三", "投诉")
            ok, reason = agent.send_gate("张三")
            self.assertFalse(ok)
            self.assertEqual(reason, "manual_takeover")
            # 人工亲自发不受接管限制
            ok, _reason = agent.send_gate("张三", manual=True)
            self.assertTrue(ok)
            agent.close()


class TestHandoffStore(unittest.TestCase):
    def test_persist_and_release(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = os.path.join(tmp, "handoff.json")
            store = HandoffStore(path)
            store.take("张三", "投诉")
            self.assertTrue(store.is_taken("张三"))
            again = HandoffStore(path)
            self.assertTrue(again.is_taken("张三"))
            self.assertTrue(again.release("张三"))
            self.assertFalse(HandoffStore(path).is_taken("张三"))


class TestDuplicateAndMemory(unittest.TestCase):
    def test_already_sent_blocks_second_reply(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            agent = Agent(cfg, MockPlatform(), dry_run=True)
            platform = MockPlatform()
            platform.current = 0
            items = platform.read_incoming_items()
            signature = normalize_name(" / ".join(item.prompt_text() for item in items))
            agent.audit.record_send("张三", signature, "在的", "verified")
            self.assertEqual(agent._already_sent("张三", signature), "verified")

            image = platform.capture()
            regions = agent.layout.regions(image.width, image.height)
            before = len(agent._pending)
            agent._respond("张三", items, image, regions, False)
            self.assertEqual(len(agent._pending), before, "同一批消息不应再生成新草稿")
            agent.close()

    def test_dry_run_draft_not_written_into_history(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)          # 没有 api_key，dry-run
            agent = Agent(cfg, MockPlatform(), dry_run=True)
            platform = MockPlatform()
            platform.current = 0
            items = platform.read_incoming_items()
            image = platform.capture()
            regions = agent.layout.regions(image.width, image.height)
            agent._respond("张三", items, image, regions, False)
            history = agent.memory.history_lines("张三", 20)
            self.assertTrue(any("对方" in line for line in history), "客户消息应照常记忆")
            self.assertFalse(any("我方" in line for line in history),
                             "未发送的草稿不能写进正式对话历史")
            agent.close()


class TestInterruptHandling(unittest.TestCase):
    def test_new_message_during_generation_regenerates(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            platform = MockPlatform()
            platform.current = 0
            agent = Agent(cfg, platform, dry_run=True)
            calls = {"n": 0}

            def fake_generate(incoming, joined, *_args, **_kwargs):
                calls["n"] += 1
                if calls["n"] == 1:
                    platform.sessions[0].messages.append(
                        {"dir": "in", "kind": "text", "text": "刚才说错了，要三个"})
                return ("回复%d" % calls["n"], "fake", 1)

            agent._generate = fake_generate
            items = platform.read_incoming_items()
            image = platform.capture()
            regions = agent.layout.regions(image.width, image.height)
            agent._respond("张三", items, image, regions, False)
            self.assertEqual(calls["n"], 2, "客户补充消息后应重新生成")
            draft = agent.pending_for("张三")
            self.assertIsNotNone(draft)
            self.assertEqual(draft["reply"], "回复2")
            self.assertIn("三个", draft["incoming"])
            agent.close()


class TestSecurityAndNaming(unittest.TestCase):
    def test_static_path_traversal_blocked(self) -> None:
        self.assertIsNone(resolve_static_path(WEB_DIR, "../config.example.yaml"))
        self.assertIsNone(resolve_static_path(WEB_DIR, "..\\config.example.yaml"))
        self.assertIsNone(resolve_static_path(WEB_DIR, "/../server.py"))
        resolved = resolve_static_path(WEB_DIR, "index.html")
        self.assertIsNotNone(resolved)
        self.assertTrue(resolved.endswith("index.html"))

    def test_strict_peer_match(self) -> None:
        self.assertTrue(is_same_peer_strict("张三", "张三"))
        self.assertTrue(is_same_peer_strict("张三 ", "张三"))
        self.assertTrue(is_same_peer_strict("产品群(25)", "产品群"))
        self.assertFalse(is_same_peer_strict("张三丰", "张三"))
        self.assertFalse(is_same_peer_strict("", "张三"))
        self.assertFalse(is_same_peer_strict("李四", "张三"))


if __name__ == "__main__":
    unittest.main()

