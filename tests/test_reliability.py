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
    cfg.safety.escalate_keywords = ["投诉", "退款"]
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
    def test_historical_bubble_cannot_confirm(self) -> None:
        """发送完全没生效、OCR 只读到历史"好的"时，绝不能再判成功。"""

        class EchoPlatform(MockPlatform):
            def __init__(self) -> None:
                super().__init__()
                self.current = 0

            def press_enter(self) -> None:      # 发送不生效
                return

            def last_outgoing_text(self):
                return "好的"                    # 历史里已经有的旧消息

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            cfg.runtime.dry_run = False
            platform = EchoPlatform()
            image = platform.capture()
            sender = SafeSender(platform, cfg, Whitelist(cfg.safety.whitelist), None)
            result = sender.send("张三", "好的，确认数量后给您报价", image, regions_for(platform))
            self.assertEqual(result.outcome.status, "uncertain", result.outcome.detail)

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
            platform = MockPlatform()
            platform.current = 0
            agent = Agent(cfg, platform, dry_run=True)
            items = platform.read_incoming_items()
            signature = normalize_name(" / ".join(item.prompt_text() for item in items))
            event_key, _ = agent._event_key("张三", signature)
            agent.audit.record_send("张三", event_key, "在的", "verified", event=1)
            self.assertEqual(agent._already_sent("张三", event_key), "verified")

            image = platform.capture()
            regions = agent.layout.regions(image.width, image.height)
            before = len(agent._pending)
            agent._respond("张三", items, image, regions, False)
            self.assertEqual(len(agent._pending), before, "同一批消息不应再生成新草稿")
            agent.close()

    def test_dry_run_draft_not_written_into_history(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)          # 没有 api_key，dry-run
            platform = MockPlatform()
            platform.current = 0
            agent = Agent(cfg, platform, dry_run=True)
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
    def test_escalation_after_customer_changes_mind(self) -> None:
        """生成期间客户改口说“退款”：必须转人工，不能把旧方案发出去。"""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            platform = MockPlatform()
            platform.current = 0
            agent = Agent(cfg, platform, dry_run=True)
            calls = {"n": 0}

            def fake_generate(*_args, **_kwargs):
                calls["n"] += 1
                platform.sessions[0].messages.append(
                    {"dir": "in", "kind": "text", "text": "不要了，我要退款"})
                return ("按三个为您处理", "fake", 1)

            agent._generate = fake_generate
            items = platform.read_incoming_items()
            image = platform.capture()
            regions = agent.layout.regions(image.width, image.height)
            handled = agent._respond("张三", items, image, regions, False)
            self.assertFalse(handled)
            self.assertNotIn("张三", agent._pending, "改口退款后不应保留自动化草稿")
            self.assertTrue(agent.handoff.is_taken("张三"), "应转入人工接管")
            agent.close()

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


class TestNoCrossCustomerSend(unittest.TestCase):
    def test_approve_requires_bound_peer(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            platform = MockPlatform()
            platform.current = 0
            agent = Agent(cfg, platform, dry_run=True)
            agent._generate = lambda *a, **k: ("张三的草稿", "fake", 1)
            items = platform.read_incoming_items()
            image = platform.capture()
            regions = agent.layout.regions(image.width, image.height)
            agent._respond("张三", items, image, regions, False)
            self.assertIn("张三", agent._pending)

            # 指定李四：不能用张三的草稿，也不能回退
            result = agent.approve_pending("李四，这笔订单报价一百元", peer="李四")
            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "blocked")
            self.assertIn("没有待发送草稿", result["detail"])
            self.assertIn("张三", agent._pending, "张三的草稿不能被李四的请求消耗")

            # 不指定联系人：直接拒绝
            result = agent.approve_pending(None, peer=None)
            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "blocked")
            agent.close()


class TestStaleDraft(unittest.TestCase):
    def test_customer_new_message_invalidates_draft(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            platform = MockPlatform()
            platform.current = 0
            agent = Agent(cfg, platform, dry_run=True)
            agent._generate = lambda *a, **k: ("按两个处理", "fake", 1)
            items = platform.read_incoming_items()
            image = platform.capture()
            regions = agent.layout.regions(image.width, image.height)
            agent._respond("张三", items, image, regions, False)

            platform.sessions[0].messages.append({"dir": "in", "kind": "text", "text": "改成三个"})
            result = agent.approve_pending(peer="张三")
            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "stale")
            self.assertNotIn("张三", agent._pending)
            agent.close()

    def test_human_reply_discards_draft(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            platform = MockPlatform()
            platform.current = 0
            agent = Agent(cfg, platform, dry_run=True)

            def generate(*_a, **_k):
                # 生成期间，人工客服回复了客户
                platform.sessions[0].messages.append({"dir": "out", "kind": "text", "text": "我来处理"})
                return ("机器人插话", "fake", 1)

            agent._generate = generate
            items = platform.read_incoming_items()
            image = platform.capture()
            regions = agent.layout.regions(image.width, image.height)
            handled = agent._respond("张三", items, image, regions, False)
            self.assertFalse(handled)
            self.assertNotIn("张三", agent._pending, "人工已回复时不应保留自动草稿")
            agent.close()


class TestSameTextNewEvent(unittest.TestCase):
    def test_repeated_text_after_reply_is_new_event(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            platform = MockPlatform()
            platform.current = 0
            agent = Agent(cfg, platform, dry_run=True)
            agent._generate = lambda *a, **k: ("在的", "fake", 1)

            first = platform.read_incoming_items()
            image = platform.capture()
            regions = agent.layout.regions(image.width, image.height)
            agent._respond("张三", first, image, regions, False)
            self.assertIn("张三", agent._pending)

            # 模拟已回复：清草稿并把事件标记为已发送
            pending = agent._pending["张三"]
            agent._event_state["张三"]["sent"] = True
            agent.audit.record_send("张三", pending["event_key"], "在的", "verified",
                                    event=pending["event_seq"])
            agent._drop_pending("张三")

            # 客户隔了一会儿又发同样一句
            platform.sessions[0].messages = [
                {"dir": "in", "kind": "text", "text": "在吗？"},
                {"dir": "out", "kind": "text", "text": "在的"},
                {"dir": "in", "kind": "text", "text": "在吗？"},
            ]
            second = platform.read_incoming_items()
            handled = agent._respond("张三", second, image, regions, False)
            self.assertTrue(handled, "内容相同的新消息应正常处理，不能被台账误拦")
            self.assertIn("张三", agent._pending)
            agent.close()


class TestLiveModeSync(unittest.TestCase):
    def test_constructor_syncs_runtime_dry_run(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            cfg.runtime.dry_run = True
            agent = Agent(cfg, MockPlatform(), dry_run=False)   # --live 启动
            self.assertFalse(agent.runtime.dry_run, "启动参数开启 live 时 runtime.dry_run 必须同步")
            agent.close()


class TestWhitelistExact(unittest.TestCase):
    def test_truncated_name_not_authorized(self) -> None:
        whitelist = Whitelist(["张三丰"], allow_group_chat=False)
        self.assertTrue(whitelist.matches("张三")[0], "宽松匹配只用于展示/预检")
        self.assertFalse(whitelist.matches_exact("张三")[0], "自动发送必须完整匹配")
        self.assertTrue(whitelist.matches_exact("张三丰")[0])

    def test_gate_rejects_truncated_name(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            cfg.safety.whitelist = ["张三丰"]
            agent = Agent(cfg, MockPlatform(), dry_run=True)
            agent.set_dry_run(False)
            agent.router.client.config.api_key = "sk-test"
            ok, reason = agent.send_gate("张三")
            self.assertFalse(ok)
            self.assertEqual(reason, "not_whitelisted")
            agent.close()


class TestRepeatProcessing(unittest.TestCase):
    def test_same_session_twice_no_duplicate_memory_or_draft(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            platform = MockPlatform()
            platform.current = 0
            agent = Agent(cfg, platform, dry_run=True)
            calls = {"n": 0}

            def generate(*_a, **_k):
                calls["n"] += 1
                return ("草稿%d" % calls["n"], "fake", 1)

            agent._generate = generate
            items = platform.read_incoming_items()
            image = platform.capture()
            regions = agent.layout.regions(image.width, image.height)
            agent._respond("张三", items, image, regions, False)
            history_first = agent.memory.history_lines("张三", 50)
            agent._respond("张三", items, image, regions, False)
            history_second = agent.memory.history_lines("张三", 50)

            self.assertEqual(calls["n"], 1, "同一批消息不应重复调用模型")
            self.assertEqual(len(history_first), len(history_second),
                             "同一句客户消息不应重复写入记忆")
            self.assertEqual(len(agent._pending), 1)
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


