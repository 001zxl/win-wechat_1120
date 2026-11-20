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


class TestRetryAndRateLimit(unittest.TestCase):
    def test_model_failure_is_retried_later(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            platform = MockPlatform()
            platform.current = 0
            agent = Agent(cfg, platform, dry_run=True)
            calls = {"n": 0}

            def flaky(*_a, **_k):
                calls["n"] += 1
                if calls["n"] == 1:
                    return (None, "fake", 0)      # 第一次模拟 503
                return ("在的", "fake", 1)

            agent._generate = flaky
            items = platform.read_incoming_items()
            image = platform.capture()
            regions = agent.layout.regions(image.width, image.height)
            self.assertFalse(agent._respond("张三", items, image, regions, False))
            state = agent._event_state["张三"]
            self.assertEqual(state["status"], "retry")
            self.assertNotIn("张三", agent._pending, "失败时不应留下可发送的草稿")

            state["next_at"] = 0                 # 模拟冷却到期
            agent.drive_pending_work()
            self.assertEqual(calls["n"], 2, "恢复后应自动重试同一条消息")
            self.assertIn("张三", agent._pending)
            agent.close()

    def test_rate_limited_draft_is_flushed_later(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            platform = MockPlatform()
            platform.current = 0
            agent = Agent(cfg, platform, dry_run=True)
            agent.set_dry_run(False)
            agent.router.client.config.api_key = "sk-test"
            agent._generate = lambda *a, **k: ("在的", "fake", 1)
            agent.limiter.allowed = lambda peer: False        # 限频中

            items = platform.read_incoming_items()
            image = platform.capture()
            regions = agent.layout.regions(image.width, image.height)
            agent._respond("张三", items, image, regions, False)
            self.assertTrue(agent._pending["张三"].get("held_reason"))
            self.assertEqual(platform.sent, [])

            agent.limiter.allowed = lambda peer: True         # 冷却结束
            agent._flush_held()
            self.assertEqual(len(platform.sent), 1, "冷却结束后应补发暂存的草稿")
            self.assertNotIn("张三", agent._pending)
            agent.close()


class TestInputBoxGuard(unittest.TestCase):
    def test_existing_input_text_blocks_send(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            cfg.runtime.dry_run = False
            platform = MockPlatform()
            platform.current = 0
            platform.draft = "这句内部备注还没有确认，不要发："
            image = platform.capture()
            sender = SafeSender(platform, cfg, Whitelist(cfg.safety.whitelist), None)
            result = sender.send("张三", "按一个为您处理", image, regions_for(platform))
            self.assertEqual(result.outcome.detail, "input_not_empty")
            self.assertEqual(platform.sent, [], "输入框有内容时不能发送")


class TestTakeoverDuringWait(unittest.TestCase):
    def test_cancel_while_waiting(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            cfg.runtime.dry_run = False
            platform = MockPlatform()
            platform.current = 0
            image = platform.capture()
            sender = SafeSender(platform, cfg, Whitelist(cfg.safety.whitelist), None)
            sender.risk.plan = lambda peer, text: (1.2, "ok")
            calls = {"n": 0}

            def guard():
                calls["n"] += 1
                return "manual_takeover" if calls["n"] >= 2 else None

            result = sender.send("张三", "按一个为您处理", image, regions_for(platform), guard=guard)
            self.assertEqual(result.outcome.status, "aborted")
            self.assertEqual(result.outcome.detail, "manual_takeover")
            self.assertEqual(platform.sent, [])


class TestConvergenceLimit(unittest.TestCase):
    def test_still_changing_customer_blocks_send(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            platform = MockPlatform()
            platform.current = 0
            agent = Agent(cfg, platform, dry_run=True)
            texts = ["我要一个", "改成两个", "改成三个", "改成四个"]
            calls = {"n": 0}

            def generate(*_a, **_k):
                calls["n"] += 1
                if calls["n"] < len(texts):
                    platform.sessions[0].messages.append(
                        {"dir": "in", "kind": "text", "text": texts[calls["n"]]})
                return ("按本轮看到的需求处理：%s" % texts[calls["n"] - 1], "fake", 1)

            agent._generate = generate
            platform.sessions[0].messages = [{"dir": "in", "kind": "text", "text": texts[0]}]
            items = platform.read_incoming_items()
            image = platform.capture()
            regions = agent.layout.regions(image.width, image.height)
            agent._respond("张三", items, image, regions, False)
            self.assertNotIn("张三", agent._pending, "客户仍在改口时不能发送旧答案")
            self.assertEqual(agent._event_state["张三"]["status"], "retry")
            agent.close()


class TestMultilineBubble(unittest.TestCase):
    def test_two_line_reply_is_recognized(self) -> None:
        class StubLine:
            def __init__(self, text, bbox):
                self.text = text
                self.bbox = bbox
                self.score = 0.95

        class StubOcr:
            available = True

            def __init__(self, sequence):
                self.sequence = list(sequence)

            def recognize(self, _crop):
                return self.sequence.pop(0) if self.sequence else []

        class FakeImage:
            width = 1160
            height = 760

            def crop(self, _box):
                return self

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            old = [StubLine("好的", (600, 100, 700, 120))]
            new = [StubLine("您好，产品目前有货，", (600, 100, 760, 120)),
                   StubLine("请问您需要多少个？", (600, 126, 760, 146))]
            stub = StubOcr([old, old + new, old, old + new])
            sender = SafeSender(MockPlatform(), cfg, Whitelist(cfg.safety.whitelist), stub)
            regions = LayoutCalculator().regions(FakeImage.width, FakeImage.height)
            ok = sender._verify_new_bubble(FakeImage(), FakeImage(), regions,
                                           "您好，产品目前有货，请问您需要多少个？")
            self.assertTrue(ok, "同一气泡的多行文字应先合并再比较")


class TestApproveWithOtherChatOpen(unittest.TestCase):
    def test_approve_switches_back_before_version_check(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            platform = MockPlatform()
            platform.current = 0
            agent = Agent(cfg, platform, dry_run=True)
            agent._generate = lambda *a, **k: ("按一个处理", "fake", 1)
            items = platform.read_incoming_items()
            image = platform.capture()
            regions = agent.layout.regions(image.width, image.height)
            agent._respond("张三", items, image, regions, False)

            agent.set_dry_run(False)
            agent.router.client.config.api_key = "sk-test"
            platform.current = 1          # 微信当前打开的是李四

            def fake_open(peer):
                platform.current = 0
                return True

            agent.open_conversation = fake_open
            result = agent.approve_pending(peer="张三")
            self.assertNotEqual(result["status"], "blocked", result)
            self.assertEqual(result["status"], "verified", result)
            agent.close()


class TestRetryPersistence(unittest.TestCase):
    def test_retry_queue_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            platform = MockPlatform()
            platform.current = 0
            agent = Agent(cfg, platform, dry_run=True)
            agent._generate = lambda *a, **k: (None, "fake", 0)   # 模型一直失败

            items = platform.read_incoming_items()
            image = platform.capture()
            regions = agent.layout.regions(image.width, image.height)
            agent._respond("张三", items, image, regions, False)
            self.assertEqual(agent._event_state["张三"]["status"], "retry")
            rows = agent.audit.list_retries()
            self.assertEqual(len(rows), 1, "失败的待重试事件应落库")
            self.assertEqual(rows[0][0], "张三")
            agent.close()

            # 模拟进程重启：同一套数据库重新建 Agent
            platform2 = MockPlatform()
            platform2.current = 0
            agent2 = Agent(cfg, platform2, dry_run=True)
            self.assertIn("张三", agent2._event_state, "重启后应恢复待重试队列")
            self.assertEqual(agent2._event_state["张三"]["status"], "retry")

            calls = {"n": 0}

            def ok(*_a, **_k):
                calls["n"] += 1
                return ("在的", "fake", 1)

            agent2._generate = ok
            agent2._event_state["张三"]["next_at"] = 0
            agent2.drive_pending_work()
            self.assertEqual(calls["n"], 1, "重启后应自动补回这条消息")
            self.assertIn("张三", agent2._pending)
            agent2.close()

    def test_retry_row_removed_after_success(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            platform = MockPlatform()
            platform.current = 0
            agent = Agent(cfg, platform, dry_run=True)
            agent._generate = lambda *a, **k: ("在的", "fake", 1)
            items = platform.read_incoming_items()
            image = platform.capture()
            regions = agent.layout.regions(image.width, image.height)
            agent._respond("张三", items, image, regions, False)
            # dry-run 下是 held，不应该是 retry；恢复后可被 flush，这里只验证记录不在重试表
            self.assertEqual(agent.audit.list_retries(), [])
            agent.close()


class TestDraftBinding(unittest.TestCase):
    def test_mismatched_draft_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            platform = MockPlatform()
            platform.current = 0
            agent = Agent(cfg, platform, dry_run=True)
            agent._generate = lambda *a, **k: ("按一个处理", "fake", 1)
            items = platform.read_incoming_items()
            image = platform.capture()
            regions = agent.layout.regions(image.width, image.height)
            agent._respond("张三", items, image, regions, False)
            pending = agent._pending["张三"]
            self.assertTrue(pending.get("draft_id"), "草稿必须有唯一 ID")

            agent.set_dry_run(False)
            agent.router.client.config.api_key = "sk-test"
            bad = agent.approve_pending(peer="张三", draft_id="stale-id")
            self.assertFalse(bad["ok"])
            self.assertEqual(bad["status"], "stale")
            self.assertIn("张三", agent._pending, "错误 ID 不应消耗草稿")
            agent.close()


class TestGuardOnAuthorizationChange(unittest.TestCase):
    def _sender(self, cfg, platform):
        return SafeSender(platform, cfg, Whitelist(cfg.safety.whitelist), None)

    def test_paused_cancels_waiting_send(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            cfg.runtime.dry_run = False
            platform = MockPlatform()
            platform.current = 0
            image = platform.capture()
            sender = self._sender(cfg, platform)
            sender.risk.plan = lambda peer, text: (1.2, "ok")
            calls = {"n": 0}

            def guard():
                calls["n"] += 1
                return "paused" if calls["n"] >= 2 else None

            result = sender.send("张三", "在的", image, regions_for(platform), guard=guard)
            self.assertEqual(result.outcome.detail, "paused")
            self.assertEqual(platform.sent, [])

    def test_whitelist_removed_cancels_send(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            cfg.runtime.dry_run = False
            platform = MockPlatform()
            platform.current = 0
            image = platform.capture()
            sender = self._sender(cfg, platform)
            sender.risk.plan = lambda peer, text: (1.0, "ok")
            calls = {"n": 0}

            def guard():
                calls["n"] += 1
                return "not_whitelisted" if calls["n"] >= 2 else None

            result = sender.send("张三", "在的", image, regions_for(platform), guard=guard)
            self.assertEqual(result.outcome.detail, "not_whitelisted")
            self.assertEqual(platform.sent, [])


class TestShortInputDetection(unittest.TestCase):
    def test_single_character_in_input_blocks_send(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            cfg.runtime.dry_run = False
            platform = MockPlatform()
            platform.current = 0
            platform.draft = "不"          # 只有一个字
            image = platform.capture()
            sender = SafeSender(platform, cfg, Whitelist(cfg.safety.whitelist), None)
            result = sender.send("张三", "按一个为您处理", image, regions_for(platform))
            self.assertEqual(result.outcome.detail, "input_not_empty")
            self.assertEqual(platform.sent, [])


class TestHeldFlushFairness(unittest.TestCase):
    def test_two_customers_both_flushed(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            cfg.safety.whitelist = ["张三", "李四"]
            platform = MockPlatform()
            platform.current = 0
            agent = Agent(cfg, platform, dry_run=True)
            agent.set_dry_run(False)
            agent.router.client.config.api_key = "sk-test"
            agent._generate = lambda *a, **k: ("在的", "fake", 1)

            # 两个人各有一条待发草稿，先都被限频挡住
            platform.sessions[1].messages.append({"dir": "in", "kind": "text", "text": "这款有货吗"})
            agent.limiter.allowed = lambda peer: False
            for index, peer in enumerate(("张三", "李四")):
                platform.current = index
                items = platform.read_incoming_items()
                image = platform.capture()
                regions = agent.layout.regions(image.width, image.height)
                agent._respond(peer, items, image, regions, False)
            self.assertEqual(len(agent._pending), 2, "两位客户都应进入待发队列")

            def open_peer(peer):
                for index, session in enumerate(platform.sessions):
                    if session.name == peer:
                        platform.current = index
                        return True
                return False

            agent.open_conversation = open_peer
            agent.limiter.allowed = lambda peer: True
            agent._flush_held()
            self.assertEqual(sorted(name for name, _text in platform.sent), ["张三", "李四"])
            self.assertEqual(agent._pending, {})
            agent.close()


class TestCredentialIsolation(unittest.TestCase):
    def test_keys_do_not_leak_between_providers(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            cfg.ai.base_url = "https://api.deepseek.com/v1"
            cfg.ai.model = "deepseek-chat"
            cfg.ai.models = [
                {"id": "deepseek", "label": "DeepSeek", "model": "deepseek-chat",
                 "base_url": "https://api.deepseek.com/v1"},
                {"id": "qwen", "label": "Qwen", "model": "qwen-plus",
                 "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
            ]
            cfg.ai.model_keys = {"deepseek": "sk-deepseek", "qwen": ""}
            options = {item.id: item for item in cfg.ai.options()}
            self.assertEqual(options["deepseek"].api_key, "sk-deepseek")
            self.assertEqual(options["qwen"].api_key, "", "B 模型不能继承 A 的 key")

            # 删除 key 应立即失效
            cfg.ai.model_keys = {"deepseek": "", "qwen": ""}
            options = {item.id: item for item in cfg.ai.options()}
            self.assertEqual(options["deepseek"].api_key, "")


class TestHandoffCorruption(unittest.TestCase):
    def test_corrupt_file_blocks_autosend(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = os.path.join(tmp, "handoff.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{ this is not json")
            store = HandoffStore(path)
            self.assertTrue(store.corrupt, "损坏文件不能被静默当成空名单")

            cfg = make_config(tmp)
            cfg.handoff_path = path
            agent = Agent(cfg, MockPlatform(), dry_run=True)
            agent.set_dry_run(False)
            agent.router.client.config.api_key = "sk-test"
            ok, reason = agent.send_gate("张三")
            self.assertFalse(ok)
            self.assertEqual(reason, "handoff_corrupt")
            agent.close()

    def test_backup_restores_when_main_file_broken(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = os.path.join(tmp, "handoff.json")
            store = HandoffStore(path)
            store.take("张三", "投诉")
            self.assertTrue(os.path.exists(path + ".bak"))
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("broken")
            restored = HandoffStore(path)
            self.assertFalse(restored.corrupt, "有备份时应自动恢复")
            self.assertTrue(restored.is_taken("张三"))
            self.assertEqual(restored.recovered_from, path + ".bak")


class TestLayoutStructure(unittest.TestCase):
    def test_right_column_groups_two_cards(self) -> None:
        html_path = os.path.join(ROOT, "web", "index.html")
        with open(html_path, "r", encoding="utf-8") as handle:
            html = handle.read()
        self.assertIn('class="right-col"', html)
        self.assertIn('class="card handoff-card"', html)
        self.assertIn('class="card events-card"', html)
        self.assertIn(".right-col { display: flex; flex-direction: column", html)


class TestSummaryPerCustomer(unittest.TestCase):
    def test_other_customers_do_not_trigger_summary(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            cfg.memory.summarize_after = 5
            agent = Agent(cfg, MockPlatform(), dry_run=True)
            for _ in range(3):
                agent.memory.add("张三", "in", "text", "你好")
            for _ in range(10):
                agent.memory.add("李四", "in", "text", "在吗")
            self.assertFalse(agent.memory.needs_summary("张三"),
                             "张三只有 3 条，不能因为李四的消息多就触发摘要")
            self.assertTrue(agent.memory.needs_summary("李四"))
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


