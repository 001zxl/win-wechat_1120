from __future__ import annotations

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wxreply.brain.rag import KnowledgeBase
from wxreply.config import Config
from wxreply.executor.sender import SafeSender
from wxreply.perception.chat import ChatLine, aggregate_incoming
from wxreply.perception.layout import LayoutCalculator
from wxreply.platform.mock import MockPlatform
from wxreply.policy import Guardrails, Whitelist, normalize_name


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
    cfg.runtime.settle_s = 0.01
    cfg.runtime.aggregate_max_wait_s = 0.02
    cfg.runtime.dry_run = True
    cfg.safety.whitelist = ["张三"]
    return cfg


class TestLayout(unittest.TestCase):
    def test_regions_monotonic(self) -> None:
        layout = LayoutCalculator()
        regions = layout.regions(1000, 700)
        self.assertLess(regions.nav[2], regions.session_list[2])
        self.assertLessEqual(regions.session_list[2], regions.chat[0])
        self.assertLessEqual(regions.header[3], regions.messages[1])
        self.assertLessEqual(regions.messages[3], regions.input[1])
        self.assertEqual(regions.chat[2], 1000)
        self.assertEqual(regions.input[3], 700)


class TestPolicy(unittest.TestCase):
    def test_normalize_and_whitelist(self) -> None:
        self.assertEqual(normalize_name(" 3张三 "), "张三")
        whitelist = Whitelist(["张三"], allow_group_chat=False)
        self.assertTrue(whitelist.matches("张三")[0])
        self.assertTrue(whitelist.matches("3张三")[0])
        self.assertFalse(whitelist.matches("李四")[0])
        self.assertFalse(whitelist.matches("产品交流群(25)")[0])

    def test_guardrails(self) -> None:
        guardrails = Guardrails(["转账"], ["投诉"], ["作为AI"], 20)
        self.assertEqual(guardrails.assess_incoming("我要投诉你们").action, "escalate")
        self.assertEqual(guardrails.check_reply("给你转账吧").action, "block")
        self.assertEqual(guardrails.check_reply("作为AI我不能").action, "block")
        result = guardrails.check_reply("这是一段很长的回复内容，超过了二十个字符的限制")
        self.assertEqual(result.action, "reply")
        self.assertLessEqual(len(result.text), 20)


class TestChat(unittest.TestCase):
    def test_aggregate_only_after_last_outgoing(self) -> None:
        lines = [
            ChatLine("你好", 10, 100, "in"),
            ChatLine("这个怎么卖", 20, 700, "out"),
            ChatLine("有优惠吗", 30, 100, "in"),
            ChatLine("我要5000件", 40, 100, "in"),
        ]
        incoming = aggregate_incoming(lines)
        self.assertEqual(len(incoming), 1)
        self.assertIn("有优惠吗", incoming[0])
        self.assertIn("我要5000件", incoming[0])


class TestRag(unittest.TestCase):
    def test_bm25_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "a.md"), "w", encoding="utf-8") as handle:
                handle.write("折叠桌 FT-120 报价 3.2 美金\n\n无关内容：天气很好。")
            kb = KnowledgeBase(tmp)
            hits = kb.search("FT-120 多少钱", top_k=1)
            self.assertTrue(hits)
            self.assertIn("FT-120", hits[0].text)


class TestSenderSafety(unittest.TestCase):
    def test_abort_when_header_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(tmp)
            platform = MockPlatform()
            platform.force_open("李四")  # 当前会话不是目标
            image = platform.capture()
            regions = LayoutCalculator().regions(image.width, image.height)
            sender = SafeSender(platform, cfg, Whitelist(["张三"]), None)
            result = sender.send("张三", "在的", image, regions)
            self.assertEqual(result.outcome.status, "aborted")
            self.assertIn("header", result.outcome.detail)
            self.assertEqual(platform.sent, [])

    def test_verified_send_on_mock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(tmp)
            platform = MockPlatform()
            platform.force_open("张三")
            image = platform.capture()
            regions = LayoutCalculator().regions(image.width, image.height)
            sender = SafeSender(platform, cfg, Whitelist(["张三"]), None)
            result = sender.send("张三", "好的，稍等我看下", image, regions)
            self.assertEqual(result.outcome.status, "verified", result.outcome.detail)
            self.assertEqual(platform.sent[-1][0], "张三")


class TestAgentDryRun(unittest.TestCase):
    def test_cycle_produces_draft(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(tmp)
            platform = MockPlatform()
            from wxreply.agent import Agent

            agent = Agent(cfg, platform, dry_run=True, notify=lambda _msg: None)
            processed = agent.cycle()
            self.assertEqual(processed, 1)
            rows = agent.audit.recent(5)
            self.assertTrue(any(row[2] == "draft" and row[3] == "dry_run" for row in rows))
            self.assertEqual(platform.sent, [])  # dry-run 不发送


class TestMemory(unittest.TestCase):
    def test_memory_roundtrip(self) -> None:
        from wxreply.brain.memory import MemoryStore

        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(os.path.join(tmp, "m.db"), recent_limit=5, summarize_after=3)
            store.add("张三", "in", "text", "你好")
            store.add("张三", "out", "text", "在的")
            store.add("张三", "in", "text", "还有优惠吗")
            lines = store.history_lines("张三")
            self.assertEqual(len(lines), 3)
            self.assertIn("在的", lines[-2])
            self.assertTrue(store.needs_summary("张三"))
            store.set_summary("张三", "客户在问价", store.last_id("张三"))
            self.assertEqual(store.profile("张三")["summary"], "客户在问价")
            self.assertFalse(store.needs_summary("张三"))
            store.close()


class TestTools(unittest.TestCase):
    def test_price_and_stock(self) -> None:
        import json

        from wxreply.brain.tools import ToolBox

        with tempfile.TemporaryDirectory() as tmp:
            price_path = os.path.join(tmp, "price.json")
            with open(price_path, "w", encoding="utf-8") as handle:
                json.dump({
                    "products": [{
                        "sku": "FT-120", "name": "户外折叠桌 FT-120", "aliases": ["折叠桌"],
                        "price_ranges": [{"min_qty": 500, "max_qty": 1999, "price": [2.9, 3.3]}],
                    }],
                    "stock": {"FT-120": 1200},
                }, handle)
            box = ToolBox(price_path, os.path.join(tmp, "leads.jsonl"))
            results = box.prefetch("FT-120 折叠桌 500 件什么价？")
            names = [result.name for result in results]
            self.assertIn("price_lookup", names)
            self.assertIn("2.9", [r for r in results if r.name == "price_lookup"][0].summary)
            stock = box.stock_check("FT-120")
            self.assertTrue(stock.ok)
            self.assertIn("1200", stock.summary)


class TestGroupPolicy(unittest.TestCase):
    def test_group_modes(self) -> None:
        from wxreply.platform.base import MessageItem
        from wxreply.policy import group_allows

        items = [MessageItem(kind="text", text="这个多少钱", sender="小美")]
        self.assertFalse(group_allows("产品交流群(25)", items, False, [], "mention", [])[0])
        self.assertFalse(group_allows("产品交流群(25)", items, True, [], "mention", [])[0])
        self.assertTrue(group_allows("产品交流群(25)", items, True, [], "keyword", ["多少钱"])[0])
        mentioned = [MessageItem(kind="text", text="@我 报价", sender="小美", mention=True)]
        self.assertTrue(group_allows("产品交流群(25)", mentioned, True, [], "mention", [])[0])
        self.assertFalse(group_allows("别的群(10)", mentioned, True, ["产品交流群"], "mention", [])[0])


class TestModelSwitch(unittest.TestCase):
    def test_switch_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(tmp)
            cfg.ai.models = [
                {"id": "deepseek", "label": "DS", "model": "deepseek-chat"},
                {"id": "qwen", "label": "QW", "model": "qwen-plus",
                 "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
            ]
            from wxreply.agent import Agent

            agent = Agent(cfg, MockPlatform(), dry_run=True, notify=lambda _msg: None)
            self.assertEqual(agent.current_model(), "deepseek-chat")
            self.assertTrue(agent.set_model("qwen"))
            self.assertEqual(agent.current_model(), "qwen-plus")
            self.assertEqual(agent.router.client.config.model, "qwen-plus")


class TestMediaMessage(unittest.TestCase):
    def test_image_event_and_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(tmp)
            platform = MockPlatform()
            platform.sessions[0].messages = [
                {"dir": "in", "kind": "image", "text": "", "description": ""},
                {"dir": "in", "kind": "text", "text": "这个桌子有货吗"},
            ]
            from wxreply.agent import Agent

            agent = Agent(cfg, platform, dry_run=True, notify=lambda _msg: None)
            processed = agent.cycle()
            self.assertEqual(processed, 1)
            events = []
            while not agent.events.empty():
                events.append(agent.events.get())
            self.assertTrue(any(event["type"] == "media" for event in events))
            pending = agent.pending_for("张三")
            self.assertIsNotNone(pending)
            self.assertIn("[图片]", pending["incoming"])


class TestSetupChecks(unittest.TestCase):
    def test_ready_and_missing(self) -> None:
        from server import build_setup_checks

        ready = build_setup_checks("win32", True, True, True, ["张三"], True)
        self.assertTrue(ready["ready"])
        missing = build_setup_checks("mock", True, False, False, [], False)
        self.assertFalse(missing["ready"])
        ocr = [item for item in missing["checks"] if item["id"] == "ocr"][0]
        self.assertFalse(ocr["ok"])
        self.assertIn("rapidocr", ocr["hint"])


class TestUnreadBadge(unittest.TestCase):
    def test_unread_cleared_after_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(tmp)
            platform = MockPlatform()
            from wxreply.agent import Agent

            agent = Agent(cfg, platform, dry_run=True, notify=lambda _msg: None)
            agent.cycle()
            conversations = {item["peer"]: item for item in agent.conversations()}
            self.assertEqual(conversations["张三"]["unread"], 0)

    def test_mark_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(tmp)
            from wxreply.agent import Agent

            agent = Agent(cfg, MockPlatform(), dry_run=True, notify=lambda _msg: None)
            agent._touch("李四", status="草稿待发", unread=1)
            agent.mark_read("李四")
            conversations = {item["peer"]: item for item in agent.conversations()}
            self.assertEqual(conversations["李四"]["unread"], 0)


class TestDualTrack(unittest.TestCase):
    def test_pixel_sentinel_skips_unchanged(self) -> None:
        from PIL import Image
        from wxreply.perception.dual import PixelSentinel
        from wxreply.perception.layout import Regions

        image = Image.new("RGB", (400, 300), (245, 245, 245))
        regions = Regions(width=400, height=300, nav=(0, 0, 0, 0),
                          session_list=(0, 0, 150, 300), chat=(150, 0, 400, 300),
                          header=(150, 0, 400, 40), messages=(150, 40, 400, 240),
                          input=(150, 240, 400, 300), send_button=(370, 260, 400, 290))
        sentinel = PixelSentinel()
        self.assertTrue(sentinel.should_scan(image, regions, force=True))
        self.assertTrue(sentinel.should_scan(image, regions))  # 首次记录
        self.assertFalse(sentinel.should_scan(image, regions))  # 未变化、无红点
        changed = image.copy()
        changed.putpixel((200, 100), (0, 0, 0))
        self.assertTrue(sentinel.should_scan(changed, regions))

    def test_colored_bubble_direction(self) -> None:
        from PIL import Image, ImageDraw
        from wxreply.perception.dual import lines_from_ocr_colored
        from wxreply.perception.layout import Regions
        from wxreply.perception.ocr import OcrLine

        image = Image.new("RGB", (400, 300), (245, 245, 245))
        draw = ImageDraw.Draw(image)
        draw.rectangle((250, 60, 390, 110), fill=(149, 236, 105))
        draw.rectangle((160, 150, 300, 200), fill=(255, 255, 255))
        regions = Regions(width=400, height=300, nav=(0, 0, 0, 0),
                          session_list=(0, 0, 150, 300), chat=(150, 0, 400, 300),
                          header=(150, 0, 400, 40), messages=(150, 40, 400, 240),
                          input=(150, 240, 400, 300), send_button=(370, 260, 400, 290))
        lines = [
            OcrLine(text="好的", bbox=(260, 65, 330, 95), score=0.95),
            OcrLine(text="多少钱", bbox=(175, 155, 250, 185), score=0.95),
        ]
        result = lines_from_ocr_colored(image, regions, lines)
        by_text = {item.text: item.direction for item in result}
        self.assertEqual(by_text["好的"], "out")
        self.assertEqual(by_text["多少钱"], "in")


class TestRisk(unittest.TestCase):
    def test_risk_limits(self) -> None:
        from types import SimpleNamespace
        from wxreply.executor.risk import SendRiskController
        safety = SimpleNamespace(risk_enabled=True, risk_min_interval_s=0,
                                 risk_max_per_hour=100, risk_max_per_day=1,
                                 typing_chars_per_second=1000, risk_jitter=0,
                                 risk_max_wait_s=0)
        risk = SendRiskController(safety, now=lambda: 1000.0)
        wait, _reason = risk.plan("张三", "hi")
        self.assertEqual(wait, 0.0)
        risk.record("张三")
        wait, reason = risk.plan("张三", "hi")
        self.assertIsNone(wait)
        self.assertIn("上限", reason)


class TestInbound(unittest.TestCase):
    def test_parse_line_and_telegram(self) -> None:
        from wxreply.channels.inbound import parse_inbound
        line = {"events": [{"type": "message",
                            "source": {"userId": "U123"},
                            "message": {"type": "text", "text": "多少钱"}}]}
        message = parse_inbound("line", line)
        self.assertEqual((message.peer, message.text), ("U123", "多少钱"))
        telegram = {"message": {"chat": {"id": 1001}, "text": "有库存吗"}}
        message = parse_inbound("telegram", telegram)
        self.assertEqual((message.peer, message.text), ("1001", "有库存吗"))
        self.assertIsNone(parse_inbound("line", {"events": []}))

    def test_handle_inbound_does_not_touch_wechat(self) -> None:
        import json
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(tmp)
            with open(cfg.tools.price_file, "w", encoding="utf-8") as handle:
                json.dump({"products": [{"sku": "FT-120", "name": "折叠桌",
                                          "aliases": ["折叠桌"],
                                          "price_ranges": [{"min_qty": 500, "max_qty": 1999,
                                                            "price": [2.9, 3.3]}]}],
                           "stock": {"FT-120": 10}}, handle)
            from wxreply.agent import Agent
            platform = MockPlatform()
            agent = Agent(cfg, platform, dry_run=True, notify=lambda _msg: None)
            result = agent.handle_inbound("line:U123", "折叠桌 500 件多少钱", auto=True)
            self.assertIsNotNone(result)
            self.assertEqual(platform.sent, [])
            history = agent.conversation("line:U123")
            self.assertEqual([item["dir"] for item in history], ["in", "out"])


class TestStrictWhitelist(unittest.TestCase):
    def test_strict_matching(self) -> None:
        from wxreply.policy import is_name_matched_strict, is_service_account, Whitelist

        self.assertTrue(is_name_matched_strict("张三", ["张三"])[0])
        self.assertFalse(is_name_matched_strict("张三丰", ["张三"])[0])
        self.assertTrue(is_name_matched_strict("张三", ["张三丰"])[0])
        self.assertTrue(is_service_account("服务号"))
        self.assertFalse(is_service_account("张须霞"))
        whitelist = Whitelist(["张三", "文件传输助手"])
        self.assertTrue(whitelist.matches("张三")[0])
        self.assertTrue(whitelist.matches("文件传输助手")[0])
        self.assertFalse(whitelist.matches("服务号")[0])
        self.assertFalse(whitelist.matches("吃喝玩乐(25)")[0])


class TestChannels(unittest.TestCase):
    def test_contact_key(self) -> None:
        from wxreply.channels import contact_key
        self.assertEqual(contact_key("line", "U123"), "line:U123")

    def test_webhook_payloads(self) -> None:
        from wxreply.channels import WebhookChannel
        line = WebhookChannel("line", {"provider": "line", "endpoint": "http://x", "token": "t"})
        self.assertEqual(line._payload("U1", "hi")["to"], "U1")
        self.assertEqual(line._payload("group:G1", "hi")["groupId"], "G1")
        telegram = WebhookChannel("tg", {"provider": "telegram", "endpoint": "http://x"})
        self.assertEqual(telegram._payload("100", "hi"), {"chat_id": "100", "text": "hi"})


class TestSimulate(unittest.TestCase):
    def test_simulate_in_project(self) -> None:
        import json
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(tmp)
            with open(cfg.tools.price_file, "w", encoding="utf-8") as handle:
                json.dump({
                    "products": [{"sku": "FT-120", "name": "户外折叠桌 FT-120",
                                  "aliases": ["折叠桌"],
                                  "price_ranges": [{"min_qty": 500, "max_qty": 1999,
                                                    "price": [2.9, 3.3]}]}],
                    "stock": {"FT-120": 1200},
                }, handle)
            platform = MockPlatform()
            from wxreply.agent import Agent

            agent = Agent(cfg, platform, dry_run=True, notify=lambda _msg: None)
            result = agent.simulate_incoming("测试客户", "FT-120 折叠桌 500 件什么价？", auto=True)
            self.assertIsNotNone(result)
            self.assertIn("2.9", result["reply"])
            self.assertTrue(result["peer"].startswith("【模拟】"))
            self.assertEqual(platform.sent, [])  # 项目内模拟，不操作微信
            history = agent.conversation(result["peer"])
            self.assertEqual(len(history), 2)
            self.assertEqual(history[0]["dir"], "in")
            self.assertEqual(history[1]["dir"], "out")

    def test_simulate_escalate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(tmp)
            cfg.safety.escalate_keywords = ["投诉"]
            from wxreply.agent import Agent

            agent = Agent(cfg, MockPlatform(), dry_run=True, notify=lambda _msg: None)
            result = agent.simulate_incoming("投诉客户", "我要投诉你们", auto=True)
            self.assertTrue(result.get("escalated"))
            self.assertIn("转人工", result["reply"])


class TestSettings(unittest.TestCase):
    def test_settings_overlay(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "data"))
            with open(os.path.join(tmp, "config.yaml"), "w", encoding="utf-8") as handle:
                handle.write("ai:\n  models:\n    - {id: ds, model: deepseek-chat}\n    - {id: qw, model: qwen-plus}\n")
            with open(os.path.join(tmp, "data", "settings.json"), "w", encoding="utf-8") as handle:
                json.dump({"api_keys": {"qw": "sk-test"}, "whitelist": ["王五"],
                           "layout": {"nav_ratio": 0.07}, "vision_model": "qwen-vl-plus"}, handle)
            from wxreply.config import load_config

            cfg = load_config(os.path.join(tmp, "config.yaml"))
            self.assertEqual(cfg.safety.whitelist, ["王五"])
            self.assertAlmostEqual(cfg.layout.nav_ratio, 0.07, places=3)
            self.assertEqual(cfg.ai.vision_model, "qwen-vl-plus")
            options = {option.id: option for option in cfg.ai.options()}
            self.assertEqual(options["qw"].api_key, "sk-test")

    def test_agent_apply_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(tmp)
            cfg.ai.models = [{"id": "ds", "model": "deepseek-chat"},
                             {"id": "qw", "model": "qwen-plus"}]
            from wxreply.agent import Agent

            agent = Agent(cfg, MockPlatform(), dry_run=True, notify=lambda _msg: None)
            agent.apply_settings({"api_keys": {"qw": "sk-x"}, "whitelist": ["王五"],
                                  "layout": {"list_ratio": 0.4}})
            self.assertEqual(cfg.safety.whitelist, ["王五"])
            self.assertAlmostEqual(cfg.layout.list_ratio, 0.4, places=3)
            options = {item["id"]: item for item in agent.model_options()}
            self.assertTrue(options["qw"]["has_key"])


if __name__ == "__main__":
    unittest.main()
