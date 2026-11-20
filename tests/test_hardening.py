"""本轮复查 9 项问题的回归测试。

每一条都对应一个"真实会用错"的场景，用固定假模型 + 模拟微信验证，
不调用真实模型、不发送真实微信。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.test_reliability import make_config
from wxreply.agent import Agent
from wxreply.platform.mock import MockPlatform

FAQ = """# 配送与售后

## 配送
偏远地区（西藏/新疆/内蒙古）可以配送，走邮政小包，运费另计。
常规地区满 500 件包邮。

## 售后
质量问题 30 天内可退换，运输破损需 48 小时内提供照片。

## 规格
折叠桌展开尺寸 120cm × 60cm × 70cm，承重 50kg。

## 报价
FT-120 500-1999 件 FOB 2.9-3.3 美金。
"""

CATALOG = {
    "products": [{
        "sku": "FT-120", "name": "户外折叠桌 FT-120", "aliases": ["折叠桌"],
        "price_ranges": [{"min_qty": 500, "max_qty": 1999, "price": [2.9, 3.3]}],
    }],
    "stock": {"FT-120": 1200},
}


def build(tmp: str, messages, *, kb: bool = False, catalog: bool = False,
          dry_run: bool = True):
    cfg = make_config(tmp)
    cfg.kb_dir = os.path.join(tmp, "kb")
    os.makedirs(cfg.kb_dir, exist_ok=True)
    if kb:
        with open(os.path.join(cfg.kb_dir, "faq.md"), "w", encoding="utf-8") as handle:
            handle.write(FAQ)
    if catalog:
        with open(cfg.tools.price_file, "w", encoding="utf-8") as handle:
            json.dump(CATALOG, handle, ensure_ascii=False)
    platform = MockPlatform()
    platform.current = 0
    platform.sessions[0].messages = [{"dir": "in", "kind": "text", "text": text}
                                     if isinstance(text, str) else text
                                     for text in messages]
    agent = Agent(cfg, platform, dry_run=dry_run, notify=lambda _msg: None)
    return agent, platform


def respond(agent, platform, peer: str = "张三"):
    items = platform.read_incoming_items()
    image = platform.capture()
    regions = agent.layout.regions(image.width, image.height)
    return agent._respond(peer, items, image, regions, False)


class TestEvidenceRecheck(unittest.TestCase):
    """问题 1：生成期间追加的消息必须重新判定依据。"""

    def test_appended_price_question_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            agent, platform = build(tmp, ["你好"])
            calls = {"n": 0}

            def fake_generate(*_a, **_k):
                calls["n"] += 1
                if calls["n"] == 1:
                    platform.sessions[0].messages.append(
                        {"dir": "in", "kind": "text", "text": "ZX-999 多少钱"})
                return ("ZX-999 是 5 美金一件", "fake", 1)

            agent._generate = fake_generate
            handled = respond(agent, platform)
            self.assertFalse(handled)
            self.assertNotIn("张三", agent._pending, "追加报价后不能留下可发送草稿")
            self.assertEqual(platform.sent, [])
            self.assertTrue(agent.handoff.is_taken("张三"), "没有依据的报价必须转人工")
            agent.close()

    def test_appended_shipping_question_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            agent, platform = build(tmp, ["你好"])
            calls = {"n": 0}

            def fake_generate(*_a, **_k):
                calls["n"] += 1
                if calls["n"] == 1:
                    platform.sessions[0].messages.append(
                        {"dir": "in", "kind": "text", "text": "可以送到西藏吗"})
                return ("可以的，我们发西藏包邮", "fake", 1)

            agent._generate = fake_generate
            respond(agent, platform)
            self.assertNotIn("张三", agent._pending)
            self.assertTrue(agent.handoff.is_taken("张三"))
            agent.close()


class TestFactCoverage(unittest.TestCase):
    """问题 2：无依据分流要覆盖配送/售后/规格，而不仅是价格。"""

    def _ask(self, text: str, kb: bool = False):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        agent, platform = build(tmp.name, [text], kb=kb)
        agent._generate = lambda *_a, **_k: ("我查一下再回复您", "fake", 1)
        respond(agent, platform)
        return agent, platform

    def test_shipping_without_kb_goes_to_human(self) -> None:
        agent, _platform = self._ask("可以送到西藏吗")
        self.assertTrue(agent.handoff.is_taken("张三"))
        self.assertNotIn("张三", agent._pending)
        agent.close()

    def test_after_sales_without_kb_goes_to_human(self) -> None:
        agent, _platform = self._ask("坏了能退吗")
        self.assertTrue(agent.handoff.is_taken("张三"))
        agent.close()

    def test_spec_without_kb_goes_to_human(self) -> None:
        agent, _platform = self._ask("尺寸多大")
        self.assertTrue(agent.handoff.is_taken("张三"))
        agent.close()

    def test_shipping_with_kb_is_answered(self) -> None:
        agent, _platform = self._ask("可以送到西藏吗", kb=True)
        self.assertFalse(agent.handoff.is_taken("张三"), "有配送依据时不该转人工")
        draft = agent.pending_for("张三")
        self.assertIsNotNone(draft)
        self.assertTrue(draft["knowledge"]["sources"], "草稿必须记录用了哪些知识片段")
        agent.close()

    def test_repeated_missing_field_escalates(self) -> None:
        """连续追问两次还问不出型号，就不再追问，转人工。"""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            agent, platform = build(tmp, ["这个多少钱"])
            respond(agent, platform)
            self.assertIn("张三", agent._pending, "第一次缺型号应追问")
            self.assertIn("哪一款", agent._pending["张三"]["reply"])

            platform.sessions[0].messages.append(
                {"dir": "in", "kind": "text", "text": "就这个，多少钱"})
            respond(agent, platform)
            self.assertEqual(agent._pending["张三"]["verdict"]["action"], "ask")

            platform.sessions[0].messages.append(
                {"dir": "in", "kind": "text", "text": "还是这个"})
            respond(agent, platform)
            self.assertNotIn("张三", agent._pending, "第三次不应再追问")
            self.assertTrue(agent.handoff.is_taken("张三"))
            agent.close()

    def test_invented_number_in_reply_blocks(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            agent, platform = build(tmp, ["可以送到西藏吗"], kb=True)
            # 88 在知识库/客户原话里都不存在：属于模型自己编的运费
            agent._generate = lambda *_a, **_k: ("运费 88 美金，3 天到", "fake", 1)
            respond(agent, platform)
            self.assertNotIn("张三", agent._pending, "编造运费的回复不能进入待发")
            self.assertTrue(agent.handoff.is_taken("张三"))
            agent.close()


class TestMediaExit(unittest.TestCase):
    """问题 3：图片读不出来要有确定的人工出口。"""

    def test_image_without_vision_asks_then_handoffs(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            agent, platform = build(tmp, [
                {"dir": "in", "kind": "image", "text": "", "description": ""},
                {"dir": "in", "kind": "text", "text": "这个桌子有货吗"},
            ])
            respond(agent, platform)
            draft = agent.pending_for("张三")
            self.assertIsNotNone(draft, "第一次读不出图应追问，而不是直接转人工")
            self.assertIn("文字", draft["reply"])
            self.assertNotIn("有货", draft["reply"])
            self.assertEqual(draft["verdict"]["action"], "ask")

            agent._drop_pending("张三")
            # 客户又发一张图（新消息批次）：同一张图读不出来第二次必须转人工
            platform.sessions[0].messages.append(
                {"dir": "in", "kind": "image", "text": "", "description": ""})
            respond(agent, platform)
            self.assertTrue(agent.handoff.is_taken("张三"),
                            "同一张图追问过一次仍读不出，必须转人工")
            agent.close()

    def test_media_detection_not_bound_to_vision(self) -> None:
        """没有视觉模型时，图片检测不能跟着被关掉。"""
        import wxreply.agent as agent_module

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            agent, platform = build(tmp, ["在吗"])
            captured = {}

            def fake_extract(image, regions, lines, save_dir, detect_media=True):
                captured["detect_media"] = detect_media
                return []

            def fake_recognize(_crop):
                return []

            class StubOcr:
                available = True
                load_error = None
                backend_name = "stub"

                def recognize(self, crop):
                    return fake_recognize(crop)

            agent.ocr = StubOcr()
            platform.read_incoming_items = lambda: None   # 走 OCR 分支
            original = agent_module.extract_items
            agent_module.extract_items = fake_extract
            try:
                image = platform.capture()
                regions = agent.layout.regions(image.width, image.height)
                agent._read_items(image, regions)
            finally:
                agent_module.extract_items = original
            self.assertTrue(captured.get("detect_media"), "媒体检测不得依赖视觉模型")
            agent.close()


class TestToolResultTypes(unittest.TestCase):
    """问题 4：工具结果全链路统一类型。"""

    def test_regenerate_with_tool_results(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            agent, _platform = build(tmp, ["FT-120 800 件多少钱"], catalog=True)
            result = agent.simulate_incoming("测试客户", "FT-120 800 件多少钱", auto=False)
            self.assertIsNotNone(result)
            peer = result["peer"]
            pending = agent._pending[peer]
            self.assertTrue(pending["tool_results"], "报价查询结果应存进草稿")
            again = agent.regenerate_pending(peer=peer)
            self.assertIsNotNone(again, "带工具结果的草稿必须能重新生成")
            self.assertTrue(again["ok"])
            self.assertIn("2.9", agent._pending[peer]["reply"])
            agent.close()

    def test_regenerate_accepts_dict_tool_results(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            agent, _platform = build(tmp, ["FT-120 800 件多少钱"], catalog=True)
            result = agent.simulate_incoming("测试客户", "FT-120 800 件多少钱", auto=False)
            peer = result["peer"]
            pending = agent._pending[peer]
            # 模拟旧版本遗留的字典形态
            pending["tool_results"] = [item.as_dict() for item in pending["tool_results"]]
            again = agent.regenerate_pending(peer=peer)
            self.assertIsNotNone(again)
            self.assertTrue(again["ok"])
            agent.close()


class TestDraftVersion(unittest.TestCase):
    """问题 5：草稿版本必须闭合。"""

    def _draft(self, tmp: str):
        agent, platform = build(tmp, ["FT-120 800 件多少钱"], catalog=True)
        result = agent.simulate_incoming("测试客户", "FT-120 800 件多少钱", auto=False)
        return agent, platform, result["peer"]

    def test_regenerate_changes_draft_id(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            agent, _platform, peer = self._draft(tmp)
            first = agent._pending[peer]["draft_id"]
            agent.regenerate_pending(peer=peer)
            second = agent._pending[peer]["draft_id"]
            self.assertNotEqual(first, second, "重新生成必须换新版本号")
            self.assertEqual(agent._pending[peer]["draft_version"], 2)
            agent.close()

    def test_approve_requires_draft_id(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            agent, _platform, peer = self._draft(tmp)
            result = agent.approve_pending(peer=peer)
            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "blocked")
            self.assertIn("draft_id", result["detail"])
            agent.close()

    def test_old_draft_id_rejected(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            agent, _platform, peer = self._draft(tmp)
            first = agent._pending[peer]["draft_id"]
            agent.regenerate_pending(peer=peer)
            stale = agent.approve_pending(peer=peer, draft_id=first)
            self.assertFalse(stale["ok"])
            self.assertEqual(stale["status"], "stale")
            self.assertIn(peer, agent._pending, "旧版本被拒后草稿仍在")
            agent.close()

    def test_current_draft_id_sends(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            agent, _platform, peer = self._draft(tmp)
            draft_id = agent._pending[peer]["draft_id"]
            result = agent.approve_pending(peer=peer, draft_id=draft_id)
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["status"], "verified")
            agent.close()

    def test_event_seq_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            agent, _platform, peer = self._draft(tmp)
            draft_id = agent._pending[peer]["draft_id"]
            result = agent.approve_pending(peer=peer, draft_id=draft_id, event_seq=99)
            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "stale")
            agent.close()


class TestPaddleV3Parsing(unittest.TestCase):
    """问题 6：PaddleOCR 3.x 结果格式。"""

    def test_res_envelope(self) -> None:
        from wxreply.perception.ocr import PaddleOcrBackend

        raw = [{"res": {
            "rec_texts": ["你好", "世界"],
            "rec_scores": [0.99, 0.90],
            "rec_polys": [[[0, 0], [10, 0], [10, 10], [0, 10]],
                          [[0, 20], [30, 20], [30, 40], [0, 40]]],
        }}]
        lines = PaddleOcrBackend.parse_result(raw)
        self.assertEqual([line.text for line in lines], ["你好", "世界"])
        self.assertAlmostEqual(lines[0].score, 0.99, places=3)
        self.assertEqual(lines[1].bbox, (0.0, 20.0, 30.0, 40.0))

    def test_flat_dict(self) -> None:
        from wxreply.perception.ocr import PaddleOcrBackend

        raw = [{"rec_texts": ["你好"], "rec_scores": [0.99],
                "rec_polys": [[[0, 0], [10, 0], [10, 10], [0, 10]]]}]
        self.assertEqual(len(PaddleOcrBackend.parse_result(raw)), 1)

    def test_json_property_object(self) -> None:
        from wxreply.perception.ocr import PaddleOcrBackend

        class Result(dict):
            @property
            def json(self):
                return {"res": {"rec_texts": ["报价单"], "rec_scores": [0.8],
                                "rec_polys": [[[1, 2], [11, 2], [11, 12], [1, 12]]]}}

        lines = PaddleOcrBackend.parse_result([Result()])
        self.assertEqual(lines[0].text, "报价单")
        self.assertEqual(lines[0].bbox, (1.0, 2.0, 11.0, 12.0))

    def test_rec_boxes_only(self) -> None:
        from wxreply.perception.ocr import PaddleOcrBackend

        raw = [{"rec_texts": ["FT-120"], "rec_scores": [0.7],
                "rec_polys": [], "rec_boxes": [[5, 6, 50, 26]]}]
        lines = PaddleOcrBackend.parse_result(raw)
        self.assertEqual(lines[0].text, "FT-120")
        self.assertEqual(lines[0].bbox, (5.0, 6.0, 50.0, 26.0))

    def test_v2_payload_still_works(self) -> None:
        from wxreply.perception.ocr import PaddleOcrBackend

        page = [[[[0, 0], [10, 0], [10, 10], [0, 10]], ("你好", 0.9)]]
        lines = PaddleOcrBackend.parse_result([page])       # 2.x：list[page]
        self.assertEqual(lines[0].text, "你好")
        self.assertAlmostEqual(lines[0].score, 0.9, places=3)
        # 少一层 page 列表的旧形态也要能解析
        loose = PaddleOcrBackend.parse_result(page)
        self.assertEqual(loose[0].text, "你好")

    def test_empty_and_none(self) -> None:
        from wxreply.perception.ocr import PaddleOcrBackend

        self.assertEqual(PaddleOcrBackend.parse_result(None), [])
        self.assertEqual(PaddleOcrBackend.parse_result([{"res": {}}]), [])


class TestTraceability(unittest.TestCase):
    """问题 7 + 8：来源/分数/版本可追溯，多轮指代进入检索。"""

    def test_chunk_carries_score_source_id(self) -> None:
        from wxreply.brain.rag import KnowledgeBase

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            with open(os.path.join(tmp, "faq.md"), "w", encoding="utf-8") as handle:
                handle.write(FAQ)
            kb = KnowledgeBase(tmp, "bm25")
            hits = kb.search("西藏 配送", top_k=2)
            self.assertTrue(hits)
            hit = hits[0]
            self.assertGreater(hit.score, 0)
            self.assertTrue(hit.source.endswith("faq.md"))
            self.assertIn("faq.md", hit.chunk_id)

    def test_draft_binds_knowledge(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            agent, platform = build(tmp, ["可以送到西藏吗"], kb=True)
            agent._generate = lambda *_a, **_k: ("偏远地区可以配送，运费另计", "fake", 1)
            respond(agent, platform)
            draft = agent.pending_for("张三")
            knowledge = draft["knowledge"]
            self.assertEqual(knowledge["provider"], "local")
            self.assertEqual(knowledge["kb_version"], agent.kb.version)
            self.assertTrue(knowledge["sources"])
            self.assertTrue(knowledge["sources"][0]["source"])
            agent.close()

    def test_audit_row_records_kb_version(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            agent, platform = build(tmp, ["可以送到西藏吗"], kb=True)
            agent._generate = lambda *_a, **_k: ("偏远地区可以配送，运费另计", "fake", 1)
            respond(agent, platform)
            rows = agent.audit.recent(10)
            draft_rows = [row for row in rows if row[2] == "draft"]
            self.assertTrue(draft_rows)
            self.assertEqual(draft_rows[0][7], agent.kb.version)
            self.assertIn("sources", draft_rows[0][8] or "")
            agent.close()

    def test_context_query_uses_recent_sku(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            agent, _platform = build(tmp, ["在吗"])
            query, ambiguous = agent._context_query(
                "张三", "这个多少钱", ["对方：SKU-XYZ 500 件 2.9 美金"])
            self.assertEqual(ambiguous, "")
            self.assertIn("SKU-XYZ", query, "多轮指代要把历史型号带进检索")
            agent.close()

    def test_context_query_ambiguous_asks(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            agent, _platform = build(tmp, ["在吗"])
            query, ambiguous = agent._context_query(
                "张三", "这个多少钱",
                ["对方：FT-120 500 件", "对方：FC-200 300 件"])
            self.assertTrue(ambiguous, "型号不唯一时应追问")
            self.assertNotIn("FT-120", query)
            agent.close()

    def test_context_remembers_confirmed_product(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            agent, _platform = build(tmp, ["在吗"], catalog=True)
            agent.simulate_incoming("测试客户", "FT-120 500 件多少钱", auto=True)
            peer = "【模拟】测试客户"
            query, _ambiguous = agent._context_query(peer, "这个有货吗", [])
            self.assertIn("FT-120", query)
            agent.close()


class TestCredentialRevocation(unittest.TestCase):
    """问题 9：撤销凭据不能留继承后门。"""

    def test_revoke_blocks_top_level_key(self) -> None:
        from wxreply.config import AIConfig

        config = AIConfig(api_key="sk-top", model="deepseek-chat",
                          models=[], model_keys={})
        self.assertEqual(config.options()[0].api_key, "sk-top")
        config.revoke(["deepseek-chat"])
        self.assertEqual(config.options()[0].api_key, "", "撤销后不能再从顶层 key 继承")
        self.assertEqual(config.api_key, "")

    def test_settings_empty_key_means_delete(self) -> None:
        from wxreply.settings import SettingsStore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            store = SettingsStore(os.path.join(tmp, "settings.json"))
            store.set_api_keys({"ds": "sk-1", "qw": "sk-2"})
            store.set_api_keys({"ds": ""})
            self.assertNotIn("ds", store.api_keys())
            self.assertIn("ds", store.revoked())
            store.set_api_keys({"ds": "sk-3"})
            self.assertEqual(store.api_keys()["ds"], "sk-3")
            self.assertNotIn("ds", store.revoked(), "重新填 key 后撤销标记要解除")

    def test_settings_overlay_keeps_revocation(self) -> None:
        from wxreply.config import load_config

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            with open(os.path.join(tmp, "config.yaml"), "w", encoding="utf-8") as handle:
                handle.write("ai:\n  api_key: sk-top\n  model: deepseek-chat\n")
            os.makedirs(os.path.join(tmp, "data"))
            with open(os.path.join(tmp, "data", "settings.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"api_keys": {"deepseek-chat": ""},
                           "revoked": ["deepseek-chat"]}, handle)
            cfg = load_config(os.path.join(tmp, "config.yaml"))
            self.assertEqual(cfg.ai.options()[0].api_key, "")

    def test_agent_apply_settings_clears_ready(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            agent, _platform = build(tmp, ["在吗"])
            agent.config.ai.api_key = "sk-test"
            agent.config.ai.models = []
            agent.router = __import__("wxreply.brain.llm", fromlist=["ModelRouter"]).ModelRouter(
                agent.config.ai, dry_run=False)
            self.assertTrue(agent.router.client.ready)
            agent.apply_settings({"api_keys": {"deepseek-chat": ""},
                                  "revoked": ["deepseek-chat"]})
            self.assertFalse(agent.router.client.ready, "撤销后必须立刻变为未就绪")
            agent.close()

    def test_revoke_all_clears_vision_and_embedding(self) -> None:
        from wxreply.brain.vision import VisionClient
        from wxreply.config import AIConfig

        config = AIConfig(api_key="sk-top", vision_api_key="sk-v",
                          embedding_api_key="sk-e", vision_model="qwen-vl")
        config.revoke(["*"])
        self.assertEqual(config.api_key, "")
        self.assertEqual(config.vision_api_key, "")
        self.assertEqual(config.embedding_api_key, "")
        self.assertFalse(config._fallback_key(config.base_url))
        self.assertFalse(VisionClient(config).available, "撤销后视觉不能再从聊天 key 兜底")

    def test_revoke_vision_only(self) -> None:
        from wxreply.brain.vision import VisionClient
        from wxreply.config import AIConfig

        config = AIConfig(api_key="sk-top", vision_model="qwen-vl")
        self.assertTrue(VisionClient(config).available, "同网关时视觉可复用聊天 key")
        config.revoke(["vision"])
        self.assertFalse(VisionClient(config).available)
        self.assertTrue(VisionClient(AIConfig(api_key="sk-new", vision_model="qwen-vl")).available)

    def test_new_key_works_after_revoke_all(self) -> None:
        """撤销全部之后重新录入 key 必须能恢复，不能被通配符永久锁死。"""
        from wxreply.settings import SettingsStore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            store = SettingsStore(os.path.join(tmp, "settings.json"))
            store.set_api_keys({"deepseek-chat": "sk-old"})
            store.revoke(["*"])
            self.assertNotIn("deepseek-chat", store.api_keys())
            self.assertIn("deepseek-chat", store.revoked())
            store.set_api_keys({"deepseek-chat": "sk-new"})
            self.assertNotIn("deepseek-chat", store.revoked())

    def test_revoke_default_still_allows_explicit_key(self) -> None:
        from wxreply.config import AIConfig

        config = AIConfig(api_key="sk-top", model="deepseek-chat", models=[],
                          model_keys={"deepseek-chat": "sk-own"})
        config.revoke(["default"])
        self.assertEqual(config.options()[0].api_key, "sk-own",
                         "撤销顶层继承不应连带作废用户显式录入的模型 key")

    def test_agent_apply_settings_can_restore(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            agent, _platform = build(tmp, ["在吗"])
            agent.config.ai.models = []
            agent.apply_settings({"api_keys": {"deepseek-chat": ""},
                                  "revoked": ["deepseek-chat"]})
            self.assertFalse(agent.router.client.ready)
            agent.apply_settings({"api_keys": {"deepseek-chat": "sk-new"},
                                  "revoked": []})
            self.assertTrue(agent.router.client.ready, "重新录入 key 后必须恢复就绪")
            agent.close()


if __name__ == "__main__":
    unittest.main()
