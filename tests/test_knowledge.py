from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wxreply.brain.rag import KnowledgeBase
from wxreply.brain.tools import ToolBox
from wxreply.channels import ChannelConfig, build_registry


def make_catalog(tmpdir: str) -> str:
    path = os.path.join(tmpdir, "price.json")
    data = {
        "products": [
            {
                "sku": "FT-120",
                "name": "户外折叠桌 FT-120",
                "aliases": ["折叠桌"],
                "price_ranges": [
                    {"min_qty": 0, "max_qty": 500, "price": [3.2, 3.6]},
                    {"min_qty": 501, "max_qty": 2000, "price": [2.9, 3.3]},
                ],
            }
        ],
        "stock": {"FT-120": 120},
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False)
    return path


class TestPriceLookupSafety(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.toolbox = ToolBox(make_catalog(self.tmp.name),
                               os.path.join(self.tmp.name, "leads.jsonl"))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_unknown_product_is_not_quoted(self) -> None:
        result = self.toolbox.price_lookup("帐篷 500 件")
        self.assertFalse(result.ok)
        self.assertIn("没有匹配到产品", result.summary)
        self.assertEqual(result.data.get("need"), "product")

    def test_single_digit_quantity_is_parsed(self) -> None:
        self.assertEqual(self.toolbox._parse_quantity("折叠桌 1件"), 1)
        self.assertIsNone(self.toolbox._parse_quantity("折叠桌"))

    def test_missing_quantity_asks_instead_of_defaulting(self) -> None:
        result = self.toolbox.price_lookup("折叠桌")
        self.assertFalse(result.ok)
        self.assertEqual(result.data.get("need"), "quantity")

    def test_quantity_outside_bands_is_not_quoted(self) -> None:
        result = self.toolbox.price_lookup("折叠桌 5000件")
        self.assertFalse(result.ok)
        self.assertEqual(result.data.get("need"), "band")
        self.assertIn("不要套用", result.summary)

    def test_valid_quantity_returns_band(self) -> None:
        result = self.toolbox.price_lookup("折叠桌 800件")
        self.assertTrue(result.ok)
        self.assertEqual(result.data.get("range", {}).get("min_qty"), 501)

    def test_prefetch_passes_failure_as_constraint(self) -> None:
        results = self.toolbox.prefetch("这个多少钱")
        self.assertTrue(results, "没匹配到产品也要把约束交给模型")
        self.assertFalse(results[0].ok)
        self.assertIn("型号", results[0].summary)


class TestKnowledgeBase(unittest.TestCase):
    def _write(self, directory: str, name: str, text: str) -> None:
        with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
            handle.write(text)

    def test_no_hit_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            self._write(tmp, "a.md", "# 产品\n户外折叠桌 FT-120，承重 50kg。")
            kb = KnowledgeBase(tmp, "bm25")
            self.assertEqual(kb.search("量子计算机维修", top_k=3), [],
                             "关键词完全没命中时要允许没有答案")

    def test_reload_and_version(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            self._write(tmp, "a.md", "# 价格\nFT-120 售价 100 元。")
            kb = KnowledgeBase(tmp, "bm25")
            first_version = kb.version
            first = kb.search("FT-120 售价", top_k=1)
            self.assertTrue(first)
            self.assertIn("100", first[0].text)

            self._write(tmp, "a.md", "# 价格\nFT-120 售价 200 元。")
            self.assertEqual(kb.reload(), 1)
            self.assertNotEqual(kb.version, first_version, "内容改动后版本必须变化")
            second = kb.search("FT-120 售价", top_k=1)
            self.assertIn("200", second[0].text, "重新加载后应读到新价格")


class TestToolBudget(unittest.TestCase):
    def test_tool_rounds_do_not_consume_network_retries(self) -> None:
        from wxreply.brain.llm import LLMClient
        from wxreply.config import AIConfig
        import wxreply.brain.llm as llm_module

        config = AIConfig(api_key="sk-test", base_url="https://example.invalid/v1",
                          model="test-model", timeout_s=1)
        client = LLMClient(config, dry_run=False)

        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload
                self.status_code = 200
                self.text = ""

            def json(self):
                return self._payload

        responses = [
            {"choices": [{"message": {"content": "", "tool_calls": [
                {"id": "1", "function": {"name": "price_lookup", "arguments": "{}"}}]}}]},
            {"choices": [{"message": {"content": "最终回复"}}]},
        ]
        calls = {"n": 0}

        def fake_post(*_args, **_kwargs):
            payload = responses[min(calls["n"], len(responses) - 1)]
            calls["n"] += 1
            return FakeResponse(payload)

        original = llm_module.requests.post
        llm_module.requests.post = fake_post
        try:
            result = client.chat([{"role": "user", "content": "报价"}], retries=0,
                                 tools=[{"type": "function", "function": {"name": "price_lookup"}}],
                                 executor=lambda name, args: "ok")
        finally:
            llm_module.requests.post = original

        self.assertEqual(result.error, None, "工具调用不能占满网络重试预算")
        self.assertEqual(result.text, "最终回复")
        self.assertEqual(result.tool_calls, 1)


class TestOcrBackendSwitch(unittest.TestCase):
    def test_paddle_backend_falls_back_to_rapidocr(self) -> None:
        from wxreply.perception.ocr import OcrEngine

        engine = OcrEngine(backend="paddle")
        self.assertTrue(engine.available, "没装 PaddleOCR 时应自动退回 RapidOCR")
        self.assertIn("PaddleOCR", engine.load_error or "")


class TestWeKnoraAdapter(unittest.TestCase):
    def test_parse_common_payloads(self) -> None:
        from wxreply.brain.kb_weknora import WeKnoraClient

        hits = WeKnoraClient.parse_hits({"data": [
            {"content": "FT-120 售价 2.9-3.3 美金", "source": "price.md", "score": 0.8},
            "纯文本片段",
        ]})
        # 0 分片段不能当"有依据"：有正分命中时，0 分会被丢掉
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].source, "price.md")

    def test_official_envelope_and_fields(self) -> None:
        from wxreply.brain.kb_weknora import WeKnoraClient

        payload = {
            "success": True,
            "data": [{
                "id": "chunk-00000001",
                "content": "偏远地区（西藏/新疆）走邮政小包，运费另计。",
                "knowledge_id": "knowledge-00000001",
                "knowledge_base_id": "kb-00000001",
                "knowledge_title": "配送说明",
                "knowledge_filename": "shipping.md",
                "score": 0.95,
                "metadata": {"model_score": 0.95},
            }],
            "meta": {"rerank": {"applied": True}},
        }
        hits = WeKnoraClient.parse_hits(payload)
        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertEqual(hit.text[:4], "偏远地区")
        self.assertEqual(hit.source, "shipping.md")
        self.assertEqual(hit.title, "配送说明")
        self.assertEqual(hit.chunk_id, "chunk-00000001")
        self.assertEqual(hit.kb_id, "kb-00000001")
        self.assertAlmostEqual(hit.score, 0.95, places=3)

    def test_request_contract(self) -> None:
        from wxreply.brain.kb_weknora import WeKnoraClient

        client = WeKnoraClient("http://127.0.0.1:8080/", "sk-test",
                               kb_ids=["kb-1", "kb-2"], tenant_id="tenant-1")
        self.assertEqual(client.search_path, "/api/v1/knowledge-search")
        headers = client.headers()
        self.assertEqual(headers.get("X-API-Key"), "sk-test", "官方机器接口用 X-API-Key")
        self.assertNotIn("Authorization", headers)
        self.assertEqual(headers.get("X-Tenant-ID"), "tenant-1")
        body = client.payload("部署要求", 5)
        self.assertEqual(body["query"], "部署要求")
        self.assertEqual(body["match_count"], 5, "官方字段是 match_count，不是 top_k")
        self.assertEqual(body["knowledge_base_ids"], ["kb-1", "kb-2"])

    def test_search_failure_returns_empty(self) -> None:
        from wxreply.brain.kb_weknora import WeKnoraClient

        client = WeKnoraClient("http://127.0.0.1:9")
        self.assertEqual(client.search("报价"), [])
        self.assertTrue(client.last_error)


class TestWeKnoraHttp(unittest.TestCase):
    """按官方契约发请求、按官方信封收结果（离线回放，不连真实服务）。"""

    def _client(self, monkeypatch_response):
        import wxreply.brain.kb_weknora as kb_module

        captured = {}

        class FakeResponse:
            def __init__(self, status, payload):
                self.status_code = status
                self._payload = payload

            def json(self):
                return self._payload

        def fake_post(url, json=None, headers=None, timeout=None):
            captured.update(url=url, json=json, headers=headers)
            return monkeypatch_response(FakeResponse)

        original = kb_module.requests.post
        kb_module.requests.post = fake_post
        self.addCleanup(lambda: setattr(kb_module.requests, "post", original))
        return captured

    def test_round_trip(self) -> None:
        captured = self._client(lambda Fake: Fake(200, {
            "success": True,
            "data": [{
                "id": "chunk-1",
                "content": "偏远地区可以配送，运费另计。",
                "knowledge_id": "knowledge-1",
                "knowledge_base_id": "kb-1",
                "knowledge_title": "配送说明",
                "knowledge_filename": "shipping.md",
                "score": 0.95,
                "metadata": {"model_score": 0.95},
            }],
            "meta": {"rerank": {"applied": True}},
        }))
        from wxreply.brain.kb_weknora import WeKnoraClient

        client = WeKnoraClient("http://kb.local/", "sk-1", kb_ids=["kb-1"])
        hits = client.search("西藏 配送", top_k=4)
        self.assertEqual(captured["url"], "http://kb.local/api/v1/knowledge-search")
        self.assertEqual(captured["headers"]["X-API-Key"], "sk-1")
        self.assertEqual(captured["json"]["match_count"], 4)
        self.assertEqual(captured["json"]["knowledge_base_ids"], ["kb-1"])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].source, "shipping.md")
        self.assertEqual(client.last_error, "")

    def test_401_is_reported(self) -> None:
        self._client(lambda Fake: Fake(401, {
            "success": False, "error": {"code": 1001, "message": "unauthorized"}}))
        from wxreply.brain.kb_weknora import WeKnoraClient

        client = WeKnoraClient("http://kb.local", "bad")
        self.assertEqual(client.search("配送"), [])
        self.assertIn("API Key", client.last_error)
        self.assertIn("401", client.last_error)

    def test_success_false_body_is_error(self) -> None:
        self._client(lambda Fake: Fake(200, {
            "success": False, "error": {"code": 1002, "message": "forbidden"}}))
        from wxreply.brain.kb_weknora import WeKnoraClient

        client = WeKnoraClient("http://kb.local", "sk-1")
        self.assertEqual(client.search("配送"), [])
        self.assertTrue(client.last_error)

    def test_zero_score_hits_dropped(self) -> None:
        from wxreply.brain.kb_weknora import WeKnoraClient

        hits = WeKnoraClient.parse_hits({"success": True, "data": [
            {"content": "相关片段", "score": 0.9},
            {"content": "凑数片段", "score": 0.0},
        ]})
        self.assertEqual([hit.text for hit in hits], ["相关片段"])


class TestKnowledgeTriage(unittest.TestCase):
    def test_price_question_without_evidence_goes_to_human(self) -> None:
        from wxreply.agent import Agent
        from wxreply.platform.mock import MockPlatform
        from tests.test_reliability import make_config

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            cfg.kb_dir = os.path.join(tmp, "empty_kb")
            os.makedirs(cfg.kb_dir, exist_ok=True)
            cfg.tools.price_file = os.path.join(tmp, "missing_prices.json")
            platform = MockPlatform()
            platform.current = 0
            # 已点名型号但库里没有：不能"问一下型号"糊过去，必须转人工
            platform.sessions[0].messages = [{"dir": "in", "kind": "text",
                                              "text": "ZX-999 多少钱？"}]
            agent = Agent(cfg, platform, dry_run=True)
            items = platform.read_incoming_items()
            image = platform.capture()
            regions = agent.layout.regions(image.width, image.height)
            handled = agent._respond("张三", items, image, regions, False)
            self.assertFalse(handled)
            self.assertTrue(agent.handoff.is_taken("张三"),
                            "没有可靠依据的价格问题必须转人工")
            self.assertNotIn("张三", agent._pending)
            agent.close()

    def test_missing_model_asks_instead_of_quoting(self) -> None:
        from wxreply.agent import Agent
        from wxreply.platform.mock import MockPlatform
        from tests.test_reliability import make_config

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cfg = make_config(tmp)
            cfg.kb_dir = os.path.join(tmp, "empty_kb")
            os.makedirs(cfg.kb_dir, exist_ok=True)
            cfg.tools.price_file = os.path.join(tmp, "missing_prices.json")
            platform = MockPlatform()
            platform.current = 0
            platform.sessions[0].messages = [{"dir": "in", "kind": "text",
                                              "text": "这个多少钱？"}]
            agent = Agent(cfg, platform, dry_run=True)
            called = {"n": 0}

            def fake_generate(*_a, **_k):
                called["n"] += 1
                return ("报价 5 美金", "fake", 1)

            agent._generate = fake_generate
            items = platform.read_incoming_items()
            image = platform.capture()
            regions = agent.layout.regions(image.width, image.height)
            agent._respond("张三", items, image, regions, False)
            self.assertEqual(called["n"], 0, "没有型号时不能调用模型编报价")
            draft = agent.pending_for("张三")
            self.assertIsNotNone(draft)
            self.assertIn("哪一款", draft["reply"])
            self.assertNotIn("美金", draft["reply"])
            agent.close()


class TestExternalChannelsDisabled(unittest.TestCase):
    def test_line_channel_not_registered_by_default(self) -> None:
        configs = [
            ChannelConfig(id="wechat", type="wechat", enabled=True),
            ChannelConfig(id="line", type="line", enabled=True,
                          options={"endpoint": "https://example.invalid", "token": "x"}),
        ]
        registry = build_registry(configs, agent=None)
        self.assertIsNone(registry.get("line"), "首版外部通道默认关闭")
        self.assertIsNone(registry.get("wechat"), "没有 agent 时不注册微信通道")

    def test_external_channel_requires_opt_in(self) -> None:
        os.environ["WX_ENABLE_EXTERNAL_CHANNELS"] = "1"
        try:
            configs = [ChannelConfig(id="line", type="line", enabled=True,
                                     options={"endpoint": "https://example.invalid", "token": "x"})]
            registry = build_registry(configs, agent=None)
            self.assertIsNotNone(registry.get("line"))
        finally:
            os.environ.pop("WX_ENABLE_EXTERNAL_CHANNELS", None)


if __name__ == "__main__":
    unittest.main()
